"""Stage 5 — Empirically derived substitution matrix construction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .config import Config
from .utils import save_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_substitution_matrix(
    cfg: Config,
    vocabulary: dict[str, Any],
    syllables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build acoustically derived substitution matrices.

    Computes three matrices (at the 50th, 75th, and 95th percentile mismatch
    penalties) and selects the primary (95th percentile) for downstream use.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    vocabulary:
        Vocabulary dict from :func:`~syrinx.vocabulary.build_vocabulary`.
    syllables:
        Feature-augmented syllable records. Required when
        ``cfg.use_temporal_features`` is True.

    Returns
    -------
    dict
        Keys:
        - ``matrices``: dict mapping percentile → ``(n × n)`` ndarray
        - ``primary_matrix``: the selected primary matrix
        - ``cluster_letters``: label mapping from vocabulary
        - ``match_scores``: per-cluster match scores
        - ``gap_open``: selected gap open penalty
        - ``gap_extend``: selected gap extend penalty
        - ``label_order``: list of cluster integer labels in matrix row/col order
    """
    centroids = vocabulary["centroids"]
    labels_array = vocabulary["labels"]
    cluster_letters = vocabulary["cluster_letters"]
    unique_labels = sorted(cluster_letters.keys())
    n_clusters = len(unique_labels)

    X = (
        np.vstack([s["features"] for s in syllables]).astype(np.float64)
        if syllables else None
    )

    logger.info("Building substitution matrix for %d clusters", n_clusters)

    if cfg.use_temporal_features and syllables:
        # Temporal mode: use mean intra-cluster DTW distance for match scores and
        # mean inter-cluster DTW centroid distance for mismatch penalties.
        logger.info("Temporal mode: using DTW distances for substitution matrix")
        match_scores, dtw_centroids = _compute_match_scores_dtw(
            labels_array, unique_labels, syllables
        )
        matrices: dict[int, np.ndarray] = {}
        for pct in cfg.mismatch_percentiles:
            mat = _build_matrix_dtw(dtw_centroids, match_scores, unique_labels, percentile=pct)
            matrices[pct] = mat
            logger.debug("Built DTW substitution matrix at %dth percentile", pct)
    else:
        # Default: Euclidean centroid distances
        match_scores = _compute_match_scores(labels_array, unique_labels, X, centroids)
        matrices: dict[int, np.ndarray] = {}
        for pct in cfg.mismatch_percentiles:
            mat = _build_matrix(centroids, match_scores, unique_labels, percentile=pct)
            matrices[pct] = mat
            logger.debug("Built substitution matrix at %dth percentile", pct)

    primary_matrix = matrices[cfg.primary_mismatch_percentile]

    # Gap penalty selection
    gap_open, gap_extend = _select_gap_penalties(
        cfg, vocabulary, syllables, primary_matrix, unique_labels, cluster_letters
    )
    logger.info("Selected gap penalties: open=%.2f, extend=%.2f", gap_open, gap_extend)

    return {
        "matrices": matrices,
        "primary_matrix": primary_matrix,
        "cluster_letters": cluster_letters,
        "match_scores": match_scores,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "label_order": unique_labels,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def _compute_match_scores(
    labels_array: np.ndarray,
    unique_labels: list[int],
    X: np.ndarray | None,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute per-cluster match scores from mean within-cluster Euclidean distance.

    Parameters
    ----------
    labels_array:
        Full label array from HDBSCAN.
    unique_labels:
        Sorted cluster integer labels.
    X:
        Feature matrix (may be None if only centroids available).
    centroids:
        Centroid array.

    Returns
    -------
    np.ndarray
        Shape ``(n_clusters,)`` — positive match scores (negated mean distance).
    """
    match_scores = np.zeros(len(unique_labels))
    for i, lb in enumerate(unique_labels):
        if X is not None:
            pts = X[labels_array == lb]
            if len(pts) > 1:
                pw = cdist(pts, pts, metric="euclidean")
                mean_dist = pw[np.triu_indices(len(pts), k=1)].mean()
            else:
                mean_dist = 0.0
        else:
            mean_dist = 0.0
        match_scores[i] = -mean_dist  # negate → positive match score
    return match_scores


def _build_matrix(
    centroids: np.ndarray,
    match_scores: np.ndarray,
    unique_labels: list[int],
    percentile: int,
) -> np.ndarray:
    """Construct a single substitution matrix at the given mismatch percentile.

    Parameters
    ----------
    centroids:
        Cluster centroid array, shape ``(n_clusters, n_features)``.
    match_scores:
        Per-cluster match scores, shape ``(n_clusters,)``.
    unique_labels:
        Sorted cluster labels (determines row/column order).
    percentile:
        Percentile of between-cluster distances to use as max mismatch penalty.

    Returns
    -------
    np.ndarray
        Symmetric substitution matrix of shape ``(n_clusters, n_clusters)``.
    """
    n = len(unique_labels)
    pairwise = cdist(centroids, centroids, metric="euclidean")

    # Upper triangle of between-cluster distances
    off_diag = pairwise[np.triu_indices(n, k=1)]
    max_penalty = float(np.percentile(off_diag, percentile))

    # Normalise between-cluster distances to [0, 1]
    max_dist = float(pairwise.max()) if pairwise.max() > 0 else 1.0
    normalised = pairwise / max_dist

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = match_scores[i]
            else:
                # Linear scale from 0 to max_penalty
                mat[i, j] = -(normalised[i, j] * max_penalty)

    return mat


# ---------------------------------------------------------------------------
# DTW-based matrix construction (temporal mode)
# ---------------------------------------------------------------------------

def _compute_match_scores_dtw(
    labels_array: np.ndarray,
    unique_labels: list[int],
    syllables: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cluster match scores and DTW-distance centroids.

    Match score for cluster k = −mean pairwise DTW distance between syllable
    clips in that cluster (uses the ``mfcc_temporal`` matrices stored during
    feature extraction).

    Returns
    -------
    match_scores : np.ndarray, shape (n_clusters,)
    dtw_centroids : np.ndarray, shape (n_clusters, n_clusters)
        Mean DTW distance between every pair of clusters (upper-triangle used
        for mismatch penalty scaling).
    """
    from dtaidistance import dtw_ndim

    _MAX_PAIRS = 50  # cap expensive pairwise DTW within each cluster

    n = len(unique_labels)
    match_scores = np.zeros(n)
    temporal_by_label: dict[int, list[np.ndarray]] = {}

    for i, syl in enumerate(syllables):
        lb = int(labels_array[i])
        if lb == -1:
            continue
        mt = syl.get("mfcc_temporal")
        if mt is not None:
            temporal_by_label.setdefault(lb, []).append(np.array(mt, dtype=np.float64))

    for i, lb in enumerate(unique_labels):
        clips = temporal_by_label.get(lb, [])
        if len(clips) < 2:
            match_scores[i] = 0.0
            continue
        sampled = clips[:_MAX_PAIRS]
        dists = []
        for a in range(len(sampled)):
            for b in range(a + 1, len(sampled)):
                try:
                    d = dtw_ndim.distance(sampled[a].T, sampled[b].T)
                    dists.append(d)
                except Exception:
                    pass
        match_scores[i] = -float(np.mean(dists)) if dists else 0.0

    # Inter-cluster DTW using one representative clip per cluster
    dtw_centroids = np.zeros((n, n))
    reps = []
    for lb in unique_labels:
        clips = temporal_by_label.get(lb, [])
        reps.append(clips[0] if clips else np.zeros((26, 1)))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = dtw_ndim.distance(reps[i].T, reps[j].T)
            except Exception:
                d = 0.0
            dtw_centroids[i, j] = d
            dtw_centroids[j, i] = d

    return match_scores, dtw_centroids


def _build_matrix_dtw(
    dtw_centroids: np.ndarray,
    match_scores: np.ndarray,
    unique_labels: list[int],
    percentile: int,
) -> np.ndarray:
    """Build a substitution matrix using DTW inter-cluster distances."""
    n = len(unique_labels)
    off_diag = dtw_centroids[np.triu_indices(n, k=1)]
    max_penalty = float(np.percentile(off_diag, percentile)) if off_diag.size > 0 else 1.0
    max_dist = float(dtw_centroids.max()) if dtw_centroids.max() > 0 else 1.0
    normalised = dtw_centroids / max_dist

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = match_scores[i]
            else:
                mat[i, j] = -(normalised[i, j] * max_penalty)
    return mat


# ---------------------------------------------------------------------------
# Gap penalty selection
# ---------------------------------------------------------------------------

def _select_gap_penalties(
    cfg: Config,
    vocabulary: dict[str, Any],
    syllables: list[dict[str, Any]] | None,
    sub_matrix: np.ndarray,
    unique_labels: list[int],
    cluster_letters: dict[int, str],
) -> tuple[float, float]:
    """Grid search for gap penalties that maximise within- vs between-species separation.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    vocabulary:
        Vocabulary dict.
    syllables:
        Feature-augmented syllable records.
    sub_matrix:
        Primary substitution matrix.
    unique_labels:
        Cluster label order.
    cluster_letters:
        Mapping from cluster int to letter.

    Returns
    -------
    tuple[float, float]
        ``(gap_open, gap_extend)`` penalties.
    """
    if syllables is None or len(syllables) < 10:
        logger.warning("Insufficient syllables for gap penalty selection; using defaults")
        return cfg.gap_open_grid[0], cfg.gap_extend_grid[0]

    # Build song strings for held-out species
    species_strings = _build_song_strings_for_gap_search(
        syllables, vocabulary["labels"], cluster_letters, cfg
    )
    if len(species_strings) < 4:
        return cfg.gap_open_grid[0], cfg.gap_extend_grid[0]

    holdout = dict(list(species_strings.items())[:cfg.gap_penalty_holdout_n_species])

    best_sep = -np.inf
    best_gap_open = cfg.gap_open_grid[0]
    best_gap_extend = cfg.gap_extend_grid[0]

    letter_to_int = {v: k for k, v in cluster_letters.items()}

    for gap_open in cfg.gap_open_grid:
        for gap_extend in cfg.gap_extend_grid:
            sep = _evaluate_gap_penalty(
                holdout, sub_matrix, unique_labels, letter_to_int,
                gap_open, gap_extend
            )
            if sep > best_sep:
                best_sep = sep
                best_gap_open = gap_open
                best_gap_extend = gap_extend

    return best_gap_open, best_gap_extend


def _build_song_strings_for_gap_search(
    syllables: list[dict[str, Any]],
    labels: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> dict[str, str]:
    """Build per-species song strings for gap penalty evaluation.

    Parameters
    ----------
    syllables:
        Syllable records with labels applied.
    labels:
        Cluster label array.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    species_seqs: dict[str, list[str]] = {}
    for i, syl in enumerate(syllables):
        lb = labels[i]
        if lb == -1:
            continue
        letter = cluster_letters.get(lb, "?")
        sp = syl.get("species", "unknown")
        species_seqs.setdefault(sp, []).append(letter)

    return {sp: "".join(seq) for sp, seq in species_seqs.items() if len(seq) >= 5}


def _evaluate_gap_penalty(
    species_strings: dict[str, str],
    sub_matrix: np.ndarray,
    unique_labels: list[int],
    letter_to_int: dict[str, int],
    gap_open: float,
    gap_extend: float,
) -> float:
    """Compute separation between within- and between-species alignment scores.

    Parameters
    ----------
    species_strings:
        Per-species song strings.
    sub_matrix:
        Substitution matrix.
    unique_labels:
        Label order for matrix indexing.
    letter_to_int:
        Letter-to-cluster-int mapping.
    gap_open:
        Gap open penalty.
    gap_extend:
        Gap extend penalty.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    label_to_idx = {lb: i for i, lb in enumerate(unique_labels)}
    species = sorted(species_strings.keys())
    n = len(species)

    self_scores = []
    cross_scores = []

    aligner = PairwiseAligner()
    aligner.mode = "global"

    alphabet_list = sorted(letter_to_int.keys())
    n = len(alphabet_list)
    array_2d = np.zeros((n, n))
    for i, li in enumerate(alphabet_list):
        for j, lj in enumerate(alphabet_list):
            ii = label_to_idx[letter_to_int[li]]
            ij = label_to_idx[letter_to_int[lj]]
            array_2d[i, j] = float(sub_matrix[ii, ij])
    if all(len(lt) == 1 for lt in alphabet_list):
        alphabet_key: str | tuple = "".join(alphabet_list)
    else:
        alphabet_key = tuple(alphabet_list)
    aligner.substitution_matrix = substitution_matrices.Array(alphabet_key, 2, data=array_2d)

    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    for i in range(min(n, 10)):
        s_i = species_strings[species[i]]
        score_ii = _align_score(s_i, s_i, sub_matrix, label_to_idx, letter_to_int, aligner)
        self_scores.append(score_ii)
        for j in range(i + 1, min(n, 10)):
            s_j = species_strings[species[j]]
            score_ij = _align_score(s_i, s_j, sub_matrix, label_to_idx, letter_to_int, aligner)
            cross_scores.append(score_ij)

    if not self_scores or not cross_scores:
        return 0.0

    mean_self = np.mean(self_scores)
    mean_cross = np.mean(cross_scores)
    return float(mean_self - mean_cross)


def _align_score(
    seq1: str,
    seq2: str,
    sub_matrix: np.ndarray,
    label_to_idx: dict[int, int],
    letter_to_int: dict[str, int],
    aligner: Any,
) -> float:
    """Compute alignment score for two letter strings.

    Parameters
    ----------
    seq1, seq2:
        Letter strings.
    sub_matrix:
        Substitution matrix.
    label_to_idx:
        Mapping from cluster int to matrix row/col index.
    letter_to_int:
        Mapping from letter to cluster int.
    aligner:
        BioPython PairwiseAligner.
    """
    try:
        score = float(aligner.score(seq1, seq2))
    except Exception:
        score = 0.0
    return score
