"""Stage 6 — Needleman–Wunsch pairwise alignment + null model."""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .config import Config
from .utils import save_array

logger = logging.getLogger(__name__)


class PipelineGatingError(Exception):
    """Raised when the null model gating test fails."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def align_all(
    cfg: Config,
    syllables: list[dict[str, Any]],
    vocabulary: dict[str, Any],
    substitution: dict[str, Any],
    run_log: Any = None,
    dataset: str = "genus",
) -> dict[str, Any]:
    """Run all pairwise alignments and null model validation.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    syllables:
        Feature-augmented and labelled syllable records.
    vocabulary:
        Vocabulary dict from Stage 4.
    substitution:
        Substitution dict from Stage 5.
    run_log:
        Optional PipelineRunLog.
    dataset:
        ``'genus'`` or ``'within_species'``.

    Returns
    -------
    dict
        Keys:
        - ``distance_matrix``: ``(n × n)`` acoustic distance array
        - ``species_names``: ordered list of species/cell names
        - ``alignment_scores``: raw scores before normalisation
        - ``null_model_result``: null model test results
        - ``bootstrap_cis``: 95% CI per pair from recording bootstrap
        - ``noise_floors``: strict and lenient noise floor estimates

    Raises
    ------
    PipelineGatingError
        If the null model binomial test is not significant.
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    cluster_letters = vocabulary["cluster_letters"]
    labels_array = vocabulary["labels"]

    # Build song strings
    if dataset == "within_species":
        entities, song_strings = _build_cell_strings(syllables, labels_array, cluster_letters, cfg)
    else:
        entities, song_strings = _build_species_strings(syllables, labels_array, cluster_letters)

    logger.info("Aligning %d entities", len(entities))

    # Two parallel analyses: all subspecies and nominate only
    nominate_strings = _filter_nominate(entities, song_strings, syllables, labels_array, cluster_letters)

    aligner = _make_aligner(substitution)

    # Primary analysis
    scores_all, D_all = _pairwise_distance_matrix(song_strings, aligner)
    # Nominate subspecies analysis
    if nominate_strings and len(nominate_strings) >= 3:
        _, D_nom = _pairwise_distance_matrix(nominate_strings, aligner)
    else:
        D_nom = None

    # Null model
    null_result = _run_null_model(
        D_all, song_strings, aligner, cfg, run_log
    )

    _make_figure_3(null_result, cfg)

    if not null_result["passed"]:
        diag = {
            "error": "PipelineGatingError",
            "stage": "stage6_align",
            "null_model_result": null_result,
        }
        raise PipelineGatingError(
            "Null model gating test failed: per-pair sequential structure not detected",
            diag,
        )

    # Recording-level bootstrap
    bootstrap_cis = _recording_bootstrap(
        syllables, labels_array, cluster_letters, cfg, aligner,
        entities, song_strings, dataset
    )

    # Noise floors
    noise_floors = _compute_noise_floors(syllables, labels_array, cluster_letters, aligner, cfg)

    # Save distance matrix
    dist_path = cfg.data_path / "distances" / f"acoustic_distance_{dataset}.pkl"
    save_array(
        {"D": D_all, "entities": entities},
        dist_path,
        config_hash=cfg.hash,
        random_seed=cfg.random_seed,
        n_species=len(entities),
        extra={"dataset": dataset},
    )

    return {
        "distance_matrix": D_all,
        "species_names": entities,
        "alignment_scores": scores_all,
        "null_model_result": null_result,
        "bootstrap_cis": bootstrap_cis,
        "noise_floors": noise_floors,
        "nominate_only_distance_matrix": D_nom,
        "nominate_entities": sorted(nominate_strings.keys()) if nominate_strings else [],
    }


# ---------------------------------------------------------------------------
# Song string construction
# ---------------------------------------------------------------------------

def _build_species_strings(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
) -> tuple[list[str], dict[str, str]]:
    """Build concatenated song string per species.

    Parameters
    ----------
    syllables:
        Syllable records.
    labels_array:
        Cluster label per syllable.
    cluster_letters:
        Label-to-letter mapping.
    """
    species_seqs: dict[str, list[str]] = {}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        letter = cluster_letters[lb]
        sp = syl.get("species", "unknown")
        species_seqs.setdefault(sp, []).append(letter)

    sorted_species = sorted(species_seqs.keys())
    strings = {"".join(species_seqs[sp]): sp for sp in sorted_species}
    return sorted_species, {sp: "".join(species_seqs[sp]) for sp in sorted_species}


def _build_cell_strings(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> tuple[list[str], dict[str, str]]:
    """Build per-geographic-cell song strings for within-species analysis.

    Strings are truncated to the shortest qualifying cell length
    (rounded to multiple of 10 syllables).

    Parameters
    ----------
    syllables:
        Syllable records with ``lat`` and ``lon``.
    labels_array:
        Cluster label per syllable.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    rng = random.Random(cfg.random_seed)
    cell_seqs: dict[str, list[str]] = {}
    half = cfg.cell_size_degrees / 2

    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        lat = syl.get("lat")
        lon = syl.get("lon")
        if lat is None or lon is None:
            continue
        cell_lat = round(int(lat / cfg.cell_size_degrees) * cfg.cell_size_degrees, 4)
        cell_lon = round(int(lon / cfg.cell_size_degrees) * cfg.cell_size_degrees, 4)
        cell_id = f"{cell_lat:.2f}_{cell_lon:.2f}"
        letter = cluster_letters[lb]
        cell_seqs.setdefault(cell_id, []).append(letter)

    # Keep only qualifying cells
    qualified = {
        cell: seq for cell, seq in cell_seqs.items()
        if len(seq) >= cfg.min_recordings_per_cell
    }

    if not qualified:
        return [], {}

    # Truncate to shortest cell (multiple of 10)
    min_len = min(len(seq) for seq in qualified.values())
    truncate_to = (min_len // 10) * 10
    if truncate_to == 0:
        truncate_to = min_len

    result: dict[str, str] = {}
    for cell, seq in qualified.items():
        if len(seq) > truncate_to:
            sampled = rng.sample(seq, truncate_to)
        else:
            sampled = seq[:truncate_to]
        result[cell] = "".join(sampled)

    sorted_cells = sorted(result.keys())
    return sorted_cells, result


def _filter_nominate(
    entities: list[str],
    song_strings: dict[str, str],
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
) -> dict[str, str]:
    """Build nominate-subspecies-only song strings.

    Parameters
    ----------
    entities:
        List of species/cell names.
    song_strings:
        Full song strings.
    syllables:
        Syllable records with ``subspecies`` field.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    """
    nominate_seqs: dict[str, list[str]] = {}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        ssp = str(syl.get("subspecies", "")).strip()
        if ssp and ssp.lower() not in ("", "none", "nan"):
            continue  # skip non-nominate
        sp = syl.get("species", "unknown")
        letter = cluster_letters[lb]
        nominate_seqs.setdefault(sp, []).append(letter)

    return {sp: "".join(seqs) for sp, seqs in nominate_seqs.items() if len(seqs) >= 5}


# ---------------------------------------------------------------------------
# Pairwise alignment
# ---------------------------------------------------------------------------

def _make_aligner(substitution: dict[str, Any]) -> Any:
    """Configure a BioPython PairwiseAligner with the acoustic substitution matrix.

    Parameters
    ----------
    substitution:
        Substitution dict from Stage 5.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = substitution["gap_open"]
    aligner.extend_gap_score = substitution["gap_extend"]

    # Build substitution matrix as dict
    cluster_letters = substitution["cluster_letters"]
    label_order = substitution["label_order"]
    primary_matrix = substitution["primary_matrix"]
    letters = [cluster_letters[lb] for lb in label_order]

    # Build as 2-D ndarray — substitution_matrices.Array requires alphabet as str or tuple.
    n = len(letters)
    array_2d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            array_2d[i, j] = float(primary_matrix[i, j])

    # Single-char alphabet → str; multi-char → tuple (for >26 cluster vocabularies)
    if all(len(lt) == 1 for lt in letters):
        alphabet: str | tuple = "".join(letters)
    else:
        alphabet = tuple(letters)

    aligner.substitution_matrix = substitution_matrices.Array(alphabet, 2, data=array_2d)
    return aligner


def _pairwise_distance_matrix(
    song_strings: dict[str, str],
    aligner: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all n(n−1)/2 pairwise alignment distances.

    Parameters
    ----------
    song_strings:
        Dict mapping entity name → song string.
    aligner:
        Configured BioPython PairwiseAligner.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(scores, D)`` where ``scores[i,j]`` is the raw alignment score
        and ``D[i,j] = 1 - S(i,j) / max(S(i,i), S(j,j))``.
    """
    entities = sorted(song_strings.keys())
    n = len(entities)
    scores = np.zeros((n, n))

    for i, sp_i in enumerate(entities):
        scores[i, i] = _score_pair(song_strings[sp_i], song_strings[sp_i], aligner)
        for j in range(i + 1, n):
            s = _score_pair(song_strings[sp_i], song_strings[entities[j]], aligner)
            scores[i, j] = scores[j, i] = s

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(scores[i, i], scores[j, j])
            if denom <= 0:
                d = 1.0
            else:
                d = 1.0 - scores[i, j] / denom
            d = float(np.clip(d, 0.0, 1.0))
            D[i, j] = D[j, i] = d

    return scores, D


def _score_pair(seq1: str, seq2: str, aligner: Any) -> float:
    """Align two sequences and return the alignment score.

    Parameters
    ----------
    seq1, seq2:
        Letter sequences.
    aligner:
        Configured aligner.
    """
    if not seq1 or not seq2:
        return 0.0
    try:
        return float(aligner.score(seq1, seq2))
    except Exception as exc:
        logger.debug("Alignment error: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------

def _run_null_model(
    D: np.ndarray,
    song_strings: dict[str, str],
    aligner: Any,
    cfg: Config,
    run_log: Any,
) -> dict[str, Any]:
    """Run per-pair null model validation.

    For each pair (i, j), shuffle each species' string ``cfg.null_model_permutations``
    times (preserving syllable-type frequencies). Tests whether observed
    distances are above the 97.5th percentile of the null more often than expected.

    Parameters
    ----------
    D:
        Observed distance matrix.
    song_strings:
        Per-species song strings.
    aligner:
        Configured aligner.
    cfg:
        Pipeline configuration.
    run_log:
        Optional PipelineRunLog.
    """
    rng = np.random.RandomState(cfg.random_seed)
    entities = sorted(song_strings.keys())
    n = len(entities)
    n_perms = cfg.null_model_permutations
    upper_tail = cfg.null_model_upper_tail

    above_tail: list[bool] = []
    percentile_ranks: list[float] = []

    for i in range(n):
        # Compute re-scored self-alignment with shuffled string for null
        s_i = song_strings[entities[i]]
        score_ii = _score_pair(s_i, s_i, aligner)

        for j in range(i + 1, n):
            s_j = song_strings[entities[j]]
            score_jj = _score_pair(s_j, s_j, aligner)
            obs_d = float(D[i, j])

            null_distances = []
            for _ in range(n_perms):
                shuffled_i = _shuffle_string(s_i, rng)
                shuffled_j = _shuffle_string(s_j, rng)
                null_score = _score_pair(shuffled_i, shuffled_j, aligner)
                denom = max(score_ii, score_jj)
                if denom <= 0:
                    null_d = 1.0
                else:
                    null_d = float(np.clip(1.0 - null_score / denom, 0.0, 1.0))
                null_distances.append(null_d)

            pct_rank = float(np.mean([nd <= obs_d for nd in null_distances]))
            percentile_ranks.append(pct_rank)
            above_tail.append(pct_rank > upper_tail)

    n_pairs = len(above_tail)
    n_above = sum(above_tail)
    expected_p = 1.0 - upper_tail  # 0.025

    btest = binomtest(n_above, n_pairs, expected_p, alternative="greater")
    p_value = float(btest.pvalue)
    proportion_above = float(n_above / n_pairs) if n_pairs > 0 else 0.0

    passed = p_value < cfg.null_model_binomial_alpha
    logger.info(
        "Null model: %d/%d pairs above %.1f%% tail (proportion=%.3f, binomial p=%.4f, passed=%s)",
        n_above, n_pairs, upper_tail * 100, proportion_above, p_value, passed,
    )

    if run_log is not None:
        run_log.record_threshold(
            name="null_model_binomial_p",
            value=p_value,
            threshold=cfg.null_model_binomial_alpha,
            passed=passed,
            stage="stage6_align",
        )

    return {
        "passed": passed,
        "n_pairs": n_pairs,
        "n_above_tail": n_above,
        "proportion_above": proportion_above,
        "binomial_p": p_value,
        "threshold_alpha": cfg.null_model_binomial_alpha,
        "percentile_ranks": percentile_ranks,
    }


def _shuffle_string(seq: str, rng: np.random.RandomState) -> str:
    """Shuffle a song string, preserving syllable-type frequencies.

    Parameters
    ----------
    seq:
        Input letter string.
    rng:
        NumPy random state.
    """
    arr = list(seq)
    rng.shuffle(arr)
    return "".join(arr)


# ---------------------------------------------------------------------------
# Recording-level bootstrap
# ---------------------------------------------------------------------------

def _recording_bootstrap(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
    aligner: Any,
    entities: list[str],
    song_strings: dict[str, str],
    dataset: str,
) -> dict[str, Any]:
    """Bootstrap 95% CIs on acoustic distances by resampling recordings.

    Parameters
    ----------
    syllables:
        Syllable records.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    aligner:
        Configured aligner.
    entities:
        Ordered entity names.
    song_strings:
        Observed song strings.
    dataset:
        ``'genus'`` or ``'within_species'``.
    """
    rng = random.Random(cfg.random_seed + 1)
    n = len(entities)
    bootstrap_distances = np.zeros((cfg.recording_bootstrap_n, n, n))

    # Group syllables by entity (species or cell) and recording
    entity_recordings: dict[str, dict[str, list[int]]] = {}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        sp = syl.get("species", "unknown")
        rec_id = syl.get("xc_id", f"rec_{i}")
        entity_recordings.setdefault(sp, {}).setdefault(rec_id, []).append(i)

    for b in range(cfg.recording_bootstrap_n):
        boot_strings: dict[str, str] = {}
        for sp, recordings in entity_recordings.items():
            rec_list = list(recordings.keys())
            sampled = rng.choices(rec_list, k=len(rec_list))
            seq_parts = []
            for rec_id in sampled:
                for idx in recordings[rec_id]:
                    seq_parts.append(cluster_letters[labels_array[idx]])
            boot_strings[sp] = "".join(seq_parts)

        available = sorted(boot_strings.keys())
        _, D_boot = _pairwise_distance_matrix(
            {sp: boot_strings[sp] for sp in available if sp in entities},
            aligner,
        )
        for i, e_i in enumerate(entities):
            for j, e_j in enumerate(entities):
                if e_i in available and e_j in available:
                    ai = available.index(e_i)
                    aj = available.index(e_j)
                    bootstrap_distances[b, i, j] = D_boot[ai, aj]

    lo = np.percentile(bootstrap_distances, 2.5, axis=0)
    hi = np.percentile(bootstrap_distances, 97.5, axis=0)
    return {"ci_lower": lo, "ci_upper": hi}


# ---------------------------------------------------------------------------
# Noise floors
# ---------------------------------------------------------------------------

def _compute_noise_floors(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    aligner: Any,
    cfg: Config,
) -> dict[str, Any]:
    """Estimate within- and between-individual noise floors.

    Parameters
    ----------
    syllables:
        Syllable records.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    aligner:
        Configured aligner.
    cfg:
        Pipeline configuration.
    """
    # Strict: same individual, same day (same xc_id proxy)
    # Lenient: same species, different xc_id

    by_species_rec: dict[str, dict[str, list[str]]] = {}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        sp = syl.get("species", "unknown")
        rec_id = syl.get("xc_id", "")
        letter = cluster_letters[lb]
        by_species_rec.setdefault(sp, {}).setdefault(rec_id, []).append(letter)

    strict_dists: list[float] = []
    lenient_dists: list[float] = []

    for sp, recs in by_species_rec.items():
        rec_ids = sorted(recs.keys())
        strings = {"".join(recs[r]): r for r in rec_ids if recs[r]}
        string_list = [("".join(recs[r]), r) for r in rec_ids if recs[r]]

        # Lenient: between different recordings of same species
        for i in range(len(string_list)):
            for j in range(i + 1, len(string_list)):
                s_i, r_i = string_list[i]
                s_j, r_j = string_list[j]
                sc_i = _score_pair(s_i, s_i, aligner)
                sc_j = _score_pair(s_j, s_j, aligner)
                sc_ij = _score_pair(s_i, s_j, aligner)
                denom = max(sc_i, sc_j)
                if denom > 0:
                    d = float(np.clip(1.0 - sc_ij / denom, 0.0, 1.0))
                    lenient_dists.append(d)

    result = {
        "strict_mean": None,
        "strict_p95": None,
        "lenient_mean": float(np.mean(lenient_dists)) if lenient_dists else None,
        "lenient_p95": float(np.percentile(lenient_dists, 95)) if lenient_dists else None,
    }
    return result


# ---------------------------------------------------------------------------
# Figure 3 — null model percentile rank histogram
# ---------------------------------------------------------------------------

def _make_figure_3(null_result: dict[str, Any], cfg: Config) -> None:
    """Histogram of per-pair percentile ranks overlaid on the uniform null (Figure 3).

    Parameters
    ----------
    null_result:
        Output of :func:`_run_null_model`, containing ``percentile_ranks``.
    cfg:
        Pipeline configuration.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available; Figure 3 not generated")
        return

    ranks = null_result.get("percentile_ranks", [])
    if not ranks:
        logger.warning("No percentile ranks available; Figure 3 not generated")
        return

    ranks_arr = np.array(ranks)
    n_bins = 20
    bin_width = 1.0 / n_bins
    uniform_density = 1.0  # density = 1 for Uniform[0,1]

    binom_p = null_result.get("binomial_p", float("nan"))
    prop_above = null_result.get("proportion_above", float("nan"))
    n_pairs = null_result.get("n_pairs", len(ranks))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ranks_arr,
        nbinsx=n_bins,
        histnorm="probability density",
        name="Observed percentile ranks",
        marker_color="#3498db",
        opacity=0.75,
    ))
    fig.add_hline(
        y=uniform_density,
        line_dash="dash",
        line_color="red",
        annotation_text="Uniform null",
        annotation_position="top right",
    )
    fig.add_vline(
        x=null_result.get("threshold_alpha", 0.025),
        line_dash="dot",
        line_color="orange",
        annotation_text="97.5th-pct tail",
        annotation_position="top left",
    )
    fig.update_layout(
        title=(
            f"Figure 3: Null model percentile rank distribution "
            f"(n={n_pairs} pairs, binomial p={binom_p:.4f}, "
            f"prop above tail={prop_above:.3f})"
        ),
        xaxis_title="Percentile rank of observed distance within null distribution",
        yaxis_title="Density",
        bargap=0.05,
    )

    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(fig_dir / "figure_3.html"))
    logger.info("Figure 3 saved")
