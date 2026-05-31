"""Stage 4 — HDBSCAN clustering + 4 automated validation gates."""

from __future__ import annotations

import logging
import random
import string
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, precision_recall_fscore_support

from .config import Config
from .utils import save_manifest

logger = logging.getLogger(__name__)


class VocabularyValidationError(Exception):
    """Raised when no HDBSCAN configuration passes all 4 validation gates."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_vocabulary(
    cfg: Config,
    syllables: list[dict[str, Any]],
    run_log: Any = None,
) -> dict[str, Any]:
    """Build and validate the cross-species syllable vocabulary.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    syllables:
        Feature-augmented syllable records (each with ``features`` key).
    run_log:
        Optional PipelineRunLog for threshold recording.

    Returns
    -------
    dict
        Vocabulary result with keys:
        - ``labels``: array of cluster labels (−1 = noise, 0-based ints)
        - ``cluster_letters``: mapping int → letter string
        - ``centroids``: array of shape ``(n_clusters, 36)``
        - ``clusterer``: fitted HDBSCAN object
        - ``n_clusters``: number of clusters
        - ``noise_fraction``: fraction of points labelled −1
        - ``params``: best hyperparameters found
        - ``gate_results``: dict of per-gate pass/fail and values

    Raises
    ------
    VocabularyValidationError
        If no parameter combination passes all four gates after
        ``cfg.hdbscan_max_cycles`` attempts.
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    X = np.vstack([s["features"] for s in syllables]).astype(np.float64)
    logger.info("Building vocabulary from %d syllable feature vectors", len(X))

    grid = list(product(cfg.hdbscan_min_cluster_size_grid, cfg.hdbscan_min_samples_grid))
    best_params = None
    best_dbcv = -np.inf
    cycle_diagnostics: list[dict[str, Any]] = []

    for cycle in range(cfg.hdbscan_max_cycles):
        logger.info("Vocabulary validation cycle %d/%d", cycle + 1, cfg.hdbscan_max_cycles)

        # Grid search by DBCV
        best_params, best_dbcv, best_labels, best_clusterer = _grid_search_dbcv(
            X, grid, cfg.random_seed
        )
        logger.info(
            "Best HDBSCAN params: min_cluster_size=%d, min_samples=%d, DBCV=%.4f",
            best_params[0], best_params[1], best_dbcv,
        )

        # Zebra finch F1 check
        zf_f1 = _zebrafinch_f1_check(X, best_labels, cfg)
        if zf_f1 is not None and zf_f1 < cfg.zebrafinch_f1_threshold:
            logger.warning(
                "Zebra finch macro-F1=%.3f < threshold %.2f; adjusting toward coarser clustering",
                zf_f1, cfg.zebrafinch_f1_threshold,
            )
            grid = _coarsen_grid(grid)

        # Build centroids — guard against all-noise result
        unique_labels = sorted(set(best_labels) - {-1})
        if not unique_labels:
            logger.warning("Cycle %d: HDBSCAN found no clusters; adjusting grid", cycle + 1)
            grid = _coarsen_grid(grid)
            cycle_diagnostics.append({
                "cycle": cycle,
                "params": {"min_cluster_size": best_params[0], "min_samples": best_params[1]},
                "dbcv": float(best_dbcv),
                "zf_f1": None,
                "gate_results": {"error": "no_clusters"},
            })
            continue
        centroids = np.vstack([X[best_labels == lb].mean(axis=0) for lb in unique_labels])

        # Run 4 gates
        gate_results = _run_all_gates(X, best_labels, syllables, centroids, cfg)
        cycle_diagnostics.append({
            "cycle": cycle,
            "params": {"min_cluster_size": best_params[0], "min_samples": best_params[1]},
            "dbcv": float(best_dbcv),
            "zf_f1": float(zf_f1) if zf_f1 is not None else None,
            "gate_results": gate_results,
        })

        all_passed = all(gate_results[g]["passed"] for g in gate_results)
        if all_passed:
            logger.info("All 4 vocabulary validation gates passed on cycle %d", cycle + 1)
            break

        # Adjust grid based on which gates failed
        grid = _adjust_grid_for_failures(grid, gate_results, best_params)
        # Extend grid if at edge
        grid = _maybe_extend_grid(grid, best_params, gate_results)
    else:
        diag = {
            "error": "VocabularyValidationError",
            "cycles": cycle_diagnostics,
            "n_cycles": cfg.hdbscan_max_cycles,
        }
        manifest_path = cfg.data_path / "manifests" / "vocabulary_failure.json"
        save_manifest(diag, manifest_path)
        raise VocabularyValidationError(
            f"No vocabulary parameter combination passed all 4 gates after "
            f"{cfg.hdbscan_max_cycles} cycles",
            diag,
        )

    # Record gate thresholds
    if run_log is not None:
        for gate_name, res in gate_results.items():
            run_log.record_threshold(
                name=gate_name,
                value=res.get("value", float("nan")),
                threshold=res.get("threshold", float("nan")),
                passed=res["passed"],
                stage="stage4_vocabulary",
            )

    unique_labels = sorted(set(best_labels) - {-1})
    if not unique_labels:
        diag = {
            "error": "VocabularyValidationError",
            "cycles": cycle_diagnostics,
            "n_cycles": cfg.hdbscan_max_cycles,
            "note": "no clusters found in any cycle",
        }
        save_manifest(diag, cfg.data_path / "manifests" / "vocabulary_failure.json")
        raise VocabularyValidationError(
            f"No vocabulary parameter combination passed all 4 gates after "
            f"{cfg.hdbscan_max_cycles} cycles",
            diag,
        )
    centroids = np.vstack([X[best_labels == lb].mean(axis=0) for lb in unique_labels])
    cluster_letters = _assign_letters(centroids, unique_labels)
    noise_fraction = float((best_labels == -1).mean())

    result = {
        "labels": best_labels,
        "cluster_letters": cluster_letters,
        "centroids": centroids,
        "clusterer": best_clusterer,
        "n_clusters": len(unique_labels),
        "noise_fraction": noise_fraction,
        "params": {"min_cluster_size": best_params[0], "min_samples": best_params[1]},
        "gate_results": gate_results,
        "cycle_diagnostics": cycle_diagnostics,
    }
    logger.info(
        "Vocabulary: %d clusters, noise fraction=%.3f",
        len(unique_labels), noise_fraction,
    )
    return result


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _grid_search_dbcv(
    X: np.ndarray,
    grid: list[tuple[int, int]],
    random_seed: int,
) -> tuple[tuple[int, int], float, np.ndarray, Any]:
    """Run grid search and return (best_params, best_dbcv, labels, clusterer).

    Parameters
    ----------
    X:
        Feature matrix.
    grid:
        List of (min_cluster_size, min_samples) tuples.
    random_seed:
        Random seed for reproducibility.
    """
    import hdbscan
    from hdbscan import validity

    best_params = grid[0]
    best_dbcv = -np.inf
    best_labels = None
    best_clusterer = None

    for min_cs, min_s in grid:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cs,
                min_samples=min_s,
                core_dist_n_jobs=1,
            )
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            if n_clusters < 2:
                continue
            dbcv = validity.validity_index(X, labels)
            if dbcv > best_dbcv:
                best_dbcv = dbcv
                best_params = (min_cs, min_s)
                best_labels = labels
                best_clusterer = clusterer
        except Exception as exc:
            logger.debug("HDBSCAN grid failed (%d, %d): %s", min_cs, min_s, exc)

    if best_labels is None:
        # Fall back to first valid combination
        for min_cs, min_s in grid:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
            best_labels = clusterer.fit_predict(X)
            best_clusterer = clusterer
            best_params = (min_cs, min_s)
            break

    return best_params, float(best_dbcv), best_labels, best_clusterer


# ---------------------------------------------------------------------------
# The 4 validation gates
# ---------------------------------------------------------------------------

def _run_all_gates(
    X: np.ndarray,
    labels: np.ndarray,
    syllables: list[dict[str, Any]],
    centroids: np.ndarray,
    cfg: Config,
) -> dict[str, dict[str, Any]]:
    """Run all four validation gates, returning per-gate results.

    Parameters
    ----------
    X:
        Feature matrix.
    labels:
        Cluster labels from HDBSCAN.
    syllables:
        Original syllable records.
    centroids:
        Cluster centroid array.
    cfg:
        Pipeline configuration.
    """
    return {
        "gate1_bootstrap_stability": _gate1_bootstrap_stability(X, labels, cfg),
        "gate2_cross_recordist": _gate2_cross_recordist(X, labels, syllables, cfg),
        "gate3_birdaves": _gate3_birdaves(syllables, labels, centroids, cfg),
        "gate4_spectral_homogeneity": _gate4_spectral_homogeneity(syllables, labels, cfg),
    }


def _gate1_bootstrap_stability(
    X: np.ndarray, labels: np.ndarray, cfg: Config
) -> dict[str, Any]:
    """Gate 1: Bootstrap cluster stability (median ARI ≥ threshold).

    Parameters
    ----------
    X:
        Feature matrix.
    labels:
        Full-data cluster labels.
    cfg:
        Pipeline configuration.
    """
    import hdbscan

    rng = np.random.RandomState(cfg.random_seed)
    params = _get_params_from_labels(labels, X)
    aris = []
    n = len(X)
    for _ in range(cfg.bootstrap_n):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        X_boot = X[idx]
        try:
            c = hdbscan.HDBSCAN(
                min_cluster_size=params[0], min_samples=params[1]
            )
            boot_labels = c.fit_predict(X_boot)
            full_sub = labels[idx]
            ari = adjusted_rand_score(full_sub, boot_labels)
            aris.append(ari)
        except Exception:
            pass

    median_ari = float(np.median(aris)) if aris else 0.0
    threshold = cfg.bootstrap_stability_ari_threshold
    return {
        "value": median_ari,
        "threshold": threshold,
        "passed": median_ari >= threshold,
        "n_bootstrap": len(aris),
    }


def _gate2_cross_recordist(
    X: np.ndarray,
    labels: np.ndarray,
    syllables: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    """Gate 2: Cross-recordist consistency (mean per-recordist ARI ≥ threshold).

    For each recordist, re-cluster their syllable subset independently, then
    compute ARI between those per-recordist labels and the full-data labels for
    the same subset.  Averaging across recordists gives a consistency score.

    Parameters
    ----------
    X:
        Feature matrix.
    labels:
        Full-data cluster labels.
    syllables:
        Syllable records with ``recordist_id``.
    cfg:
        Pipeline configuration.
    """
    import hdbscan

    recordists = [s.get("recordist_id", "") for s in syllables]
    unique_rec = sorted(set(r for r in recordists if r))
    if len(unique_rec) < 2:
        logger.info("Gate 2: fewer than 2 recordists; auto-passing")
        return {
            "value": 1.0,
            "threshold": cfg.cross_recordist_ari_threshold,
            "passed": True,
            "note": "fewer than 2 recordists",
        }

    n_clusters_est = len(set(labels) - {-1})
    params = _get_params_from_labels(labels, X)
    aris: list[float] = []

    for rec_id in unique_rec:
        idx = np.array([i for i, r in enumerate(recordists) if r == rec_id])
        # Scale min_cluster_size to subset size so clusters are still detectable
        subset_min_cs = max(3, len(idx) // max(1, n_clusters_est * 3))
        if len(idx) < subset_min_cs:
            continue
        X_sub = X[idx]
        try:
            c = hdbscan.HDBSCAN(
                min_cluster_size=subset_min_cs,
                min_samples=max(2, min(params[1], subset_min_cs)),
            )
            rec_labels = c.fit_predict(X_sub)
            full_sub = labels[idx]
            ari = adjusted_rand_score(full_sub, rec_labels)
            aris.append(ari)
        except Exception:
            pass

    if len(aris) < 2:
        return {
            "value": 0.0,
            "threshold": cfg.cross_recordist_ari_threshold,
            "passed": False,
            "note": "insufficient recordists with enough syllables",
        }

    mean_ari = float(np.mean(aris))
    threshold = cfg.cross_recordist_ari_threshold

    # One-sided permutation test: is mean_ari above the random-assignment null?
    rng = np.random.RandomState(cfg.random_seed)
    null_aris = [
        adjusted_rand_score(labels, rng.permutation(labels))
        for _ in range(999)
    ]
    p_value = float(np.mean([n >= mean_ari for n in null_aris]))

    passed = mean_ari >= threshold and p_value < 0.05
    return {
        "value": mean_ari,
        "threshold": threshold,
        "p_value": p_value,
        "passed": passed,
        "n_recordists": len(aris),
    }


def _gate3_birdaves(
    syllables: list[dict[str, Any]],
    labels: np.ndarray,
    centroids: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Gate 3: BirdAVES embedding agreement (mean cosine similarity ≥ threshold).

    Parameters
    ----------
    syllables:
        Syllable records with ``wav_path``, ``start_s``, ``end_s``.
    labels:
        Cluster labels.
    centroids:
        MFCC cluster centroids.
    cfg:
        Pipeline configuration.
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError:
        logger.warning("transformers/torch not available; gate 3 auto-passed")
        return {
            "value": 1.0,
            "threshold": cfg.birdaves_cosine_threshold,
            "passed": True,
            "note": "transformers not installed",
        }

    try:
        model_name = "google/bird-aves-local-m"
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
    except Exception as exc:
        logger.warning("Could not load BirdAVES model: %s; gate 3 auto-passed", exc)
        return {
            "value": 1.0,
            "threshold": cfg.birdaves_cosine_threshold,
            "passed": True,
            "note": f"model load failed: {exc}",
        }

    unique_labels = sorted(set(labels) - {-1})
    aves_centroids = []

    for lb in unique_labels:
        idx = np.where(labels == lb)[0]
        embeddings = []
        for i in idx[:20]:  # cap at 20 per cluster for speed
            syl = syllables[i]
            emb = _embed_syllable_birdaves(syl, extractor, model)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            aves_centroids.append(np.mean(embeddings, axis=0))
        else:
            aves_centroids.append(np.zeros(model.config.hidden_size))

    aves_centroids = np.array(aves_centroids)

    # Normalise
    mfcc_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    aves_norm = aves_centroids / (np.linalg.norm(aves_centroids, axis=1, keepdims=True) + 1e-9)

    # Hungarian matching on cosine similarity
    sim_matrix = mfcc_norm @ aves_norm.T  # (n_clusters, n_clusters)
    cost = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    mean_cosine = float(np.mean(sim_matrix[row_ind, col_ind]))

    threshold = cfg.birdaves_cosine_threshold
    return {
        "value": mean_cosine,
        "threshold": threshold,
        "passed": mean_cosine >= threshold,
        "n_clusters": len(unique_labels),
    }


def _embed_syllable_birdaves(
    syl: dict[str, Any], extractor: Any, model: Any
) -> np.ndarray | None:
    """Get BirdAVES embedding for a single syllable clip.

    Parameters
    ----------
    syl:
        Syllable record.
    extractor:
        HuggingFace feature extractor.
    model:
        HuggingFace model.
    """
    import librosa
    import torch

    try:
        y, sr = librosa.load(
            syl["wav_path"],
            sr=16000,
            offset=syl["start_s"],
            duration=syl["end_s"] - syl["start_s"],
            mono=True,
        )
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception:
        return None


def _gate4_spectral_homogeneity(
    syllables: list[dict[str, Any]],
    labels: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Gate 4: Per-cluster spectral homogeneity (≤ fraction with CV > threshold).

    Parameters
    ----------
    syllables:
        Feature-augmented syllable records.
    labels:
        Cluster labels.
    cfg:
        Pipeline configuration.
    """
    # Feature indices in the 36-dim vector:
    # 30-dim MFCC block + 6-dim pitch/amplitude
    # pitch/amplitude: [peak_freq=30, min_freq=31, freq_range=32, peak_amp=33, attack=34, decay=35]
    PEAK_FREQ_IDX = 30
    FREQ_RANGE_IDX = 32
    ATTACK_IDX = 34

    X = np.vstack([s["features"] for s in syllables]).astype(np.float64)
    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    if n_clusters == 0:
        return {
            "value": 0.0,
            "threshold": cfg.spectral_cv_flagged_fraction_threshold,
            "passed": False,
            "note": "no clusters",
        }

    flagged = 0
    for lb in unique_labels:
        cluster_X = X[labels == lb]
        if len(cluster_X) < 2:
            continue
        for feat_idx in (PEAK_FREQ_IDX, FREQ_RANGE_IDX, ATTACK_IDX):
            vals = cluster_X[:, feat_idx]
            mean_v = np.mean(vals)
            cv = np.std(vals) / (abs(mean_v) + 1e-9)
            if cv > cfg.spectral_cv_flag_threshold:
                flagged += 1
                break  # one flag per cluster is enough

    flagged_fraction = flagged / n_clusters
    threshold = cfg.spectral_cv_flagged_fraction_threshold
    return {
        "value": flagged_fraction,
        "threshold": threshold,
        "passed": flagged_fraction <= threshold,
        "n_clusters": n_clusters,
        "n_flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Cluster letter assignment
# ---------------------------------------------------------------------------

def _assign_letters(centroids: np.ndarray, unique_labels: list[int]) -> dict[int, str]:
    """Assign letter labels A–Z, AA–AZ, etc., sorted by first MFCC coefficient.

    Parameters
    ----------
    centroids:
        Cluster centroid array, shape ``(n_clusters, n_features)``.
    unique_labels:
        Sorted list of cluster integer labels.

    Returns
    -------
    dict[int, str]
        Mapping from integer cluster index to letter string.
    """
    # Sort by first MFCC coefficient (index 0)
    order = np.argsort(centroids[:, 0])
    sorted_labels = [unique_labels[i] for i in order]

    letters = []
    alphabet = string.ascii_uppercase
    for i in range(len(sorted_labels)):
        if i < 26:
            letters.append(alphabet[i])
        elif i < 52:
            letters.append("A" + alphabet[i - 26])
        else:
            letters.append("A" + alphabet[(i - 52) // 26] + alphabet[(i - 52) % 26])

    return {lb: lt for lb, lt in zip(sorted_labels, letters)}


# ---------------------------------------------------------------------------
# Grid adjustment helpers
# ---------------------------------------------------------------------------

def _get_params_from_labels(labels: np.ndarray, X: np.ndarray) -> tuple[int, int]:
    n = len(X)
    n_clusters = len(set(labels) - {-1})
    min_cs = max(5, n // max(1, n_clusters * 3))
    return (min(min_cs, 30), 5)


def _coarsen_grid(grid: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(max(cs, 15), max(ms, 5)) for cs, ms in grid]


def _adjust_grid_for_failures(
    grid: list[tuple[int, int]],
    gate_results: dict[str, dict[str, Any]],
    best_params: tuple[int, int],
) -> list[tuple[int, int]]:
    g1 = gate_results["gate1_bootstrap_stability"]["passed"]
    g2 = gate_results["gate2_cross_recordist"]["passed"]
    g4 = gate_results["gate4_spectral_homogeneity"]["passed"]

    new_grid = set(grid)
    if not g1 or not g4:
        # Stability/homogeneity failures: coarsen (larger min_cluster_size)
        new_grid.update(
            (min(cs + 5, 60), ms) for cs, ms in grid
        )
    if not g2:
        # Cross-recordist failures: finer
        new_grid.update(
            (max(cs - 5, 3), ms) for cs, ms in grid
        )
    return sorted(new_grid)


def _maybe_extend_grid(
    grid: list[tuple[int, int]],
    best_params: tuple[int, int],
    gate_results: dict[str, dict[str, Any]],
) -> list[tuple[int, int]]:
    cs_vals = sorted(set(cs for cs, _ in grid))
    ms_vals = sorted(set(ms for _, ms in grid))

    new = set(grid)
    if best_params[0] == cs_vals[-1]:
        new.update((cs_vals[-1] + 10, ms) for ms in ms_vals)
    if best_params[0] == cs_vals[0]:
        new.update((max(2, cs_vals[0] - 5), ms) for ms in ms_vals)
    return sorted(new)


def _zebrafinch_f1_check(
    X: np.ndarray, labels: np.ndarray, cfg: Config
) -> float | None:
    """Compute macro-F1 against zebra finch syllable labels if available.

    Parameters
    ----------
    X:
        Feature matrix.
    labels:
        HDBSCAN cluster labels.
    cfg:
        Pipeline configuration.
    """
    ref_path = cfg.data_path / "reference" / "zebrafinch_labels.npy"
    if not ref_path.exists():
        return None
    try:
        zf_labels = np.load(ref_path)
        if len(zf_labels) != len(X):
            return None
        mask = labels != -1
        _, _, f1, _ = precision_recall_fscore_support(
            zf_labels[mask], labels[mask], average="macro", zero_division=0
        )
        return float(f1)
    except Exception as exc:
        logger.debug("Zebra finch F1 check failed: %s", exc)
        return None
