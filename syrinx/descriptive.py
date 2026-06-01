"""Descriptive measures: entropy (H1, H2, C), species/cell profiling, UMAP scatter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_descriptives(
    cfg: Config,
    syllables: list[dict[str, Any]],
    vocabulary: dict[str, Any],
    run_log: Any = None,
) -> dict[str, Any]:
    """Compute all descriptive statistics.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    syllables:
        Feature-augmented syllable records with cluster assignments.
    vocabulary:
        Vocabulary dict from Stage 4.
    run_log:
        Optional PipelineRunLog.

    Returns
    -------
    dict
        Keys:
        - ``species_stats``: per-species entropy and acoustic summary stats
        - ``region_diversity``: per-BBS-region diversity for H2
        - ``umap_figure_path``: path to Figure 2 HTML
    """
    labels_array = vocabulary["labels"]
    cluster_letters = vocabulary["cluster_letters"]

    # Per-recording string sequences
    sp_sequences = _build_sequences(syllables, labels_array, cluster_letters, key="species")
    rec_sequences = _build_sequences(syllables, labels_array, cluster_letters, key="xc_id")

    # Entropy measures per species: compute per recording, then take mean
    species_entropy: dict[str, dict[str, float]] = {}
    sp_of_rec = {xc_id: syl.get("species", "unknown")
                 for syl in syllables for xc_id in [syl.get("xc_id", "")]}
    for sp, seqs in sp_sequences.items():
        per_rec_h1, per_rec_h2, per_rec_c = [], [], []
        for xc_id, seq in zip(
            [s.get("xc_id", "") for s in syllables if s.get("species") == sp],
            seqs,
        ):
            if not seq:
                continue
            h1, h2, c = compute_entropy_measures(seq)
            per_rec_h1.append(h1)
            per_rec_h2.append(h2)
            per_rec_c.append(c)
        if per_rec_h1:
            species_entropy[sp] = {
                "H1": float(np.mean(per_rec_h1)),
                "H2": float(np.mean(per_rec_h2)),
                "C": float(np.mean(per_rec_c)),
            }
        else:
            species_entropy[sp] = {"H1": 0.0, "H2": 0.0, "C": 0.0}

    # Acoustic summary statistics per species
    species_acoustic = _acoustic_summary_per_species(syllables, labels_array)

    species_stats = {
        sp: {**species_entropy.get(sp, {}), **species_acoustic.get(sp, {})}
        for sp in sorted(set(list(species_entropy.keys()) + list(species_acoustic.keys())))
    }

    # Per-BBS-region diversity for H2
    region_diversity = _compute_region_diversity(
        syllables, labels_array, cluster_letters, cfg
    )

    # UMAP figure
    fig_path = _make_umap_figure(syllables, labels_array, cluster_letters, cfg)

    result = {
        "species_stats": species_stats,
        "region_diversity": region_diversity,
        "umap_figure_path": str(fig_path),
        "bbs_region_boundary_note": (
            "BBS region boundaries are approximate lat/lon rectangles, not official "
            "BTO reporting polygons. Recordings near region borders may be "
            "misclassified. This feeds the preregistered H2 analysis."
        ),
    }

    if run_log is not None:
        run_log.record_stage("descriptive", {
            "n_species": len(species_stats),
            "n_regions_with_data": len(region_diversity),
        })

    return result


# ---------------------------------------------------------------------------
# Entropy measures
# ---------------------------------------------------------------------------

def compute_entropy_measures(seq: str) -> tuple[float, float, float]:
    """Compute H1, H2, and syntactic complexity C for a syllable string.

    Parameters
    ----------
    seq:
        String of syllable letters (noise points already excluded).

    Returns
    -------
    tuple[float, float, float]
        ``(H1, H2, C)`` where C = H1 − H2.
    """
    if not seq:
        return 0.0, 0.0, 0.0

    # H1: Shannon entropy of unigram frequencies
    counts: dict[str, int] = {}
    for ch in seq:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(seq)
    h1 = -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)

    # H2: mean conditional entropy of bigrams
    if len(seq) < 2:
        return float(h1), 0.0, float(h1)

    bigram_counts: dict[str, dict[str, int]] = {}
    for i in range(len(seq) - 1):
        prev, nxt = seq[i], seq[i + 1]
        bigram_counts.setdefault(prev, {}).setdefault(nxt, 0)
        bigram_counts[prev][nxt] += 1

    # p(prev): marginal probabilities
    prev_totals = {prev: sum(bc.values()) for prev, bc in bigram_counts.items()}
    grand_total = sum(prev_totals.values())

    h2 = 0.0
    for prev, bc in bigram_counts.items():
        p_prev = prev_totals[prev] / grand_total
        tot_prev = prev_totals[prev]
        cond_entropy = -sum(
            (cnt / tot_prev) * np.log2(cnt / tot_prev)
            for cnt in bc.values() if cnt > 0
        )
        h2 += p_prev * cond_entropy

    c = float(h1) - float(h2)
    return float(h1), float(h2), float(c)


# ---------------------------------------------------------------------------
# Acoustic summary statistics
# ---------------------------------------------------------------------------

def _acoustic_summary_per_species(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-species mean±SD for each acoustic feature.

    Parameters
    ----------
    syllables:
        Feature-augmented syllable records.
    labels_array:
        Cluster labels per syllable.
    """
    FEATURE_NAMES = ["peak_freq", "min_freq", "freq_range", "peak_amplitude", "attack_ms", "decay_ms"]
    FEAT_OFFSET = 30  # Features 30–35 in the 36-dim vector

    species_feats: dict[str, list[np.ndarray]] = {}
    species_fm_depths: dict[str, list[float]] = {}
    for i, syl in enumerate(syllables):
        if labels_array[i] == -1:
            continue
        sp = syl.get("species", "unknown")
        feats = syl.get("features")
        if feats is None:
            continue
        species_feats.setdefault(sp, []).append(np.array(feats[FEAT_OFFSET:FEAT_OFFSET + 6]))
        fm_d = syl.get("fm_depth_mean")
        if fm_d is not None:
            species_fm_depths.setdefault(sp, []).append(float(fm_d))

    result: dict[str, dict[str, float]] = {}
    for sp, feat_list in species_feats.items():
        X = np.array(feat_list)
        stats: dict[str, float] = {}
        for j, name in enumerate(FEATURE_NAMES):
            stats[f"mean_{name}"] = float(X[:, j].mean())
            stats[f"sd_{name}"] = float(X[:, j].std())
        fm_d_list = species_fm_depths.get(sp, [])
        stats["mean_fm_depth"] = float(np.mean(fm_d_list)) if fm_d_list else stats.get("mean_freq_range", 0.0)
        result[sp] = stats

    return result


# ---------------------------------------------------------------------------
# Region diversity for H2
# ---------------------------------------------------------------------------

def _compute_region_diversity(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> dict[str, dict[str, Any]]:
    """Compute per-BBS-region vocal diversity measures.

    Parameters
    ----------
    syllables:
        Syllable records with ``lat``, ``lon``.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    region_names = cfg.bbs_regions
    region_seqs: dict[str, list[str]] = {r: [] for r in region_names}

    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        lat = syl.get("lat")
        lon = syl.get("lon")
        if lat is None or lon is None:
            continue
        region = _assign_bbs_region(float(lat), float(lon), cfg)
        if region:
            region_seqs[region].append(cluster_letters[lb])

    diversity: dict[str, dict[str, Any]] = {}
    for region, seq_list in region_seqs.items():
        if not seq_list:
            continue
        combined = "".join(seq_list)
        h1, h2, c = compute_entropy_measures(combined)
        vocab_size = len(set(combined))
        diversity[region] = {
            "vocab_size": vocab_size,
            "mean_complexity": c,
            "H1": h1,
            "H2": h2,
            "n_syllables": len(combined),
        }

    # Pairwise within-region acoustic distances (for third diversity component)
    _add_mean_pairwise_distance(diversity, syllables, labels_array, cluster_letters, cfg)

    # Compute composite diversity via PCA
    _add_composite_diversity(diversity)

    return diversity


def _add_mean_pairwise_distance(
    diversity: dict[str, dict[str, Any]],
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> None:
    """Add mean pairwise within-region acoustic distance.

    Parameters
    ----------
    diversity:
        Partially-filled diversity dict (modified in place).
    syllables:
        Syllable records.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    region_feat: dict[str, list[np.ndarray]] = {r: [] for r in cfg.bbs_regions}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        lat = syl.get("lat")
        lon = syl.get("lon")
        if lat is None or lon is None:
            continue
        region = _assign_bbs_region(float(lat), float(lon), cfg)
        feats = syl.get("features")
        if region and feats is not None:
            region_feat[region].append(np.array(feats))

    for region, feats in region_feat.items():
        if region not in diversity or len(feats) < 2:
            diversity.setdefault(region, {})["mean_pairwise_distance"] = None
            continue
        X = np.array(feats[:cfg.max_syllables_pairwise_distance])
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric="euclidean")
        diversity[region]["mean_pairwise_distance"] = float(dists.mean()) if len(dists) > 0 else None


def _add_composite_diversity(diversity: dict[str, dict[str, Any]]) -> None:
    """Compute PC1 of (vocab_size, mean_complexity, mean_pairwise_distance) as composite.

    Parameters
    ----------
    diversity:
        Diversity dict (modified in place).
    """
    regions = [r for r in diversity if
               diversity[r].get("vocab_size") is not None and
               diversity[r].get("mean_complexity") is not None and
               diversity[r].get("mean_pairwise_distance") is not None]

    if len(regions) < 2:
        for r in diversity:
            diversity[r]["composite_diversity"] = diversity[r].get("mean_complexity")
        return

    X = np.array([
        [diversity[r]["vocab_size"], diversity[r]["mean_complexity"], diversity[r]["mean_pairwise_distance"]]
        for r in regions
    ])
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).ravel()
    for i, r in enumerate(regions):
        diversity[r]["composite_diversity"] = float(pc1[i])
    for r in diversity:
        if r not in regions:
            diversity[r]["composite_diversity"] = None


def _assign_bbs_region(lat: float, lon: float, cfg: Config) -> str | None:
    """Assign a lat/lon coordinate to a BBS reporting region.

    .. warning:: **APPROXIMATION** — these boundaries are simplified lat/lon
        rectangles, NOT the official BTO BBS reporting region polygons.  No
        machine-readable version of the BTO boundaries was available at the
        time of preregistration.  This function feeds the preregistered H2
        analysis (vocal diversity ~ BBS population trend).  Misclassification
        of recordings near region borders could affect the H2 result; this
        limitation is reported in the output JSON and should be noted in the
        supplement.  Replace with official BTO shapefiles if they become
        available.

    Parameters
    ----------
    lat, lon:
        Decimal degree coordinates.
    cfg:
        Pipeline configuration.
    """
    # Northern Ireland must be checked before Scotland: NI extends to ≈55.4°N,
    # which overlaps the Scotland threshold (55.0°N) along the north Antrim coast.
    if 54.0 <= lat <= 55.4 and -8.2 <= lon <= -5.3:
        return "Northern Ireland"
    # Wales (approximate bounding box)
    if 51.3 <= lat <= 53.5 and -5.4 <= lon <= -2.6:
        return "Wales"
    # Scotland
    if lat > cfg.bbs_scotland_lat_threshold:
        return "Scotland"
    # England quadrants (simplified — see approximation warning above)
    if lat < cfg.bbs_scotland_lat_threshold and lon > -2.0 and lat > 52.5:
        return "England-Midlands"
    if lat < 52.5 and lon > -2.0:
        return "England-SE"
    if lat < 52.5 and lon <= -2.0:
        return "England-SW"
    if lat >= 52.5 and lon <= -2.0:
        return "England-N"
    return None


# ---------------------------------------------------------------------------
# UMAP figure
# ---------------------------------------------------------------------------

def _make_umap_figure(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    cfg: Config,
) -> str:
    """Generate UMAP 3D scatter of syllable feature space (Figure 2).

    Parameters
    ----------
    syllables:
        Feature-augmented syllable records.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    cfg:
        Pipeline configuration.
    """
    import plotly.express as px
    import umap

    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "figure_2.html"

    X = np.vstack([s["features"] for s in syllables if "features" in s])
    species = [s.get("species", "unknown") for s in syllables if "features" in s]
    idx_map = {i: j for j, (i, s) in enumerate((i, s) for i, s in enumerate(syllables) if "features" in s)}

    try:
        reducer = umap.UMAP(n_components=3, random_state=cfg.random_seed, n_jobs=1)
        embedding = reducer.fit_transform(X)

        import pandas as pd
        df = pd.DataFrame({
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "z": embedding[:, 2],
            "species": species,
        })

        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="species",
            title="Figure 2: UMAP 3D syllable feature space",
            opacity=0.6,
        )
        fig.update_traces(marker_size=2)
        fig.write_html(str(output_path))
        logger.info("UMAP figure saved to %s", output_path)
    except Exception as exc:
        logger.warning("UMAP figure generation failed: %s", exc)

    return str(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sequences(
    syllables: list[dict[str, Any]],
    labels_array: np.ndarray,
    cluster_letters: dict[int, str],
    key: str,
) -> dict[str, list[str]]:
    """Group syllable letter sequences by a metadata key.

    Parameters
    ----------
    syllables:
        Syllable records.
    labels_array:
        Cluster labels.
    cluster_letters:
        Label-to-letter mapping.
    key:
        Metadata key to group by (e.g. ``'species'``, ``'xc_id'``).
    """
    result: dict[str, list[str]] = {}
    for i, syl in enumerate(syllables):
        lb = labels_array[i]
        if lb == -1:
            continue
        group = syl.get(key, "unknown")
        result.setdefault(group, []).append(cluster_letters[lb])
    return result
