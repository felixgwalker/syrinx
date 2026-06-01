"""Stage 7 — UPGMA/NJ tree reconstruction + RF/quartet/MS scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .utils import save_manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reconstruct_trees(
    cfg: Config,
    alignment_result: dict[str, Any],
    run_log: Any = None,
) -> dict[str, Any]:
    """Reconstruct UPGMA and NJ acoustic trees and compare to molecular reference.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    alignment_result:
        Output from :func:`~syrinx.align.align_all`.
    run_log:
        Optional PipelineRunLog.

    Returns
    -------
    dict
        Keys:
        - ``upgma_tree``: Bio.Phylo tree object
        - ``nj_tree``: Bio.Phylo tree object
        - ``upgma_newick``: Newick string
        - ``nj_newick``: Newick string
        - ``rf_distance``: normalised Robinson–Foulds distance between UPGMA and NJ
        - ``quartet_distance``: normalised quartet distance
        - ``ms_distance``: matching split distance
        - ``method_stability``: dict of percentile ranks vs null
        - ``reference_trees``: loaded molecular reference trees
    """
    D = alignment_result["distance_matrix"]
    species_names = alignment_result["species_names"]

    upgma_tree, nj_tree = _build_trees(D, species_names)

    upgma_newick = _tree_to_newick(upgma_tree)
    nj_newick = _tree_to_newick(nj_tree)

    tree_dir = cfg.data_path / "trees"
    tree_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "acoustic_upgma.nwk").write_text(upgma_newick)
    (tree_dir / "acoustic_nj.nwk").write_text(nj_newick)
    logger.info("Saved acoustic trees to %s", tree_dir)

    # Topological agreement (see metric_approximation_note below)
    rf, unweighted_rf_norm, weighted_rf_norm = _compute_tree_agreement(upgma_tree, nj_tree)
    method_stability = _null_model_tree_agreement(
        D, species_names, rf, unweighted_rf_norm, weighted_rf_norm, cfg
    )

    # Nominate-only trees (Item 4)
    D_nom = alignment_result.get("nominate_only_distance_matrix")
    nominate_entities = alignment_result.get("nominate_entities", [])
    nominate_result: dict[str, Any] = {}
    if D_nom is not None and len(nominate_entities) >= 3:
        upgma_nom, nj_nom = _build_trees(D_nom, nominate_entities)
        (tree_dir / "acoustic_upgma_nominate.nwk").write_text(_tree_to_newick(upgma_nom))
        (tree_dir / "acoustic_nj_nominate.nwk").write_text(_tree_to_newick(nj_nom))
        rf_nom, uwrf_nom, wrf_nom = _compute_tree_agreement(upgma_nom, nj_nom)
        nominate_result = {
            "upgma_newick": _tree_to_newick(upgma_nom),
            "nj_newick": _tree_to_newick(nj_nom),
            "rf_distance": rf_nom,
            "unweighted_rf_normalized": uwrf_nom,
            "weighted_rf_normalized": wrf_nom,
            "n_species": len(nominate_entities),
        }
        logger.info("Nominate-only trees saved (n=%d species)", len(nominate_entities))

    # Load reference molecular trees
    reference_trees = _load_reference_trees(cfg)

    # Figures
    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _make_tanglegram(nj_tree, reference_trees.get("alstrom2018"), fig_dir / "figure_5.html", species_names)

    result = {
        "upgma_tree": upgma_tree,
        "nj_tree": nj_tree,
        "upgma_newick": upgma_newick,
        "nj_newick": nj_newick,
        "rf_distance": rf,
        # Renamed from quartet_distance / ms_distance (Item 7).  These are
        # dendropy RF variants, not true quartet or matching-split distances.
        # See metric_approximation_note.
        "unweighted_rf_normalized": unweighted_rf_norm,
        "weighted_rf_normalized": weighted_rf_norm,
        "metric_approximation_note": (
            "unweighted_rf_normalized uses dendropy symmetric_difference / (2*(n-3)); "
            "weighted_rf_normalized uses dendropy weighted_robinson_foulds_distance / n_taxa. "
            "Neither is a true quartet distance nor a true matching-split distance. "
            "tqdist is not available in this environment. "
            "This limitation should be noted in the supplement."
        ),
        "method_stability": method_stability,
        "reference_trees": reference_trees,
        "nominate_only_trees": nominate_result,
    }

    if run_log is not None:
        run_log.record_stage("stage7_phylo", {
            "rf_distance": rf,
            "n_species": len(species_names),
            "method_stable": method_stability.get("rf_percentile_rank", 1.0) < 0.05,
        })

    return result


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_trees(D: np.ndarray, names: list[str]) -> tuple[Any, Any]:
    """Build UPGMA and NJ trees from a distance matrix.

    Parameters
    ----------
    D:
        Acoustic distance matrix.
    names:
        Species/entity names corresponding to matrix rows/columns.
    """
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    n = len(names)
    dm_data = []
    for i in range(n):
        row = [float(D[i, j]) for j in range(i + 1)]
        dm_data.append(row)

    dm = DistanceMatrix(names=names, matrix=dm_data)
    constructor = DistanceTreeConstructor()
    upgma = constructor.upgma(dm)
    nj = constructor.nj(dm)
    return upgma, nj


def _tree_to_newick(tree: Any) -> str:
    """Convert a Bio.Phylo tree to a Newick string.

    Parameters
    ----------
    tree:
        Bio.Phylo tree object.
    """
    import io
    from Bio import Phylo

    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    return buf.getvalue().strip()


# ---------------------------------------------------------------------------
# Topological agreement
# ---------------------------------------------------------------------------

def _compute_tree_agreement(
    tree1: Any, tree2: Any
) -> tuple[float, float, float]:
    """Compute RF and two RF-variant distances between two trees.

    Returns
    -------
    tuple[float, float, float]
        ``(rf_norm, unweighted_rf_normalized, weighted_rf_normalized)``

    Notes
    -----
    These are *not* true quartet distance or matching-split distance despite
    earlier labelling.  ``unweighted_rf_normalized`` wraps
    ``dendropy.symmetric_difference / (2*(n-3))`` and
    ``weighted_rf_normalized`` wraps
    ``dendropy.weighted_robinson_foulds_distance / n_taxa``.
    tqdist is not available in this environment; see metric_approximation_note
    in the ``reconstruct_trees`` output for the supplement caveat.
    """
    try:
        import dendropy
        from dendropy.calculate import treecompare

        t1_str = _tree_to_newick(tree1)
        t2_str = _tree_to_newick(tree2)

        tns = dendropy.TaxonNamespace()
        dt1 = dendropy.Tree.get(data=t1_str, schema="newick", taxon_namespace=tns)
        dt2 = dendropy.Tree.get(data=t2_str, schema="newick", taxon_namespace=tns)
        dt1.encode_bipartitions()
        dt2.encode_bipartitions()

        n_taxa = len(tns)
        rf_raw = treecompare.symmetric_difference(dt1, dt2)
        rf_norm = rf_raw / (2 * (n_taxa - 3)) if n_taxa > 3 else 0.0

        try:
            uwrf = treecompare.unweighted_robinson_foulds_distance(dt1, dt2)
            unweighted_rf_norm = float(uwrf) / max(1, n_taxa)
        except Exception:
            unweighted_rf_norm = rf_norm

        try:
            wrf = treecompare.weighted_robinson_foulds_distance(dt1, dt2)
            weighted_rf_norm = float(wrf) / max(1, n_taxa)
        except Exception:
            weighted_rf_norm = rf_norm

        return float(rf_norm), float(unweighted_rf_norm), float(weighted_rf_norm)
    except ImportError:
        raise ImportError(
            "dendropy is required for tree distance calculations. "
            "Install it with: pip install dendropy"
        ) from None
    except Exception as exc:
        logger.warning("Tree comparison failed: %s", exc)
        return float("nan"), float("nan"), float("nan")


def _null_model_tree_agreement(
    D: np.ndarray,
    names: list[str],
    obs_rf: float,
    obs_unweighted_rf_norm: float,
    obs_weighted_rf_norm: float,
    cfg: Config,
) -> dict[str, Any]:
    """Null model for topological agreement by shuffling the distance matrix.

    Parameters
    ----------
    D:
        Observed distance matrix.
    names:
        Entity names.
    obs_rf:
        Observed normalised RF distance.
    obs_unweighted_rf_norm:
        Observed unweighted-RF-normalised value.
    obs_weighted_rf_norm:
        Observed weighted-RF-normalised value.
    cfg:
        Pipeline configuration.
    """
    rng = np.random.RandomState(cfg.random_seed)
    n = len(names)
    null_rfs: list[float] = []
    null_uwrf: list[float] = []
    null_wrf: list[float] = []

    for _ in range(cfg.rf_null_permutations):
        perm = rng.permutation(n)
        D_perm = D[np.ix_(perm, perm)]
        try:
            t1, t2 = _build_trees(D_perm, names)
            rf, uwrf, wrf = _compute_tree_agreement(t1, t2)
            null_rfs.append(rf)
            null_uwrf.append(uwrf)
            null_wrf.append(wrf)
        except Exception:
            pass

    def percentile_rank(obs: float, null: list[float]) -> float:
        if not null:
            return float("nan")
        return float(np.mean([v <= obs for v in null]))

    return {
        "rf_percentile_rank": percentile_rank(obs_rf, null_rfs),
        "unweighted_rf_percentile_rank": percentile_rank(obs_unweighted_rf_norm, null_uwrf),
        "weighted_rf_percentile_rank": percentile_rank(obs_weighted_rf_norm, null_wrf),
        "n_null": len(null_rfs),
        "interpretation": {
            "lower_5pct": "stable signal (method-insensitive)",
            "upper_95pct": "method-sensitive topology",
        },
    }


# ---------------------------------------------------------------------------
# Reference trees
# ---------------------------------------------------------------------------

def _load_reference_trees(cfg: Config) -> dict[str, Any]:
    """Load molecular reference trees from disk.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    """
    from Bio import Phylo

    trees: dict[str, Any] = {}
    tree_dir = cfg.data_path / "trees"

    paths = {
        "alstrom2018": tree_dir / "alstrom2018.nwk",
        "tietze2015": tree_dir / "tietze2015.nwk",
    }
    for name, path in paths.items():
        if path.exists():
            try:
                tree = next(Phylo.parse(str(path), "newick"))
                trees[name] = tree
                logger.info("Loaded reference tree: %s", name)
            except Exception as exc:
                logger.warning("Could not load reference tree %s: %s", name, exc)
        else:
            logger.info("Reference tree not found: %s (place at %s)", name, path)

    return trees


# ---------------------------------------------------------------------------
# Tanglegram
# ---------------------------------------------------------------------------

def _make_tanglegram(
    acoustic_tree: Any,
    molecular_tree: Any,
    output_path: Path,
    species_names: list[str],
) -> None:
    """Generate a tanglegram figure (Figure 5).

    Parameters
    ----------
    acoustic_tree:
        NJ acoustic tree.
    molecular_tree:
        Molecular reference tree (or None).
    output_path:
        Output HTML path.
    species_names:
        Species names in distance matrix order.
    """
    import plotly.graph_objects as go

    if molecular_tree is None:
        logger.info("No molecular reference tree; generating acoustic tree visualisation only")
        fig = go.Figure()
        fig.add_annotation(
            text="Molecular reference tree not loaded. Place alstrom2018.nwk in data/trees/.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.write_html(str(output_path))
        return

    # Simple tanglegram: two cladograms side by side with connecting lines
    acou_tips = _get_tip_order(acoustic_tree)
    mol_tips = _get_tip_order(molecular_tree)
    shared = [t for t in acou_tips if t in mol_tips]

    acou_y = {t: i for i, t in enumerate(acou_tips)}
    mol_y = {t: i for i, t in enumerate(mol_tips)}

    traces = []
    for tip in shared:
        concordant = abs(acou_y[tip] - mol_y[tip]) <= 2
        colour = "#2ecc71" if concordant else "#e74c3c"
        traces.append(go.Scatter(
            x=[0, 1],
            y=[acou_y[tip], mol_y[tip]],
            mode="lines",
            line={"color": colour, "width": 0.8},
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Tanglegram: Acoustic NJ tree vs Alström et al. (2018) molecular tree",
        xaxis={"tickvals": [0, 1], "ticktext": ["Acoustic", "Molecular"]},
        height=max(400, len(shared) * 14),
    )
    fig.write_html(str(output_path))
    logger.info("Tanglegram saved to %s", output_path)


def _get_tip_order(tree: Any) -> list[str]:
    """Return leaf names in tree traversal order.

    Parameters
    ----------
    tree:
        Bio.Phylo tree object.
    """
    tips = []
    for clade in tree.find_clades(order="level"):
        if clade.is_terminal():
            tips.append(clade.name or "")
    return tips
