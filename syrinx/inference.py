"""Stage 9 — MRM (via R/ecodist), PGLS (via R/caper), and Mantel tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .config import Config
from .utils import geographic_distance_matrix, upper_triangle

logger = logging.getLogger(__name__)


class PipelineGatingError(Exception):
    """Raised when vocabulary or null model gates have not passed."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_inference(
    cfg: Config,
    alignment_result: dict[str, Any],
    vocabulary: dict[str, Any],
    species_metadata: list[dict[str, Any]],
    descriptive_result: dict[str, Any],
    molecular_tree: Any = None,
    run_log: Any = None,
    dataset: str = "genus",
) -> dict[str, Any]:
    """Run all inferential analyses for H1, H2, and H3.

    Gated: raises PipelineGatingError if vocabulary gates or null model
    gate have not passed.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    alignment_result:
        Output from :func:`~syrinx.align.align_all`.
    vocabulary:
        Vocabulary result from Stage 4.
    species_metadata:
        Per-species metadata (lat, lon, etc.).
    descriptive_result:
        Output from :func:`~syrinx.descriptive.compute_descriptives`.
    molecular_tree:
        Loaded reference molecular tree (Bio.Phylo) or None.
    run_log:
        Optional PipelineRunLog.
    dataset:
        ``'genus'`` or ``'within_species'``.

    Returns
    -------
    dict
        Keys: ``h1``, ``h2``, ``h3``.
    """
    _check_gates(vocabulary, alignment_result)

    result: dict[str, Any] = {}

    if dataset in ("genus", "both"):
        result["h1"] = _run_h1(cfg, alignment_result, species_metadata, run_log, molecular_tree)
    if dataset in ("within_species", "both"):
        result["h2"] = _run_h2(cfg, descriptive_result, run_log)
        result["h3"] = _run_h3(cfg, alignment_result, run_log)

    # Persist outputs
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for hyp, res in result.items():
        with (out_dir / f"results_{hyp}.json").open("w") as fh:
            json.dump(_serialise(res), fh, indent=2)
    logger.info("Inference results written to %s", out_dir)

    if run_log is not None:
        run_log.record_stage("stage9_inference", {
            "hypotheses_tested": list(result.keys()),
        })

    return result


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def _check_gates(
    vocabulary: dict[str, Any],
    alignment_result: dict[str, Any],
) -> None:
    """Raise PipelineGatingError if either gate failed.

    Parameters
    ----------
    vocabulary:
        Vocabulary dict with ``gate_results``.
    alignment_result:
        Alignment dict with ``null_model_result``.
    """
    gate_results = vocabulary.get("gate_results", {})
    failed_gates = [g for g, r in gate_results.items() if not r.get("passed")]
    if failed_gates:
        diag = {"failed_vocabulary_gates": failed_gates, "gate_results": gate_results}
        raise PipelineGatingError(
            f"Vocabulary gates failed: {failed_gates}", diag
        )

    null_result = alignment_result.get("null_model_result", {})
    if not null_result.get("passed", False):
        raise PipelineGatingError(
            "Null model gating test did not pass; inferential analyses not conducted.",
            {"null_model_result": null_result},
        )


# ---------------------------------------------------------------------------
# H1 — MRM
# ---------------------------------------------------------------------------

def _run_h1(
    cfg: Config,
    alignment_result: dict[str, Any],
    species_metadata: list[dict[str, Any]],
    run_log: Any,
    molecular_tree: Any = None,
) -> dict[str, Any]:
    """Run H1 MRM analysis and complementary PGLS.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    alignment_result:
        Alignment result dict.
    species_metadata:
        Per-species metadata with lat/lon.
    run_log:
        Optional PipelineRunLog.
    molecular_tree:
        Bio.Phylo molecular reference tree, or None.
    """
    D_acoustic = alignment_result["distance_matrix"]
    species_names = alignment_result["species_names"]
    n = len(species_names)

    # Load molecular patristic distance matrix
    D_molecular = _load_molecular_distances(cfg, species_names)
    if D_molecular is None:
        logger.warning("Molecular distances not available; H1 cannot be run")
        return {"error": "molecular distances unavailable"}

    # Geographic distance matrix
    meta_by_sp = {m["species"]: m for m in species_metadata}
    lats = [meta_by_sp.get(sp, {}).get("lat", 0.0) or 0.0 for sp in species_names]
    lons = [meta_by_sp.get(sp, {}).get("lon", 0.0) or 0.0 for sp in species_names]
    D_geo = geographic_distance_matrix(lats, lons)

    acoustic_vec = upper_triangle(D_acoustic)
    molecular_vec = upper_triangle(D_molecular)
    geo_vec = upper_triangle(D_geo)

    # Primary: MRM via R ecodist
    mrm_result = _run_mrm_r(acoustic_vec, molecular_vec, geo_vec, cfg)

    # Secondary Mantel tests
    mantel_pearson = _run_mantel_r(acoustic_vec, molecular_vec, cfg, method="pearson")
    mantel_spearman = _run_mantel_r(acoustic_vec, molecular_vec, cfg, method="spearman")
    mantel_geo = _run_mantel_r(acoustic_vec, geo_vec, cfg, method="pearson")

    # Partial Mantel if both acoustic~molecular and acoustic~geo are significant
    if mantel_pearson.get("p_value", 1.0) < 0.05 and mantel_geo.get("p_value", 1.0) < 0.05:
        partial_mantel = _run_partial_mantel_r(acoustic_vec, molecular_vec, geo_vec, cfg)
    else:
        partial_mantel = None

    # Bootstrap CI from recording resampling
    bootstrap_cis = alignment_result.get("bootstrap_cis", {})

    # Biological meaningfulness
    coef = mrm_result.get("molecular_coefficient", float("nan"))
    spr2 = mrm_result.get("semipartial_r2", float("nan"))
    statistically_sig = mrm_result.get("p_value", 1.0) < cfg.bonferroni_alpha
    biologically_meaningful = (
        not np.isnan(spr2) and spr2 >= cfg.h1_biological_threshold_semipartial_r2
    )

    interpretation = _interpret_result(
        statistically_sig, biologically_meaningful, cfg.h1_biological_threshold_semipartial_r2, spr2, "semi-partial r²"
    )

    # Complementary PGLS (secondary, α = 0.05 uncorrected)
    pgls_result = _run_pgls_r(species_names, D_acoustic, D_geo, molecular_tree, cfg)

    result = {
        "mrm": mrm_result,
        "pgls": pgls_result,
        "mantel_pearson": mantel_pearson,
        "mantel_spearman": mantel_spearman,
        "mantel_geographic": mantel_geo,
        "partial_mantel": partial_mantel,
        "n_species": n,
        "alpha": cfg.bonferroni_alpha,
        "biological_threshold": cfg.h1_biological_threshold_semipartial_r2,
        "statistically_significant": statistically_sig,
        "biologically_meaningful": biologically_meaningful,
        "interpretation": interpretation,
    }

    if run_log is not None:
        run_log.record_threshold(
            "H1_MRM_p_value",
            mrm_result.get("p_value", float("nan")),
            cfg.bonferroni_alpha,
            statistically_sig,
            stage="stage9_H1",
        )
        run_log.record_threshold(
            "H1_semipartial_r2",
            spr2,
            cfg.h1_biological_threshold_semipartial_r2,
            biologically_meaningful,
            stage="stage9_H1",
        )

    _make_figure_6(D_acoustic, D_molecular, species_names, mrm_result, cfg)
    return result


def _run_mrm_r(
    acoustic_vec: np.ndarray,
    molecular_vec: np.ndarray,
    geo_vec: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Run ecodist::MRM via rpy2.

    Parameters
    ----------
    acoustic_vec:
        Upper-triangle acoustic distances.
    molecular_vec:
        Upper-triangle molecular patristic distances.
    geo_vec:
        Upper-triangle great-circle geographic distances.
    cfg:
        Pipeline configuration.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        ecodist = importr("ecodist")

        r_acoustic = ro.FloatVector(acoustic_vec.tolist())
        r_molecular = ro.FloatVector(molecular_vec.tolist())
        r_geo = ro.FloatVector(geo_vec.tolist())

        formula = ro.Formula("acoustic ~ molecular + geo")
        env = formula.environment
        env["acoustic"] = r_acoustic
        env["molecular"] = r_molecular
        env["geo"] = r_geo

        mrm_out = ecodist.MRM(
            formula,
            nperm=cfg.mrm_permutations,
            method="pearson",
        )
        coef_matrix = np.array(mrm_out.rx2("coef"))
        intercept = float(coef_matrix[0, 0])
        mol_coef = float(coef_matrix[1, 0])
        geo_coef = float(coef_matrix[2, 0])
        mol_p = float(coef_matrix[1, 1])
        r2 = float(np.array(mrm_out.rx2("r.squared"))[0])
        f_p = float(np.array(mrm_out.rx2("F.pval"))[0])

        # Semi-partial r² approximation: partial out geographic effect
        spr2 = _estimate_semipartial_r2(acoustic_vec, molecular_vec, geo_vec)

        return {
            "intercept": intercept,
            "molecular_coefficient": mol_coef,
            "geographic_coefficient": geo_coef,
            "p_value": mol_p,
            "r_squared": r2,
            "f_pvalue": f_p,
            "semipartial_r2": spr2,
            "n_permutations": cfg.mrm_permutations,
            "method": "pearson",
        }
    except Exception as exc:
        logger.error("MRM via rpy2 failed: %s", exc)
        return {"error": str(exc), "method": "pearson"}


def _estimate_semipartial_r2(
    acoustic: np.ndarray, molecular: np.ndarray, geo: np.ndarray
) -> float:
    """Estimate semi-partial r² for the molecular predictor.

    Computed as the difference in R² between the full model and a model
    with only geographic distance.

    Parameters
    ----------
    acoustic, molecular, geo:
        Upper-triangle vectors.
    """
    from sklearn.linear_model import LinearRegression

    X_full = np.column_stack([molecular, geo])
    X_geo_only = geo.reshape(-1, 1)
    y = acoustic

    def r2(X: np.ndarray) -> float:
        lr = LinearRegression().fit(X, y)
        ss_res = np.sum((y - lr.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return max(0.0, r2(X_full) - r2(X_geo_only))


def _run_mantel_r(
    x: np.ndarray, y: np.ndarray, cfg: Config, method: str = "pearson"
) -> dict[str, Any]:
    """Run ecodist::mantel via rpy2.

    Parameters
    ----------
    x, y:
        Upper-triangle distance vectors.
    cfg:
        Pipeline configuration.
    method:
        ``'pearson'`` or ``'spearman'``.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        ecodist = importr("ecodist")

        r_x = ro.FloatVector(x.tolist())
        r_y = ro.FloatVector(y.tolist())
        out = ecodist.mantel(r_x, r_y, nperm=cfg.mantel_permutations, method=method)
        arr = np.array(out)
        return {
            "r": float(arr[0]),
            "p_value": float(arr[4]),
            "method": method,
            "n_permutations": cfg.mantel_permutations,
        }
    except Exception as exc:
        logger.warning("Mantel via rpy2 failed: %s; using Python fallback", exc)
        return _mantel_python(x, y, cfg.mantel_permutations, method)


def _run_partial_mantel_r(
    acoustic: np.ndarray, molecular: np.ndarray, geo: np.ndarray, cfg: Config
) -> dict[str, Any]:
    """Run partial Mantel (acoustic ~ molecular | geo) via rpy2.

    Parameters
    ----------
    acoustic, molecular, geo:
        Upper-triangle distance vectors.
    cfg:
        Pipeline configuration.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        ecodist = importr("ecodist")
        out = ecodist.partial_mantel(
            ro.FloatVector(acoustic.tolist()),
            ro.FloatVector(molecular.tolist()),
            ro.FloatVector(geo.tolist()),
            nperm=cfg.mantel_permutations,
        )
        arr = np.array(out)
        return {"r": float(arr[0]), "p_value": float(arr[2]), "type": "partial"}
    except Exception as exc:
        logger.warning("Partial Mantel via rpy2 failed: %s", exc)
        return {"error": str(exc)}


def _run_pgls_r(
    species_names: list[str],
    D_acoustic: np.ndarray,
    D_geo: np.ndarray,
    molecular_tree: Any,
    cfg: Config,
) -> dict[str, Any]:
    """Secondary PGLS via caper: mean acoustic distance ~ mean geographic distance.

    Uses Pagel's λ estimated by ML. Reports slope, SE, t, p, and λ.
    α = 0.05 uncorrected (preregistered secondary analysis).

    Parameters
    ----------
    species_names:
        Ordered species labels matching rows/columns of the distance matrices.
    D_acoustic:
        n×n acoustic distance matrix.
    D_geo:
        n×n great-circle geographic distance matrix.
    molecular_tree:
        Bio.Phylo molecular reference tree, or None.
    cfg:
        Pipeline configuration.
    """
    if molecular_tree is None:
        return {"error": "no molecular tree provided for PGLS"}

    n = len(species_names)
    if n < 4:
        return {"error": f"too few species for PGLS (n={n})"}

    # Per-species means across all other species (off-diagonal)
    D_ac = D_acoustic.copy().astype(float)
    D_ge = D_geo.copy().astype(float)
    np.fill_diagonal(D_ac, np.nan)
    np.fill_diagonal(D_ge, np.nan)
    acoustic_means = np.nanmean(D_ac, axis=1)
    geo_means = np.nanmean(D_ge, axis=1)

    try:
        import io
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from Bio import Phylo

        numpy2ri.activate()
        ape = importr("ape")
        caper = importr("caper")
        base = importr("base")

        # Serialise Bio.Phylo tree → Newick string
        buf = io.StringIO()
        Phylo.write(molecular_tree, buf, "newick")
        newick_str = buf.getvalue().strip()

        # Push data into R global environment with a unique prefix
        ro.globalenv["_syrinx_species"] = ro.StrVector(species_names)
        ro.globalenv["_syrinx_acoustic"] = ro.FloatVector(acoustic_means.tolist())
        ro.globalenv["_syrinx_geo"] = ro.FloatVector(geo_means.tolist())
        ro.globalenv["_syrinx_newick"] = ro.StrVector([newick_str])

        ro.r("""
            .syrinx_tree <- ape::read.tree(text = `_syrinx_newick`)
            .syrinx_df <- data.frame(
                species  = `_syrinx_species`,
                acoustic = `_syrinx_acoustic`,
                geo      = `_syrinx_geo`,
                stringsAsFactors = FALSE
            )
            .syrinx_cdat <- caper::comparative.data(
                phy       = .syrinx_tree,
                data      = .syrinx_df,
                names.col = "species",
                warn.dropped = FALSE
            )
            .syrinx_pgls  <- caper::pgls(acoustic ~ geo, data = .syrinx_cdat, lambda = "ML")
            .syrinx_summ  <- summary(.syrinx_pgls)
            .syrinx_coef  <- coef(.syrinx_summ)
            .syrinx_lam   <- .syrinx_pgls$param[["lambda"]]
        """)

        coef_matrix = np.array(ro.r[".syrinx_coef"])
        lambda_val = float(ro.r[".syrinx_lam"][0])

        # Rows: (Intercept), geo  ·  Columns: Estimate, Std.Error, t value, Pr(>|t|)
        if coef_matrix.shape[0] < 2:
            raise ValueError("PGLS coefficient matrix has unexpected shape")

        slope = float(coef_matrix[1, 0])
        se = float(coef_matrix[1, 1])
        t_stat = float(coef_matrix[1, 2])
        p_val = float(coef_matrix[1, 3])

        # Clean up R workspace
        ro.r("rm(.syrinx_tree, .syrinx_df, .syrinx_cdat, .syrinx_pgls, .syrinx_summ, .syrinx_coef, .syrinx_lam)")
        for key in ["_syrinx_species", "_syrinx_acoustic", "_syrinx_geo", "_syrinx_newick"]:
            try:
                del ro.globalenv[key]
            except Exception:
                pass

        return {
            "slope": slope,
            "se": se,
            "t": t_stat,
            "p_value": p_val,
            "pagel_lambda": lambda_val,
            "alpha_secondary": 0.05,
            "significant": p_val < 0.05,
            "method": "caper::pgls",
            "predictor": "mean_geographic_distance",
            "response": "mean_acoustic_distance",
        }

    except Exception as exc:
        logger.warning("PGLS via rpy2/caper failed: %s", exc)
        return {"error": str(exc), "method": "caper::pgls"}


def _mantel_python(
    x: np.ndarray, y: np.ndarray, n_perm: int, method: str
) -> dict[str, Any]:
    """Python fallback Mantel test.

    Parameters
    ----------
    x, y:
        Upper-triangle vectors.
    n_perm:
        Permutations.
    method:
        ``'pearson'`` or ``'spearman'``.
    """
    if method == "spearman":
        obs_r, _ = spearmanr(x, y)
    else:
        obs_r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.RandomState()
    null = []
    for _ in range(n_perm):
        xp = rng.permutation(x)
        if method == "spearman":
            r, _ = spearmanr(xp, y)
        else:
            r = float(np.corrcoef(xp, y)[0, 1])
        null.append(r)
    p = float(np.mean([abs(n) >= abs(obs_r) for n in null]))
    return {"r": float(obs_r), "p_value": p, "method": method, "fallback": True}


# ---------------------------------------------------------------------------
# H2 — Spearman
# ---------------------------------------------------------------------------

def _run_h2(
    cfg: Config,
    descriptive_result: dict[str, Any],
    run_log: Any,
) -> dict[str, Any]:
    """Run H2 Spearman analysis (vocal diversity vs BBS trend).

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    descriptive_result:
        Output from descriptive module with per-region diversity.
    run_log:
        Optional PipelineRunLog.
    """
    bbs_path = cfg.data_path / "bbs_regional_trends.json"
    if not bbs_path.exists():
        logger.warning("BBS regional trends file not found at %s", bbs_path)
        return {"error": "BBS data not found"}

    with bbs_path.open() as fh:
        bbs_data = json.load(fh)

    regions = cfg.bbs_regions
    region_diversity = descriptive_result.get("region_diversity", {})

    diversity_vals = []
    bbs_vals = []
    available_regions = []
    for reg in regions:
        div = region_diversity.get(reg, {}).get("composite_diversity")
        trend = bbs_data.get(reg)
        if div is not None and trend is not None:
            diversity_vals.append(div)
            bbs_vals.append(trend)
            available_regions.append(reg)

    if len(diversity_vals) < 3:
        return {"error": "insufficient regions with data", "n_available": len(diversity_vals)}

    diversity_arr = np.array(diversity_vals)
    bbs_arr = np.array(bbs_vals)

    rho, p_value = spearmanr(diversity_arr, bbs_arr)
    rho = float(rho)
    p_value = float(p_value)

    statistically_sig = p_value < cfg.bonferroni_alpha
    biologically_meaningful = abs(rho) >= cfg.h2_biological_threshold_spearman_rho
    interpretation = _interpret_result(
        statistically_sig, biologically_meaningful, cfg.h2_biological_threshold_spearman_rho, abs(rho), "|ρ|"
    )

    # Individual diversity measures
    component_results = {}
    for comp in ("vocab_size", "mean_complexity", "mean_pairwise_distance"):
        comp_vals = [region_diversity.get(reg, {}).get(comp) for reg in available_regions]
        if all(v is not None for v in comp_vals):
            r2, p2 = spearmanr(comp_vals, bbs_vals)
            component_results[comp] = {"rho": float(r2), "p_value": float(p2)}

    result = {
        "regions": available_regions,
        "rho": rho,
        "p_value": p_value,
        "n_regions": len(available_regions),
        "alpha": cfg.bonferroni_alpha,
        "biological_threshold": cfg.h2_biological_threshold_spearman_rho,
        "statistically_significant": statistically_sig,
        "biologically_meaningful": biologically_meaningful,
        "interpretation": interpretation,
        "component_results": component_results,
        "diversity_values": diversity_vals,
        "bbs_trends": bbs_vals,
    }

    if run_log is not None:
        run_log.record_threshold("H2_spearman_p", p_value, cfg.bonferroni_alpha, statistically_sig, "stage9_H2")
        run_log.record_threshold("H2_rho_absolute", abs(rho), cfg.h2_biological_threshold_spearman_rho, biologically_meaningful, "stage9_H2")

    _make_figure_7(available_regions, diversity_arr, bbs_arr, result, cfg)
    return result


# ---------------------------------------------------------------------------
# H3 — Mantel
# ---------------------------------------------------------------------------

def _run_h3(
    cfg: Config,
    alignment_result: dict[str, Any],
    run_log: Any,
) -> dict[str, Any]:
    """Run H3 within-species Mantel analysis.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    alignment_result:
        Alignment result dict (must be from within_species dataset).
    run_log:
        Optional PipelineRunLog.
    """
    D_acoustic = alignment_result.get("distance_matrix")
    species_names = alignment_result.get("species_names", [])

    if D_acoustic is None or len(species_names) < cfg.min_cells_for_inference:
        logger.info(
            "H3: only %d cells (minimum %d); reporting descriptive statistics only",
            len(species_names), cfg.min_cells_for_inference,
        )
        return {
            "n_cells": len(species_names),
            "min_required": cfg.min_cells_for_inference,
            "status": "insufficient_cells",
            "interpretation": "Fewer than 6 qualifying cells; no inference conducted.",
        }

    # Reconstruct geographic distance matrix from cell IDs (lat_lon format)
    lats, lons = _parse_cell_ids(species_names, cfg)
    D_geo = geographic_distance_matrix(lats, lons)

    acoustic_vec = upper_triangle(D_acoustic)
    geo_vec = upper_triangle(D_geo)

    mantel_result = _run_mantel_r(acoustic_vec, geo_vec, cfg, method="pearson")
    mantel_spearman = _run_mantel_r(acoustic_vec, geo_vec, cfg, method="spearman")

    r = mantel_result.get("r", float("nan"))
    p_value = mantel_result.get("p_value", 1.0)

    statistically_sig = p_value < cfg.bonferroni_alpha
    biologically_meaningful = not np.isnan(r) and r >= cfg.h3_biological_threshold_mantel_r
    interpretation = _interpret_result(
        statistically_sig, biologically_meaningful, cfg.h3_biological_threshold_mantel_r, r, "Mantel r"
    )

    # Scotland vs England/Wales split
    scotland_idx = [i for i, lat in enumerate(lats) if lat > cfg.bbs_scotland_lat_threshold]
    england_idx = [i for i, lat in enumerate(lats) if lat <= cfg.bbs_scotland_lat_threshold]
    split_results = {}
    for name, idx in [("scotland", scotland_idx), ("england_wales", england_idx)]:
        if len(idx) >= cfg.min_cells_for_inference:
            sub_D_ac = D_acoustic[np.ix_(idx, idx)]
            sub_D_geo = D_geo[np.ix_(idx, idx)]
            split_results[name] = _run_mantel_r(
                upper_triangle(sub_D_ac), upper_triangle(sub_D_geo), cfg
            )

    result = {
        "mantel_pearson": mantel_result,
        "mantel_spearman": mantel_spearman,
        "n_cells": len(species_names),
        "alpha": cfg.bonferroni_alpha,
        "biological_threshold": cfg.h3_biological_threshold_mantel_r,
        "statistically_significant": statistically_sig,
        "biologically_meaningful": biologically_meaningful,
        "interpretation": interpretation,
        "regional_split": split_results,
        "cell_size_degrees": cfg.cell_size_degrees,
    }

    if run_log is not None:
        run_log.record_threshold("H3_mantel_p", p_value, cfg.bonferroni_alpha, statistically_sig, "stage9_H3")
        run_log.record_threshold("H3_mantel_r", r, cfg.h3_biological_threshold_mantel_r, biologically_meaningful, "stage9_H3")

    return result


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _make_figure_6(
    D_acoustic: np.ndarray,
    D_molecular: np.ndarray,
    species_names: list[str],
    mrm_result: dict[str, Any],
    cfg: Config,
) -> None:
    """MRM partial regression scatter (Figure 6) and four-corner residuals (Figure 6b).

    Parameters
    ----------
    D_acoustic:
        Acoustic distance matrix.
    D_molecular:
        Molecular distance matrix.
    species_names:
        Species names.
    mrm_result:
        MRM result dict.
    cfg:
        Pipeline configuration.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    acoustic_vec = upper_triangle(D_acoustic)
    molecular_vec = upper_triangle(D_molecular)

    n = len(species_names)
    pair_labels = [
        f"{species_names[i]} — {species_names[j]}"
        for i in range(n) for j in range(i + 1, n)
    ]

    coef = mrm_result.get("molecular_coefficient", 0.0)
    intercept = mrm_result.get("intercept", 0.0)
    fitted = intercept + coef * molecular_vec

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Figure 6: MRM scatter", "Figure 6b: Four-corner residuals"])

    fig.add_trace(
        go.Scatter(x=molecular_vec, y=acoustic_vec, mode="markers",
                   text=pair_labels, marker={"size": 4, "opacity": 0.6},
                   name="species pairs"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=molecular_vec, y=fitted, mode="lines",
                   line={"color": "red", "width": 2}, name="MRM fit"),
        row=1, col=1,
    )

    # Four-corner residuals
    resid = acoustic_vec - fitted
    mol_med = np.median(molecular_vec)
    ac_med = np.median(acoustic_vec)
    colours = []
    for m, a in zip(molecular_vec, acoustic_vec):
        if m <= mol_med and a <= ac_med:
            colours.append("#2ecc71")
        elif m > mol_med and a > ac_med:
            colours.append("#2980b9")
        else:
            colours.append("#e74c3c")

    fig.add_trace(
        go.Scatter(x=molecular_vec, y=acoustic_vec, mode="markers",
                   marker={"color": colours, "size": 5, "opacity": 0.7},
                   text=pair_labels, name="four-corner"),
        row=1, col=2,
    )

    fig.update_layout(title="H1 MRM Analysis")
    fig.write_html(str(fig_dir / "figure_6.html"))
    logger.info("Figure 6 saved")


def _make_figure_7(
    regions: list[str],
    diversity: np.ndarray,
    bbs: np.ndarray,
    h2_result: dict[str, Any],
    cfg: Config,
) -> None:
    """Vocal diversity vs BBS trend scatter (Figure 7).

    Parameters
    ----------
    regions:
        Region names.
    diversity:
        Composite diversity index per region.
    bbs:
        BBS trend per region.
    h2_result:
        H2 result dict.
    cfg:
        Pipeline configuration.
    """
    import plotly.graph_objects as go

    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rho = h2_result.get("rho", float("nan"))
    p = h2_result.get("p_value", float("nan"))

    # Regression line
    if len(diversity) >= 2:
        m, b = np.polyfit(diversity, bbs, 1)
        x_line = np.linspace(diversity.min(), diversity.max(), 50)
        y_line = m * x_line + b
    else:
        x_line = y_line = np.array([])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diversity, y=bbs, mode="markers+text",
        text=regions, textposition="top center",
        marker={"size": 10, "color": "#3498db"},
        name="BBS regions",
    ))
    if len(x_line) > 0:
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line={"color": "red", "dash": "dash"},
            name=f"Trend (ρ={rho:.3f}, p={p:.3f})",
        ))
    fig.update_layout(
        title="H2: Vocal diversity vs BBS regional trend (Willow Warbler)",
        xaxis_title="Composite vocal diversity index",
        yaxis_title="BBS % change since 1995",
    )
    fig.write_html(str(fig_dir / "figure_7.html"))
    logger.info("Figure 7 saved")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_molecular_distances(
    cfg: Config, species_names: list[str]
) -> np.ndarray | None:
    """Load or compute molecular patristic distance matrix.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    species_names:
        Ordered species names.
    """
    import pickle

    dist_path = cfg.data_path / "distances" / "molecular_distances.pkl"
    if dist_path.exists():
        with dist_path.open("rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict) and "D" in data:
            return np.array(data["D"])
        return np.array(data)

    # Try to compute from Alström tree
    tree_path = cfg.data_path / "trees" / "alstrom2018.nwk"
    if not tree_path.exists():
        return None

    try:
        from Bio import Phylo
        import io

        tree = next(Phylo.parse(str(tree_path), "newick"))
        n = len(species_names)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    d = tree.distance(species_names[i], species_names[j])
                    D[i, j] = D[j, i] = d
                except Exception:
                    D[i, j] = D[j, i] = float("nan")
        return D
    except Exception as exc:
        logger.warning("Could not compute molecular distances: %s", exc)
        return None


def _parse_cell_ids(
    cell_names: list[str], cfg: Config
) -> tuple[list[float], list[float]]:
    """Parse lat/lon from cell ID strings of the form 'lat_lon'.

    Parameters
    ----------
    cell_names:
        Cell identifier strings.
    cfg:
        Pipeline configuration.
    """
    lats = []
    lons = []
    half = cfg.cell_size_degrees / 2
    for name in cell_names:
        try:
            lat_str, lon_str = name.split("_", 1)
            lats.append(float(lat_str) + half)
            lons.append(float(lon_str) + half)
        except (ValueError, AttributeError):
            lats.append(0.0)
            lons.append(0.0)
    return lats, lons


def _interpret_result(
    statistically_sig: bool,
    biologically_meaningful: bool,
    bio_threshold: float,
    observed: float,
    metric_name: str,
) -> str:
    """Generate a standard interpretation string.

    Parameters
    ----------
    statistically_sig:
        Whether the result was statistically significant.
    biologically_meaningful:
        Whether the observed effect exceeds the biological threshold.
    bio_threshold:
        Preregistered minimum biologically meaningful effect size.
    observed:
        Observed effect size.
    metric_name:
        Name of the effect size metric.
    """
    if statistically_sig and biologically_meaningful:
        return f"Significant and biologically meaningful ({metric_name}={observed:.3f} ≥ {bio_threshold})."
    if statistically_sig and not biologically_meaningful:
        return (
            f"Statistically significant but below minimum biologically meaningful effect size "
            f"({metric_name}={observed:.3f} < {bio_threshold})."
        )
    return f"Not statistically significant ({metric_name}={observed:.3f})."


def _serialise(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj
