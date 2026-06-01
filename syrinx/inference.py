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
        result["h1"] = _run_h1(
            cfg, alignment_result, species_metadata, run_log, molecular_tree,
            descriptive_result=descriptive_result,
        )

        # Nominate-subspecies-only analysis (Item 4)
        D_nom = alignment_result.get("nominate_only_distance_matrix")
        nominate_entities = alignment_result.get("nominate_entities", [])
        if D_nom is not None and len(nominate_entities) >= 4:
            nom_alignment = {
                **alignment_result,
                "distance_matrix": D_nom,
                "species_names": nominate_entities,
            }
            result["h1_nominate"] = _run_h1_nominate(
                cfg, D_nom, nominate_entities,
                alignment_result["species_names"],
                species_metadata, run_log,
            )
            result["nominate_vs_full_comparison"] = _compare_nominate_vs_full(
                result["h1"], result["h1_nominate"]
            )

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
    descriptive_result: dict[str, Any] | None = None,
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
    descriptive_result:
        Descriptive stats dict supplying the four PGLS response traits.
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

    # Bootstrap 95% CI on the MRM partial coefficient (recording resamples)
    mrm_bootstrap_ci = _compute_mrm_bootstrap_ci(
        alignment_result.get("bootstrap_cis", {}),
        molecular_vec, geo_vec, cfg,
    )

    # Permutation-based 95% CI on the MRM partial coefficient (preregistered)
    mrm_permutation_ci = _compute_mrm_permutation_ci(acoustic_vec, molecular_vec, geo_vec, cfg)

    # Secondary Mantel tests
    mantel_pearson = _run_mantel_r(acoustic_vec, molecular_vec, cfg, method="pearson")
    mantel_spearman = _run_mantel_r(acoustic_vec, molecular_vec, cfg, method="spearman")
    mantel_geo = _run_mantel_r(acoustic_vec, geo_vec, cfg, method="pearson")

    # Partial Mantel if both acoustic~molecular and acoustic~geo are significant
    # (α = cfg.bonferroni_alpha per preregistered analysis plan §2.8.3)
    if mantel_pearson.get("p_value", 1.0) < cfg.bonferroni_alpha and mantel_geo.get("p_value", 1.0) < cfg.bonferroni_alpha:
        partial_mantel = _run_partial_mantel_r(acoustic_vec, molecular_vec, geo_vec, cfg)
    else:
        partial_mantel = None

    # Biological meaningfulness
    spr2 = mrm_result.get("semipartial_r2", float("nan"))
    statistically_sig = mrm_result.get("p_value", 1.0) < cfg.bonferroni_alpha
    biologically_meaningful = (
        not np.isnan(spr2) and spr2 >= cfg.h1_biological_threshold_semipartial_r2
    )

    interpretation = _interpret_result(
        statistically_sig, biologically_meaningful,
        cfg.h1_biological_threshold_semipartial_r2, spr2, "semi-partial r²",
    )

    # Complementary PGLS — four preregistered acoustic traits ~ geo (Item 1).
    # Per-species geographic means (off-diagonal row means of D_geo).
    D_ge = D_geo.copy().astype(float)
    np.fill_diagonal(D_ge, np.nan)
    geo_means = np.nanmean(D_ge, axis=1)

    pgls_result: dict[str, Any]
    if descriptive_result is not None:
        trait_values = _extract_species_acoustic_traits(species_names, descriptive_result)
        pgls_result = _run_pgls_four_traits_r(
            species_names, trait_values, geo_means, molecular_tree, cfg
        )
    else:
        pgls_result = {"error": "descriptive_result not provided; PGLS skipped"}

    result = {
        "mrm": mrm_result,
        "mrm_bootstrap_ci": mrm_bootstrap_ci,
        "mrm_permutation_ci": mrm_permutation_ci,
        "pgls_four_traits": pgls_result,
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


def _run_h1_nominate(
    cfg: Config,
    D_nom: np.ndarray,
    nominate_entities: list[str],
    full_species_names: list[str],
    species_metadata: list[dict[str, Any]],
    run_log: Any,
) -> dict[str, Any]:
    """Lightweight H1 MRM for nominate-subspecies-only recordings.

    Loads molecular distances, subsets to nominate species, and runs MRM.
    Only the primary MRM comparison is run (no secondary Mantel/PGLS).

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    D_nom:
        Acoustic distance matrix for nominate entities.
    nominate_entities:
        Species labels for the nominate-only analysis.
    full_species_names:
        Full species list from the primary analysis (used to index into
        the full molecular distance matrix).
    species_metadata:
        Per-species metadata with lat/lon.
    run_log:
        Optional PipelineRunLog.
    """
    D_molecular_full = _load_molecular_distances(cfg, full_species_names)
    if D_molecular_full is None:
        return {"error": "molecular distances unavailable"}

    # Index into full molecular matrix
    full_idx = {sp: i for i, sp in enumerate(full_species_names)}
    nom_indices = [full_idx[sp] for sp in nominate_entities if sp in full_idx]
    if len(nom_indices) < 4:
        return {"error": "insufficient nominate species with molecular data"}

    D_mol_nom = D_molecular_full[np.ix_(nom_indices, nom_indices)]

    meta_by_sp = {m["species"]: m for m in species_metadata}
    lats = [meta_by_sp.get(sp, {}).get("lat", 0.0) or 0.0 for sp in nominate_entities]
    lons = [meta_by_sp.get(sp, {}).get("lon", 0.0) or 0.0 for sp in nominate_entities]
    D_geo_nom = geographic_distance_matrix(lats, lons)

    acoustic_vec = upper_triangle(D_nom)
    molecular_vec = upper_triangle(D_mol_nom)
    geo_vec = upper_triangle(D_geo_nom)

    mrm_result = _run_mrm_r(acoustic_vec, molecular_vec, geo_vec, cfg)

    spr2 = mrm_result.get("semipartial_r2", float("nan"))
    statistically_sig = mrm_result.get("p_value", 1.0) < cfg.bonferroni_alpha
    biologically_meaningful = (
        not np.isnan(spr2) and spr2 >= cfg.h1_biological_threshold_semipartial_r2
    )

    return {
        "mrm": mrm_result,
        "n_species": len(nominate_entities),
        "statistically_significant": statistically_sig,
        "biologically_meaningful": biologically_meaningful,
        "note": "Nominate-subspecies-only analysis: non-nominate recordings excluded.",
    }


def _compare_nominate_vs_full(
    full_result: dict[str, Any],
    nominate_result: dict[str, Any],
) -> dict[str, Any]:
    """Flag where nominate-only and full-dataset H1 conclusions diverge.

    Parameters
    ----------
    full_result:
        H1 result from the full (all-subspecies) analysis.
    nominate_result:
        H1 result from the nominate-only analysis.
    """
    full_sig = full_result.get("statistically_significant", False)
    nom_sig = nominate_result.get("statistically_significant", False)
    full_bio = full_result.get("biologically_meaningful", False)
    nom_bio = nominate_result.get("biologically_meaningful", False)

    divergent = (full_sig != nom_sig) or (full_bio != nom_bio)

    return {
        "divergent": divergent,
        "full_statistically_significant": full_sig,
        "nominate_statistically_significant": nom_sig,
        "full_biologically_meaningful": full_bio,
        "nominate_biologically_meaningful": nom_bio,
        "note": (
            "Nominate-only and full-dataset analyses reach different conclusions. "
            "This may indicate that non-nominate subspecies drive the main result."
            if divergent else
            "Nominate-only and full-dataset analyses agree."
        ),
    }


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


def _extract_species_acoustic_traits(
    species_names: list[str],
    descriptive_result: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extract the four preregistered PGLS response variables from descriptive stats.

    Returns a dict mapping trait label → per-species ndarray (NaN for missing species).

    Traits
    ------
    mean_C:
        Per-species syntactic complexity C = H1 − H2, averaged across syllables.
    mean_peak_freq:
        Per-species mean peak frequency (Hz).
    mean_freq_range:
        Per-species mean frequency range = peak − min frequency (Hz).
    mean_fm_depth:
        Per-species mean FM depth (approximated as frequency range;
        true instantaneous FM depth is not separately tracked in the feature set).
    """
    species_stats = descriptive_result.get("species_stats", {})
    trait_keys = {
        "mean_C": "C",
        "mean_peak_freq": "mean_peak_freq",
        "mean_freq_range": "mean_freq_range",
        "mean_fm_depth": "mean_fm_depth",
    }
    result: dict[str, np.ndarray] = {}
    for label, stat_key in trait_keys.items():
        result[label] = np.array([
            float(species_stats.get(sp, {}).get(stat_key, float("nan")))
            for sp in species_names
        ])
    return result


def _compute_mrm_bootstrap_ci(
    bootstrap_cis: dict[str, Any],
    molecular_vec: np.ndarray,
    geo_vec: np.ndarray,
    cfg: Config,
) -> dict[str, Any] | None:
    """Compute 95% CI on the MRM partial coefficient via recording bootstrap.

    For each replicate acoustic distance vector stored in *bootstrap_cis*,
    the MRM molecular partial coefficient is estimated and the 2.5th/97.5th
    percentiles are taken as the CI.  Uses R/ecodist when available; falls
    back to sklearn OLS (coefficient only, not permutation p-value) otherwise
    — the OLS partial coefficient equals the MRM coefficient.

    Parameters
    ----------
    bootstrap_cis:
        Dict returned by :func:`~syrinx.align._recording_bootstrap`; must
        contain ``"bootstrap_acoustic_vecs"``.
    molecular_vec, geo_vec:
        Upper-triangle molecular and geographic distance vectors for the
        same species set.
    cfg:
        Pipeline configuration.
    """
    boot_vecs = bootstrap_cis.get("bootstrap_acoustic_vecs", [])
    if not boot_vecs:
        return None

    r_available = False
    try:
        import rpy2.robjects  # noqa: F401
        r_available = True
    except Exception:
        pass

    coefs: list[float] = []
    for v in boot_vecs:
        av = np.array(v)
        if av.shape[0] < 3:
            continue
        if r_available:
            r = _run_mrm_r(av, molecular_vec, geo_vec, cfg)
            coef = r.get("molecular_coefficient")
            if coef is not None and not np.isnan(float(coef)):
                coefs.append(float(coef))
        else:
            # OLS fallback: the MRM molecular coefficient equals the OLS
            # partial coefficient from the multiple regression.
            try:
                from sklearn.linear_model import LinearRegression
                X = np.column_stack([molecular_vec, geo_vec])
                if X.shape[0] >= X.shape[1] + 1:
                    lr = LinearRegression().fit(X, av)
                    coefs.append(float(lr.coef_[0]))
            except Exception:
                pass

    if len(coefs) < 2:
        return None

    return {
        "ci_lower": float(np.percentile(coefs, 2.5)),
        "ci_upper": float(np.percentile(coefs, 97.5)),
        "n_replicates": len(coefs),
        "method": "recording_bootstrap_mrm_coefficient",
        "coefficient_estimator": "rpy2_ecodist_mrm" if r_available else "sklearn_ols_fallback",
    }


def _compute_mrm_permutation_ci(
    acoustic_vec: np.ndarray,
    molecular_vec: np.ndarray,
    geo_vec: np.ndarray,
    cfg: Config,
) -> dict[str, Any] | None:
    """Compute permutation-based 95% CI on the MRM molecular partial coefficient.

    Runs MRM on each permutation of the acoustic distance vector to build a null
    distribution for the molecular coefficient.  The 2.5th and 97.5th percentiles
    of that distribution form the permutation CI (preregistered output, §2.8.2).

    Uses the sklearn OLS estimator (MRM coefficient = OLS partial coefficient) to
    avoid 9999 rpy2/R invocations; the R path is used only for the primary result.

    Parameters
    ----------
    acoustic_vec, molecular_vec, geo_vec:
        Upper-triangle distance vectors.
    cfg:
        Pipeline configuration (uses ``cfg.mrm_permutations``).
    """
    if acoustic_vec.shape[0] < 4:
        return None

    rng = np.random.RandomState(cfg.random_seed + 7)
    coefs: list[float] = []
    try:
        from sklearn.linear_model import LinearRegression

        X_predictors = np.column_stack([molecular_vec, geo_vec])
        if X_predictors.shape[0] < X_predictors.shape[1] + 1:
            return None
        for _ in range(cfg.mrm_permutations):
            perm_acoustic = rng.permutation(acoustic_vec)
            lr = LinearRegression().fit(X_predictors, perm_acoustic)
            coefs.append(float(lr.coef_[0]))
    except Exception as exc:
        logger.warning("Permutation CI computation failed: %s", exc)
        return None

    if len(coefs) < 2:
        return None

    return {
        "ci_lower": float(np.percentile(coefs, 2.5)),
        "ci_upper": float(np.percentile(coefs, 97.5)),
        "n_permutations": len(coefs),
        "method": "permutation_null_distribution_mrm_coefficient",
    }


def _run_pgls_four_traits_r(
    species_names: list[str],
    trait_values: dict[str, np.ndarray],
    geo_means: np.ndarray,
    molecular_tree: Any,
    cfg: Config,
) -> dict[str, Any]:
    """Run PGLS under the Alström topology for each of the four preregistered traits.

    Each trait is regressed on per-species mean geographic distance (off-diagonal
    row mean of D_geo), i.e. ``trait ~ geo_mean``.  The geographic predictor was
    chosen as the closest measurable proxy for the RR's "species-level predictors"
    (§2.10.3), which are not named explicitly in the registered report; this choice
    must be confirmed or revised against the final RR wording before submission.
    α = 0.05 uncorrected (preregistered secondary analysis, §2.10.3).

    Parameters
    ----------
    species_names:
        Species labels in alignment order.
    trait_values:
        Dict mapping trait label → per-species ndarray (NaN for missing).
    geo_means:
        Per-species mean geographic distance (off-diagonal row mean of D_geo).
    molecular_tree:
        Bio.Phylo reference tree for PGLS correlation structure.
    cfg:
        Pipeline configuration.
    """
    if molecular_tree is None:
        return {"error": "no molecular tree provided for PGLS"}
    if len(species_names) < 4:
        return {"error": f"too few species for PGLS (n={len(species_names)})"}

    results: dict[str, Any] = {}
    for trait_label, trait_vec in trait_values.items():
        valid_mask = ~np.isnan(trait_vec)
        n_valid = int(valid_mask.sum())
        if n_valid < 4:
            results[trait_label] = {
                "error": f"insufficient valid values (n={n_valid})",
                "response": trait_label,
            }
            continue
        valid_species = [sp for sp, v in zip(species_names, valid_mask) if v]
        valid_trait = trait_vec[valid_mask]
        valid_geo = geo_means[valid_mask]
        results[trait_label] = _run_pgls_single_trait_r(
            valid_species, valid_trait, valid_geo, molecular_tree, trait_label, cfg
        )
    return results


def _run_pgls_single_trait_r(
    species_names: list[str],
    trait_values: np.ndarray,
    geo_means: np.ndarray,
    molecular_tree: Any,
    trait_label: str,
    cfg: Config,
) -> dict[str, Any]:
    """Run one caper::pgls model: trait ~ geo under Alström topology.

    Parameters
    ----------
    species_names:
        Species labels (already filtered to valid observations).
    trait_values:
        Per-species trait means (no NaN).
    geo_means:
        Per-species mean geographic distances (no NaN).
    molecular_tree:
        Bio.Phylo reference tree.
    trait_label:
        Human-readable name used in logs and output JSON.
    cfg:
        Pipeline configuration.
    """
    try:
        import io
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from Bio import Phylo

        numpy2ri.activate()
        importr("ape")
        importr("caper")

        buf = io.StringIO()
        Phylo.write(molecular_tree, buf, "newick")
        newick_str = buf.getvalue().strip()

        ro.globalenv["_syrinx_species"] = ro.StrVector(species_names)
        ro.globalenv["_syrinx_trait"] = ro.FloatVector(trait_values.tolist())
        ro.globalenv["_syrinx_geo"] = ro.FloatVector(geo_means.tolist())
        ro.globalenv["_syrinx_newick"] = ro.StrVector([newick_str])

        ro.r("""
            .syrinx_tree <- ape::read.tree(text = `_syrinx_newick`)
            .syrinx_df <- data.frame(
                species = `_syrinx_species`,
                trait   = `_syrinx_trait`,
                geo     = `_syrinx_geo`,
                stringsAsFactors = FALSE
            )
            .syrinx_cdat <- caper::comparative.data(
                phy       = .syrinx_tree,
                data      = .syrinx_df,
                names.col = "species",
                warn.dropped = FALSE
            )
            .syrinx_pgls <- caper::pgls(trait ~ geo, data = .syrinx_cdat, lambda = "ML")
            .syrinx_summ <- summary(.syrinx_pgls)
            .syrinx_coef <- coef(.syrinx_summ)
            .syrinx_lam  <- .syrinx_pgls$param[["lambda"]]
        """)

        coef_matrix = np.array(ro.r[".syrinx_coef"])
        lambda_val = float(ro.r[".syrinx_lam"][0])

        if coef_matrix.shape[0] < 2:
            raise ValueError("PGLS coef matrix has unexpected shape")

        slope = float(coef_matrix[1, 0])
        se = float(coef_matrix[1, 1])
        t_stat = float(coef_matrix[1, 2])
        p_val = float(coef_matrix[1, 3])

        ro.r(
            "rm(.syrinx_tree, .syrinx_df, .syrinx_cdat, "
            ".syrinx_pgls, .syrinx_summ, .syrinx_coef, .syrinx_lam)"
        )
        for key in ["_syrinx_species", "_syrinx_trait", "_syrinx_geo", "_syrinx_newick"]:
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
            "response": trait_label,
        }

    except Exception as exc:
        logger.warning("PGLS for %s via rpy2/caper failed: %s", trait_label, exc)
        return {"error": str(exc), "method": "caper::pgls", "response": trait_label}


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

    # Cell-size sensitivity Mantel tests (Item 5)
    cell_size_sensitivity: dict[str, Any] = {}
    for cell_size_str, sens in alignment_result.get("h3_cell_size_sensitivity", {}).items():
        cell_size_f = float(cell_size_str)
        if sens.get("insufficient_cells"):
            cell_size_sensitivity[cell_size_str] = {
                "status": "insufficient_cells",
                "cell_size_degrees": cell_size_f,
            }
            continue
        ents = sens.get("entities", [])
        D_s = sens.get("distance_matrix")
        if D_s is None or len(ents) < cfg.min_cells_for_inference:
            cell_size_sensitivity[cell_size_str] = {
                "status": "insufficient_cells",
                "n_cells": len(ents),
                "cell_size_degrees": cell_size_f,
            }
            continue
        lats_s, lons_s = _parse_cell_ids_with_size(ents, cell_size_f)
        D_geo_s = geographic_distance_matrix(lats_s, lons_s)
        mr_s = _run_mantel_r(upper_triangle(D_s), upper_triangle(D_geo_s), cfg, method="pearson")
        cell_size_sensitivity[cell_size_str] = {
            "mantel_pearson": mr_s,
            "n_cells": len(ents),
            "cell_size_degrees": cell_size_f,
            "statistically_significant": mr_s.get("p_value", 1.0) < cfg.bonferroni_alpha,
        }

    # String-length sensitivity Mantel tests (Item 6)
    length_sensitivity: dict[str, Any] = {}
    for frac_str, length_data in alignment_result.get("h3_length_sensitivity", {}).items():
        if length_data.get("insufficient_cells"):
            length_sensitivity[frac_str] = {
                "status": "insufficient_cells",
                "truncate_fraction": float(frac_str),
            }
            continue
        ents_f = length_data.get("entities", [])
        D_f = length_data.get("distance_matrix")
        if D_f is None or len(ents_f) < cfg.min_cells_for_inference:
            length_sensitivity[frac_str] = {
                "status": "insufficient_cells",
                "n_cells": len(ents_f),
                "truncate_fraction": float(frac_str),
            }
            continue
        lats_f, lons_f = _parse_cell_ids(ents_f, cfg)
        D_geo_f = geographic_distance_matrix(lats_f, lons_f)
        mr_f = _run_mantel_r(upper_triangle(D_f), upper_triangle(D_geo_f), cfg, method="pearson")
        length_sensitivity[frac_str] = {
            "mantel_pearson": mr_f,
            "n_cells": len(ents_f),
            "truncate_fraction": float(frac_str),
            "statistically_significant": mr_f.get("p_value", 1.0) < cfg.bonferroni_alpha,
        }

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
        "cell_size_sensitivity": cell_size_sensitivity,
        "length_sensitivity": length_sensitivity,
    }

    if run_log is not None:
        run_log.record_threshold("H3_mantel_p", p_value, cfg.bonferroni_alpha, statistically_sig, "stage9_H3")
        run_log.record_threshold("H3_mantel_r", r, cfg.h3_biological_threshold_mantel_r, biologically_meaningful, "stage9_H3")

    return result


# ---------------------------------------------------------------------------
# AVONET helper
# ---------------------------------------------------------------------------

def _load_avonet_habitat(cfg: Config, species_names: list[str]) -> dict[str, str]:
    """Load AVONET habitat codes for the given species if the file exists.

    Looks for ``cfg.avonet_data_path`` (CSV with columns ``Species1``/``species``
    and ``Habitat``/``habitat``).  Returns an empty dict if the file is absent or
    unreadable — the figure is generated without annotation in that case.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    species_names:
        Species labels to look up.
    """
    import csv

    candidates = [
        Path(cfg.avonet_data_path),
        cfg.data_path / "avonet_traits.csv",
    ]
    avonet_path = next((p for p in candidates if p.exists()), None)
    if avonet_path is None:
        logger.debug("AVONET file not found; habitat annotation will be skipped")
        return {}

    try:
        habitat_map: dict[str, str] = {}
        with avonet_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("Species1") or row.get("species") or ""
                habitat = row.get("Habitat") or row.get("habitat") or ""
                if name and habitat:
                    habitat_map[name.strip()] = habitat.strip()
        result = {sp: habitat_map[sp] for sp in species_names if sp in habitat_map}
        logger.info("AVONET habitat loaded for %d/%d species", len(result), len(species_names))
        return result
    except Exception as exc:
        logger.warning("AVONET load failed: %s", exc)
        return {}


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

    # Four-corner residuals with AVONET habitat-concordance annotation (§2.10.5)
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

    # AVONET habitat-concordance lookup
    avonet_habitat = _load_avonet_habitat(cfg, species_names)
    avonet_available = bool(avonet_habitat)

    # Build pair-level habitat-concordance flag for hover text
    habitat_texts = []
    for i in range(n):
        for j in range(i + 1, n):
            h_i = avonet_habitat.get(species_names[i], "")
            h_j = avonet_habitat.get(species_names[j], "")
            if h_i and h_j:
                concordant = h_i == h_j
                habitat_texts.append(
                    f"{species_names[i]}({h_i}) — {species_names[j]}({h_j}) "
                    f"[{'same' if concordant else 'different'} habitat]"
                )
            else:
                habitat_texts.append(pair_labels[len(habitat_texts)])

    fig.add_trace(
        go.Scatter(
            x=molecular_vec, y=acoustic_vec, mode="markers",
            marker={"color": colours, "size": 5, "opacity": 0.7},
            text=habitat_texts if avonet_available else pair_labels,
            name="four-corner",
        ),
        row=1, col=2,
    )

    avonet_note = (
        "AVONET habitat concordance annotated in hover text."
        if avonet_available else
        f"AVONET data not found at {cfg.avonet_data_path}; habitat annotation skipped."
    )
    fig.update_layout(
        title=f"H1 MRM Analysis — {avonet_note}",
    )
    fig.write_html(str(fig_dir / "figure_6.html"))
    logger.info("Figure 6 saved (AVONET annotation: %s)", "yes" if avonet_available else "no")


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
    return _parse_cell_ids_with_size(cell_names, cfg.cell_size_degrees)


def _parse_cell_ids_with_size(
    cell_names: list[str], cell_size: float
) -> tuple[list[float], list[float]]:
    """Parse lat/lon from cell IDs, adding half the given cell size to reach centres.

    Parameters
    ----------
    cell_names:
        Cell identifier strings (``"lat_lon"`` format, floor-of-cell corner).
    cell_size:
        Cell edge length in degrees.
    """
    lats = []
    lons = []
    half = cell_size / 2
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
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj
