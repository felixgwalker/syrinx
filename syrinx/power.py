"""Stage 8 — Power simulation for MRM, Spearman, and Mantel tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_power_simulations(
    cfg: Config,
    n_uk_cells: int = 10,
    run_log: Any = None,
) -> dict[str, Any]:
    """Run power simulations for H1 (MRM), H2 (Spearman), and H3 (Mantel).

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    n_uk_cells:
        Number of qualifying UK geographic cells (determines H3 power).
    run_log:
        Optional PipelineRunLog.

    Returns
    -------
    dict
        Keys: ``h1_power``, ``h2_power``, ``h3_power``, each containing
        power curves and minimum detectable effect sizes.
    """
    np.random.seed(cfg.random_seed)

    logger.info("Running power simulations (n_simulations=%d)…", cfg.power_n_simulations)

    h1_result = _power_h1_mrm(cfg)
    h2_result = _power_h2_spearman(cfg)
    h3_result = _power_h3_mantel(cfg, n_uk_cells)

    result = {
        "h1_power": h1_result,
        "h2_power": h2_result,
        "h3_power": h3_result,
    }

    # Save JSON
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "power_simulations.json").open("w") as fh:
        json.dump(_serialise(result), fh, indent=2)

    # Figures
    _make_power_figures(result, cfg)

    if run_log is not None:
        run_log.record_stage("stage8_power", {
            "h1_min_detectable_r": h1_result.get("min_detectable_r"),
            "h2_min_detectable_rho": h2_result.get("min_detectable_rho"),
            "h3_min_detectable_r": h3_result.get("min_detectable_r"),
        })

    return result


# ---------------------------------------------------------------------------
# H1 — MRM power
# ---------------------------------------------------------------------------

def _power_h1_mrm(cfg: Config) -> dict[str, Any]:
    """Simulate power for H1 MRM at each (n_species, true_r) combination.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    """
    alpha = cfg.bonferroni_alpha
    results: dict[str, Any] = {"grid": []}

    for n_sp in cfg.power_n_species_grid:
        row: dict[str, Any] = {"n_species": n_sp, "power_by_r": {}}
        for true_r in cfg.power_true_r_grid:
            power = _simulate_mrm_power(n_sp, true_r, alpha, cfg)
            row["power_by_r"][str(true_r)] = float(power)
        results["grid"].append(row)

    # Find min detectable r at 80% power for the actual dataset size
    # (use largest n in grid as approximation; updated later with actual n)
    actual_n = max(cfg.power_n_species_grid)
    min_r = _find_min_detectable(
        results["grid"], actual_n, target_power=0.80, param="power_by_r"
    )
    results["min_detectable_r"] = min_r
    results["alpha"] = alpha
    results["n_simulations"] = cfg.power_n_simulations
    logger.info("H1 MRM: min detectable r at 80%% power (n=%d): %.2f", actual_n, min_r or float("nan"))
    return results


def _simulate_mrm_power(
    n_sp: int, true_r: float, alpha: float, cfg: Config
) -> float:
    """Estimate MRM power at given n and effect size via simulation.

    Parameters
    ----------
    n_sp:
        Number of species.
    true_r:
        True Pearson correlation between acoustic and molecular matrices.
    alpha:
        Significance threshold.
    cfg:
        Pipeline configuration.
    """
    rng = np.random.RandomState(cfg.random_seed + int(n_sp * 1000 + true_r * 10000))
    n_pairs = n_sp * (n_sp - 1) // 2
    rejections = 0

    for _ in range(cfg.power_n_simulations):
        acoustic_upper, molecular_upper = _generate_correlated_vectors(
            n_pairs, true_r, rng
        )
        p = _run_mrm_python(acoustic_upper, molecular_upper, n_perm=199)
        if p < alpha:
            rejections += 1

    return rejections / cfg.power_n_simulations


def _generate_correlated_vectors(
    n: int, r: float, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two correlated vectors from bivariate normal with correlation r.

    Parameters
    ----------
    n:
        Length of vectors.
    r:
        Target Pearson correlation.
    rng:
        Random state.
    """
    cov = np.array([[1.0, r], [r, 1.0]])
    L = np.linalg.cholesky(cov)
    Z = rng.randn(2, n)
    X = L @ Z
    return X[0], X[1]


def _run_mrm_python(
    x: np.ndarray, y: np.ndarray, n_perm: int = 199
) -> float:
    """Simple MRM permutation test (Python approximation for power simulation).

    Uses Pearson correlation as the statistic. The rpy2/ecodist MRM is used
    for all real data analyses; this is for simulation only.

    Parameters
    ----------
    x, y:
        Upper-triangle distance vectors.
    n_perm:
        Number of permutations.
    """
    obs = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.RandomState()
    null = [float(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(n_perm)]
    p = float(np.mean([abs(n) >= abs(obs) for n in null]))
    return p


# ---------------------------------------------------------------------------
# H2 — Spearman power
# ---------------------------------------------------------------------------

def _power_h2_spearman(cfg: Config) -> dict[str, Any]:
    """Simulate power for H2 Spearman correlation (n=7 BBS regions).

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    """
    n = cfg.power_h2_fixed_n_regions
    alpha = cfg.bonferroni_alpha
    rho_grid = cfg.power_true_r_grid

    results: dict[str, Any] = {"n_regions": n, "power_by_rho": {}}
    rng = np.random.RandomState(cfg.random_seed + 99)

    for true_rho in rho_grid:
        rejections = 0
        for _ in range(cfg.power_n_simulations):
            x, y = _generate_correlated_vectors(n, true_rho, rng)
            _, p = spearmanr(x, y)
            if p < alpha:
                rejections += 1
        power = rejections / cfg.power_n_simulations
        results["power_by_rho"][str(true_rho)] = float(power)

    min_rho = _find_min_detectable_scalar(
        results["power_by_rho"], rho_grid, target_power=0.80
    )
    results["min_detectable_rho"] = min_rho
    results["alpha"] = alpha
    logger.info("H2 Spearman: min detectable ρ at 80%% power (n=%d): %.2f", n, min_rho or float("nan"))
    return results


# ---------------------------------------------------------------------------
# H3 — Mantel power
# ---------------------------------------------------------------------------

def _power_h3_mantel(cfg: Config, n_cells: int) -> dict[str, Any]:
    """Simulate power for H3 Mantel test at given number of UK cells.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    n_cells:
        Actual number of qualifying UK cells.
    """
    alpha = cfg.bonferroni_alpha
    r_grid = cfg.power_true_r_grid

    results: dict[str, Any] = {"n_cells": n_cells, "power_by_r": {}}
    rng = np.random.RandomState(cfg.random_seed + 77)

    for true_r in r_grid:
        n_pairs = n_cells * (n_cells - 1) // 2
        rejections = 0
        for _ in range(cfg.power_n_simulations):
            x, y = _generate_correlated_vectors(n_pairs, true_r, rng)
            p = _run_mantel_python(x, y, n_perm=199)
            if p < alpha:
                rejections += 1
        power = rejections / cfg.power_n_simulations
        results["power_by_r"][str(true_r)] = float(power)

    min_r = _find_min_detectable_scalar(
        results["power_by_r"], r_grid, target_power=0.80
    )
    results["min_detectable_r"] = min_r
    results["alpha"] = alpha
    logger.info("H3 Mantel: min detectable r at 80%% power (n_cells=%d): %.2f", n_cells, min_r or float("nan"))
    return results


def _run_mantel_python(
    x: np.ndarray, y: np.ndarray, n_perm: int = 199
) -> float:
    """Simple Mantel permutation test (for simulation only).

    Parameters
    ----------
    x, y:
        Upper-triangle distance vectors.
    n_perm:
        Number of permutations.
    """
    obs_r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.RandomState()
    null = [float(np.corrcoef(rng.permutation(x), y)[0, 1]) for _ in range(n_perm)]
    p = float(np.mean([abs(n) >= abs(obs_r) for n in null]))
    return p


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_power_figures(result: dict[str, Any], cfg: Config) -> None:
    """Save power heatmaps as HTML (Figure 4).

    Parameters
    ----------
    result:
        Power simulation results.
    cfg:
        Pipeline configuration.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_dir = cfg.data_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # H1 heatmap
    h1 = result["h1_power"]
    n_vals = [row["n_species"] for row in h1["grid"]]
    r_vals = cfg.power_true_r_grid
    z = [[row["power_by_r"].get(str(r), float("nan")) for r in r_vals] for row in h1["grid"]]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["H1: MRM Power", "H2: Spearman Power", "H3: Mantel Power"])
    fig.add_trace(
        go.Heatmap(z=z, x=[str(r) for r in r_vals], y=[str(n) for n in n_vals],
                   colorscale="Viridis", zmin=0, zmax=1, name="H1"),
        row=1, col=1,
    )

    h2 = result["h2_power"]
    rho_vals = cfg.power_true_r_grid
    z2 = [[h2["power_by_rho"].get(str(rho), float("nan"))] for rho in rho_vals]
    fig.add_trace(
        go.Bar(x=[str(r) for r in rho_vals], y=[h2["power_by_rho"].get(str(r), 0) for r in rho_vals], name="H2"),
        row=1, col=2,
    )

    h3 = result["h3_power"]
    z3 = [h3["power_by_r"].get(str(r), float("nan")) for r in r_vals]
    fig.add_trace(
        go.Bar(x=[str(r) for r in r_vals], y=z3, name="H3"),
        row=1, col=3,
    )

    fig.update_layout(title="Power Simulations — Syrinx Pipeline")
    fig.write_html(str(fig_dir / "figure_4.html"))
    logger.info("Power simulation figure saved")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_min_detectable(
    grid: list[dict[str, Any]],
    n_target: int,
    target_power: float,
    param: str,
) -> float | None:
    for row in grid:
        if row["n_species"] == n_target:
            powers = row[param]
            for r_str, power in sorted(powers.items(), key=lambda x: float(x[0])):
                if power >= target_power:
                    return float(r_str)
    return None


def _find_min_detectable_scalar(
    powers: dict[str, float], r_grid: list[float], target_power: float
) -> float | None:
    for r in sorted(r_grid):
        if powers.get(str(r), 0.0) >= target_power:
            return r
    return None


def _serialise(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj
