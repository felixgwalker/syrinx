"""Central configuration loader and validator for the Syrinx pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Validated pipeline configuration loaded from config.yaml."""

    random_seed: int
    recording_cap: int
    min_duration_s: float
    syllable_min_ms: int
    syllable_max_ms: int
    boundary_pad_ms: int
    segmentation_mao_threshold_ms: float
    mao_reference: str
    hdbscan_min_cluster_size_grid: list[int]
    hdbscan_min_samples_grid: list[int]
    hdbscan_max_cycles: int
    bootstrap_n: int
    bootstrap_stability_ari_threshold: float
    cross_recordist_ari_threshold: float
    birdaves_cosine_threshold: float
    spectral_cv_flag_threshold: float
    spectral_cv_flagged_fraction_threshold: float
    zebrafinch_f1_threshold: float
    mismatch_percentiles: list[int]
    primary_mismatch_percentile: int
    gap_open_grid: list[float]
    gap_extend_grid: list[float]
    null_model_permutations: int
    null_model_upper_tail: float
    null_model_binomial_alpha: float
    recording_bootstrap_n: int
    mrm_permutations: int
    mantel_permutations: int
    bonferroni_alpha: float
    h1_biological_threshold_semipartial_r2: float
    h2_biological_threshold_spearman_rho: float
    h3_biological_threshold_mantel_r: float
    cell_size_degrees: float
    min_recordings_per_cell: int
    min_cells_for_inference: int
    use_temporal_features: bool
    breeding_window_palearctic: list[int]
    breeding_window_asian: list[int]
    xeno_canto_base_url: str
    data_dir: str
    output_dir: str

    # Optional fields with defaults
    # Preregistered feature aggregation (see RR §2.9.1 and RR-AMBIGUITY resolution 2026-05-31).
    # "mean" is the only reading consistent with stated 30/36-dim counts; do not change without
    # updating mfcc_feature_dim, pitch_amplitude_dim, and all downstream dimension constants.
    zebrafinch_xc_cap: int = 20

    cepstral_aggregation: str = "mean"

    xc_query_genus: str = "gen:Phylloscopus type:song q:A"
    xc_query_within_species: str = "Phylloscopus trochilus cnt:United+Kingdom type:song q:A"
    xc_within_species_lat_min: float = 49.9
    xc_within_species_lat_max: float = 60.9
    xc_within_species_lon_min: float = -8.6
    xc_within_species_lon_max: float = 1.8
    api_backoff_start_s: float = 1.0
    api_backoff_max_s: float = 60.0
    n_mfcc: int = 13
    frame_length_ms: int = 25
    hop_length_ms: int = 10
    spectral_rolloff_threshold: float = 0.85
    mfcc_feature_dim: int = 30
    pitch_amplitude_dim: int = 6
    temporal_n_frames: int = 50
    gap_penalty_holdout_n_species: int = 10
    rf_null_permutations: int = 999
    power_n_species_grid: list[int] = field(default_factory=lambda: [10, 15, 20, 25, 30, 35, 40, 50, 60])
    power_true_r_grid: list[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    power_n_simulations: int = 1000
    power_h2_fixed_n_regions: int = 7
    bbs_regions: list[str] = field(default_factory=lambda: [
        "Scotland", "Wales", "Northern Ireland",
        "England-SE", "England-SW", "England-Midlands", "England-N",
    ])
    bbs_scotland_lat_threshold: float = 55.0
    h3_cell_size_sensitivity_degrees: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    cell_length_sensitivity_fractions: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    avonet_data_path: str = "data/avonet_traits.csv"
    max_syllables_pairwise_distance: int = 100

    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def hash(self) -> str:
        """SHA-256 of the raw config dict, for provenance tracking."""
        serialised = json.dumps(self._raw, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return self._raw


_REQUIRED_KEYS = [
    "random_seed", "recording_cap", "min_duration_s", "syllable_min_ms",
    "syllable_max_ms", "boundary_pad_ms", "segmentation_mao_threshold_ms",
    "mao_reference", "hdbscan_min_cluster_size_grid", "hdbscan_min_samples_grid",
    "hdbscan_max_cycles", "bootstrap_n", "bootstrap_stability_ari_threshold",
    "cross_recordist_ari_threshold", "birdaves_cosine_threshold",
    "spectral_cv_flag_threshold", "spectral_cv_flagged_fraction_threshold",
    "zebrafinch_f1_threshold", "mismatch_percentiles", "primary_mismatch_percentile",
    "gap_open_grid", "gap_extend_grid", "null_model_permutations",
    "null_model_upper_tail", "null_model_binomial_alpha", "recording_bootstrap_n",
    "mrm_permutations", "mantel_permutations", "bonferroni_alpha",
    "h1_biological_threshold_semipartial_r2", "h2_biological_threshold_spearman_rho",
    "h3_biological_threshold_mantel_r", "cell_size_degrees",
    "min_recordings_per_cell", "min_cells_for_inference", "use_temporal_features",
    "breeding_window_palearctic", "breeding_window_asian",
    "xeno_canto_base_url", "data_dir", "output_dir",
]


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate the pipeline configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Validated configuration object.

    Raises
    ------
    ValueError
        If required keys are missing or values are out of range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Strip optional/extra keys that Config doesn't know about (from yaml extras)
    known = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}
    filtered["_raw"] = raw

    cfg = Config(**filtered)

    _validate(cfg)
    logger.info("Loaded config from %s (hash=%s)", path, cfg.hash)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.cepstral_aggregation not in ("mean", "mean_sd"):
        raise ValueError("cepstral_aggregation must be 'mean' or 'mean_sd'")
    if cfg.cepstral_aggregation == "mean_sd" and cfg.mfcc_feature_dim == 30:
        raise ValueError(
            "cepstral_aggregation='mean_sd' yields >30 cepstral dims; "
            "update mfcc_feature_dim before using this mode. "
            "RR-NOTE: mean_sd is not the preregistered primary analysis setting."
        )
    if cfg.bonferroni_alpha <= 0 or cfg.bonferroni_alpha >= 1:
        raise ValueError("bonferroni_alpha must be in (0, 1)")
    if cfg.null_model_upper_tail <= 0.5 or cfg.null_model_upper_tail >= 1:
        raise ValueError("null_model_upper_tail must be in (0.5, 1)")
    if cfg.primary_mismatch_percentile not in cfg.mismatch_percentiles:
        raise ValueError("primary_mismatch_percentile must be in mismatch_percentiles")
    if cfg.syllable_min_ms >= cfg.syllable_max_ms:
        raise ValueError("syllable_min_ms must be < syllable_max_ms")
    if cfg.mfcc_feature_dim + cfg.pitch_amplitude_dim != 36:
        raise ValueError("mfcc_feature_dim + pitch_amplitude_dim must equal 36")
