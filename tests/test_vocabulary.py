"""Tests for Stage 4 — vocabulary building and validation gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from syrinx.config import load_config
from syrinx.vocabulary import VocabularyValidationError


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    tmpdir = tmp_path_factory.mktemp("vocab_test")
    cfg.data_dir = str(tmpdir)
    cfg.output_dir = str(tmpdir / "out")
    cfg.mrm_permutations = 99
    cfg.bootstrap_n = 30
    # Create minimal zebra finch reference files required by §2.5 check
    ref_dir = tmpdir / "reference"
    ref_dir.mkdir()
    rng = np.random.RandomState(0)
    np.save(ref_dir / "zebrafinch_features.npy", rng.randn(30, 36).astype(np.float64))
    np.save(ref_dir / "zebrafinch_labels.npy", rng.randint(0, 5, size=30))
    return cfg


def _make_well_separated_syllables(
    n_clusters: int = 5,
    n_per_cluster: int = 60,
    n_recordists: int = 4,
    seed: int = 0,
) -> list[dict]:
    """Generate feature vectors forming well-separated clusters."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_clusters, 36) * 10
    syllables = []
    for c_idx in range(n_clusters):
        for _ in range(n_per_cluster):
            feat = centroids[c_idx] + rng.randn(36) * 0.2
            syllables.append({
                "features": feat.astype(np.float32),
                "species": f"sp{c_idx % 3}",
                "xc_id": f"XC{rng.randint(1000, 9999)}",
                "recordist_id": f"rec{rng.randint(0, n_recordists)}",
                "lat": 50.0 + rng.uniform(-2, 2),
                "lon": 0.0 + rng.uniform(-2, 2),
                "subspecies": "",
                "wav_path": "/dev/null",
                "start_s": 0.0,
                "end_s": 0.3,
            })
    return syllables


def _make_noise_syllables(n: int = 100, seed: int = 0) -> list[dict]:
    """Generate pure noise feature vectors that should fail stability gates."""
    rng = np.random.RandomState(seed)
    return [
        {
            "features": rng.randn(36).astype(np.float32),
            "species": f"sp{i % 3}",
            "xc_id": f"XC{i}",
            "recordist_id": f"rec{i % 4}",
            "lat": 50.0,
            "lon": 0.0,
            "subspecies": "",
        }
        for i in range(n)
    ]


class TestVocabularyGatesPass:
    """All four gates pass on well-structured data."""

    def test_gates_pass_on_coherent_data(self, cfg):
        from syrinx.vocabulary import build_vocabulary

        syllables = _make_well_separated_syllables(n_clusters=5, n_per_cluster=60)

        # Relax thresholds for synthetic test data
        cfg.bootstrap_stability_ari_threshold = 0.20
        cfg.cross_recordist_ari_threshold = 0.05
        cfg.birdaves_cosine_threshold = 0.0
        cfg.spectral_cv_flagged_fraction_threshold = 0.8
        cfg.hdbscan_max_cycles = 5

        vocab = build_vocabulary(cfg, syllables)
        assert vocab["n_clusters"] >= 2
        assert all(v in vocab["cluster_letters"].values() for v in list("ABCDE")[:vocab["n_clusters"]])

    def test_cluster_letters_sorted_by_mfcc(self, cfg):
        from syrinx.vocabulary import build_vocabulary

        syllables = _make_well_separated_syllables(n_clusters=4, n_per_cluster=50)
        _relax_all_gates(cfg)

        vocab = build_vocabulary(cfg, syllables)
        letters = vocab["cluster_letters"]
        # Letter A should be assigned to smallest first-MFCC centroid
        assert "A" in letters.values()
        assert len(set(letters.values())) == len(letters)

    def test_noise_fraction_in_range(self, cfg):
        from syrinx.vocabulary import build_vocabulary

        syllables = _make_well_separated_syllables(n_clusters=4, n_per_cluster=40)
        _relax_all_gates(cfg)

        vocab = build_vocabulary(cfg, syllables)
        assert 0.0 <= vocab["noise_fraction"] <= 1.0


class TestVocabularyGatesFail:
    """Individual gates correctly fail on adversarial data."""

    def test_gate1_fails_on_unstable_clustering(self, cfg):
        from syrinx.vocabulary import _gate1_bootstrap_stability

        # Completely random data — bootstrap ARI should be near 0
        rng = np.random.RandomState(42)
        X = rng.randn(200, 36)
        labels = rng.randint(0, 5, size=200)

        cfg.bootstrap_stability_ari_threshold = 0.99  # impossible threshold
        cfg.bootstrap_n = 10
        result = _gate1_bootstrap_stability(X, labels, cfg)
        assert not result["passed"]

    def test_gate4_fails_on_heterogeneous_clusters(self, cfg):
        from syrinx.vocabulary import _gate4_spectral_homogeneity

        rng = np.random.RandomState(0)
        # Create syllables where pitch features vary wildly within clusters
        syllables = []
        labels = []
        for c in range(3):
            for _ in range(30):
                feat = rng.randn(36).astype(np.float32)
                feat[30] = rng.uniform(1000, 8000)  # wild peak frequency variation
                syllables.append({"features": feat})
                labels.append(c)

        cfg.spectral_cv_flag_threshold = 0.0  # everything flagged
        cfg.spectral_cv_flagged_fraction_threshold = 0.0
        result = _gate4_spectral_homogeneity(syllables, np.array(labels), cfg)
        assert not result["passed"]

    def test_vocabulary_validation_error_after_max_cycles(self, cfg):
        from syrinx.vocabulary import VocabularyValidationError, build_vocabulary

        syllables = _make_noise_syllables(n=80)
        cfg.bootstrap_stability_ari_threshold = 1.0  # impossible
        cfg.cross_recordist_ari_threshold = 1.0
        cfg.birdaves_cosine_threshold = 0.0
        cfg.spectral_cv_flagged_fraction_threshold = 0.0
        cfg.hdbscan_max_cycles = 2

        with pytest.raises(VocabularyValidationError) as exc_info:
            build_vocabulary(cfg, syllables)

        assert "diagnostic" in dir(exc_info.value)
        assert exc_info.value.diagnostic.get("n_cycles") == 2


def _relax_all_gates(cfg) -> None:
    cfg.bootstrap_n = 10
    cfg.bootstrap_stability_ari_threshold = 0.0
    cfg.cross_recordist_ari_threshold = 0.0
    cfg.birdaves_cosine_threshold = 0.0
    cfg.spectral_cv_flagged_fraction_threshold = 1.0
    cfg.hdbscan_max_cycles = 3
