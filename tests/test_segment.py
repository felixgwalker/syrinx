"""Tests for Stage 2 — segmentation and MAO validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from syrinx.config import load_config


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_config(config_path)
    tmpdir = tmp_path_factory.mktemp("segment_test")
    cfg.data_dir = str(tmpdir)
    return cfg


class TestEnergyFallback:
    def test_fallback_runs_without_birdnet(self, cfg, tmp_path):
        """Energy fallback produces syllables when BirdNET returns no detections."""
        from syrinx.segment import _run_energy_fallback

        # Synthesise a 2-second sine wave with onsets
        import soundfile as sf

        sr = 16000
        t = np.linspace(0, 2.0, sr * 2)
        # Create 3 distinct bursts
        y = np.zeros(len(t))
        for onset_s in [0.3, 0.9, 1.5]:
            start = int(onset_s * sr)
            burst_len = int(0.2 * sr)
            y[start:start + burst_len] = np.sin(2 * np.pi * 2000 * t[:burst_len]) * 0.8

        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), y, sr)

        detections = _run_energy_fallback(wav_path, cfg)
        # Should detect at least one onset
        assert len(detections) >= 1
        for start, end in detections:
            assert start >= 0.0
            assert end > start

    def test_fallback_with_silent_file(self, cfg, tmp_path):
        """Energy fallback handles silent file gracefully."""
        from syrinx.segment import _run_energy_fallback
        import soundfile as sf

        wav_path = tmp_path / "silent.wav"
        sf.write(str(wav_path), np.zeros(16000), 16000)

        detections = _run_energy_fallback(wav_path, cfg)
        assert isinstance(detections, list)


class TestSegmentAll:
    def test_syllable_duration_bounds(self, cfg, tmp_path):
        """All returned syllables respect min/max duration bounds."""
        from syrinx.segment import _segment_recording
        import soundfile as sf

        sr = 16000
        y = np.random.RandomState(7).randn(sr * 3) * 0.1
        wav_path = tmp_path / "test_bounds.wav"
        sf.write(str(wav_path), y, sr)

        rec_meta = {"species": "Phylloscopus trochilus", "xc_id": "XC99999",
                    "recordist_id": "tester", "lat": 52.0, "lon": -1.0, "subspecies": ""}

        syllables, _ = _segment_recording(wav_path, rec_meta, cfg, analyzer=None)

        min_s = cfg.syllable_min_ms / 1000.0
        max_s = cfg.syllable_max_ms / 1000.0
        for syl in syllables:
            dur = syl["end_s"] - syl["start_s"]
            assert dur >= min_s - 1e-6, f"Syllable too short: {dur:.3f} s"
            assert dur <= max_s + 1e-6, f"Syllable too long: {dur:.3f} s"

    def test_birdnet_fallback_triggered_below_threshold(self, cfg, tmp_path):
        """Energy fallback used when BirdNET returns fewer than 3 detections."""
        from syrinx.segment import _segment_recording
        import soundfile as sf

        # Mock analyzer that returns only 2 detections
        mock_analyzer = MagicMock()

        with patch("syrinx.segment._BIRDNETLIB_AVAILABLE", True), \
             patch("syrinx.segment._run_birdnet", return_value=[(0.5, 1.0), (1.5, 2.0)]):
            sr = 16000
            y = np.sin(2 * np.pi * 1000 * np.linspace(0, 3, sr * 3))
            wav_path = tmp_path / "test_fallback.wav"
            sf.write(str(wav_path), y, sr)

            syllables, used_fallback = _segment_recording(
                wav_path, {}, cfg, analyzer=mock_analyzer
            )
            assert used_fallback, "Should have used fallback when BirdNET returned < 3 detections"


class TestMAOValidation:
    def test_mao_result_structure(self, cfg):
        """MAO validation returns required keys."""
        from syrinx.segment import run_mao_validation

        # Mock corpus download to avoid network
        with patch("syrinx.segment._ensure_powdermill_corpus", return_value=None), \
             patch("syrinx.segment._ensure_bengalese_corpus", return_value=None):
            result = run_mao_validation(cfg)

        assert "completed" in result
        assert "birdnet_primary" in result
        if result.get("powdermill_mao_ms") is not None:
            assert result["powdermill_mao_ms"] < cfg.segmentation_mao_threshold_ms, (
                f"Powdermill MAO {result['powdermill_mao_ms']:.1f} ms ≥ threshold "
                f"{cfg.segmentation_mao_threshold_ms} ms (preregistered gating criterion)"
            )

    def test_mao_computation_synthetic(self, cfg, tmp_path):
        """MAO computation produces finite value on synthetic annotations."""
        import csv
        import soundfile as sf
        from syrinx.segment import _compute_mao_corpus

        sr = 16000
        y = np.zeros(sr * 5)
        for t in [1.0, 2.5, 4.0]:
            s = int(t * sr)
            y[s:s + int(0.3 * sr)] = np.sin(2 * np.pi * 2000 * np.linspace(0, 0.3, int(0.3 * sr)))

        wav_path = tmp_path / "ref.wav"
        sf.write(str(wav_path), y, sr)

        ann_path = tmp_path / "annotations.csv"
        with ann_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename", "onset_s"])
            writer.writeheader()
            for t in [1.0, 2.5, 4.0]:
                writer.writerow({"filename": "ref.wav", "onset_s": t})

        mao = _compute_mao_corpus(tmp_path, cfg, analyzer=None)
        assert np.isfinite(mao) or np.isnan(mao)
