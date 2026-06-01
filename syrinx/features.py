"""Stage 3 — MFCC, spectral, pitch, and amplitude feature extraction."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .utils import save_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features(
    cfg: Config,
    syllables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract acoustic features for every syllable.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    syllables:
        Syllable records from :mod:`segment` (each with ``wav_path``,
        ``start_s``, ``end_s``).

    Returns
    -------
    list[dict]
        Input records augmented with ``features`` (36-dim ndarray) and,
        when ``cfg.use_temporal_features`` is True, ``mfcc_temporal``
        (shape ``(26, T)``).
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    result = []
    failed = 0
    for syl in syllables:
        try:
            augmented = _extract_syllable(syl, cfg)
            result.append(augmented)
        except Exception as exc:
            logger.debug(
                "Feature extraction failed for %s [%.3f–%.3f]: %s",
                syl.get("wav_path"), syl.get("start_s"), syl.get("end_s"), exc,
            )
            failed += 1

    logger.info(
        "Extracted features for %d/%d syllables (%d failed)",
        len(result), len(syllables), failed,
    )
    return result


def save_features(
    syllables: list[dict[str, Any]],
    cfg: Config,
    species: str,
    n_syllables_total: int,
) -> Path:
    """Persist per-species feature matrix to disk.

    Parameters
    ----------
    syllables:
        Feature-augmented syllable records for one species.
    cfg:
        Pipeline configuration.
    species:
        Species name used in the filename.
    n_syllables_total:
        Total syllables across all species (for provenance metadata).

    Returns
    -------
    Path
        Path to the saved pickle file.
    """
    feat_dir = cfg.data_path / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    safe_name = species.replace(" ", "_").replace("/", "_")
    path = feat_dir / f"{safe_name}_features.pkl"

    data = {
        "species": species,
        "syllables": [
            {k: v for k, v in s.items() if k != "features" and k != "mfcc_temporal"}
            for s in syllables
        ],
        "feature_matrix": np.vstack([s["features"] for s in syllables]),
    }
    if cfg.use_temporal_features:
        data["temporal_matrix"] = np.array(
            [s["mfcc_temporal"] for s in syllables if "mfcc_temporal" in s]
        )

    save_array(
        data,
        path,
        config_hash=cfg.hash,
        random_seed=cfg.random_seed,
        n_species=1,
        n_syllables_total=n_syllables_total,
    )
    return path


# ---------------------------------------------------------------------------
# Per-syllable extraction
# ---------------------------------------------------------------------------

def _extract_syllable(syl: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Extract all features from a single syllable clip.

    Parameters
    ----------
    syl:
        Syllable record with ``wav_path``, ``start_s``, ``end_s``.
    cfg:
        Pipeline configuration.

    Returns
    -------
    dict
        Input dict augmented with ``features`` (36-dim ndarray).
    """
    import librosa

    y, sr = _load_clip(syl, cfg)

    hop = int(sr * cfg.hop_length_ms / 1000.0)
    win = int(sr * cfg.frame_length_ms / 1000.0)

    mfcc_vec, mfcc_temporal = _compute_mfcc_features(y, sr, cfg, hop, win)
    pitch_amp_vec, fm_depth = _compute_pitch_amplitude_features(y, sr, cfg, hop, win)

    features = np.concatenate([mfcc_vec, pitch_amp_vec])
    assert features.shape == (36,), f"Expected 36 features, got {features.shape}"

    result = dict(syl)
    result["features"] = features
    result["fm_depth_mean"] = fm_depth
    if cfg.use_temporal_features:
        result["mfcc_temporal"] = mfcc_temporal
    return result


def _load_clip(syl: dict[str, Any], cfg: Config) -> tuple[np.ndarray, int]:
    """Load a syllable clip from a wav file.

    Parameters
    ----------
    syl:
        Syllable record.
    cfg:
        Pipeline configuration.
    """
    import librosa

    wav_path = syl["wav_path"]
    start_s = float(syl["start_s"])
    end_s = float(syl["end_s"])
    duration = end_s - start_s

    y, sr = librosa.load(
        wav_path,
        sr=16000,
        offset=start_s,
        duration=duration,
        mono=True,
    )
    return y, sr


def _compute_mfcc_features(
    y: np.ndarray, sr: int, cfg: Config, hop: int, win: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MFCC + delta features, returning (30-dim aggregate, temporal matrix).

    Preregistered aggregation (cepstral_aggregation='mean', RR §2.9.1):
        13 MFCC means + 13 delta means + 4 spectral means = 30 dims.
    This is the only reading consistent with the RR's stated 30/36-dim counts.

    Parameters
    ----------
    y:
        Audio signal.
    sr:
        Sample rate.
    cfg:
        Pipeline configuration.
    hop:
        Hop length in samples.
    win:
        Window length in samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mfcc_vec, temporal_matrix)`` where ``mfcc_vec`` is shape ``(30,)``
        and ``temporal_matrix`` is shape ``(26, T)`` with zero-padding/truncation.
    """
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc, hop_length=hop, n_fft=win)
    delta = librosa.feature.delta(mfcc)

    # Spectral features (4 × T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop, n_fft=win)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop, n_fft=win)
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop, n_fft=win,
        roll_percent=cfg.spectral_rolloff_threshold,
    )
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop, frame_length=win)

    # Stack mfcc + delta → (26 × T) for temporal mode
    mfcc_delta_stack = np.vstack([mfcc, delta])  # (26, T)

    # 30-dim aggregate (mean-only): 13 MFCC means + 13 delta means + 4 spectral means = 30
    spectral_means = np.array([
        centroid.mean(), bandwidth.mean(), rolloff.mean(), zcr.mean(),
    ])
    mfcc_vec = np.concatenate([
        mfcc.mean(axis=1),    # 13
        delta.mean(axis=1),   # 13
        spectral_means,       # 4
    ])

    # Temporal: zero-pad/truncate mfcc_delta_stack to cfg.temporal_n_frames columns
    T = cfg.temporal_n_frames
    if mfcc_delta_stack.shape[1] >= T:
        temporal = mfcc_delta_stack[:, :T]
    else:
        pad = T - mfcc_delta_stack.shape[1]
        temporal = np.pad(mfcc_delta_stack, ((0, 0), (0, pad)))

    return mfcc_vec.astype(np.float32), temporal.astype(np.float32)


def _compute_pitch_amplitude_features(
    y: np.ndarray, sr: int, cfg: Config, hop: int, win: int
) -> np.ndarray:
    """Compute pitch + amplitude features (6-dim vector).

    Features:
    0. Peak frequency (Hz)
    1. Minimum frequency (Hz) — lowest bin > 10 dB above noise floor
    2. Frequency range = peak - minimum (Hz)
    3. Peak amplitude (max RMS)
    4. Attack time (ms) — onset to peak amplitude
    5. Decay time (ms) — peak amplitude to -10 dB below peak

    Parameters
    ----------
    y:
        Audio signal.
    sr:
        Sample rate.
    cfg:
        Pipeline configuration.
    hop:
        Hop length in samples.
    win:
        Window length in samples.
    """
    import librosa

    # Spectrogram
    D = np.abs(librosa.stft(y, n_fft=win, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=win)

    # Peak frequency
    mean_spectrum = D.mean(axis=1)
    peak_freq = float(freqs[np.argmax(mean_spectrum)])

    # Noise floor: median power of bottom 10% of bins
    power_db = librosa.amplitude_to_db(mean_spectrum, ref=np.max)
    noise_floor_db = float(np.median(power_db[:max(1, len(power_db) // 10)]))
    threshold_db = noise_floor_db + 10.0
    above_threshold = power_db > threshold_db
    if above_threshold.any():
        min_freq = float(freqs[np.argmax(above_threshold)])
    else:
        min_freq = float(freqs[0])

    freq_range = max(0.0, peak_freq - min_freq)

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=win)[0]
    peak_amplitude = float(rms.max()) if rms.size > 0 else 0.0
    peak_frame = int(np.argmax(rms)) if rms.size > 0 else 0

    # Attack time: onset frame to peak frame
    attack_ms = float(peak_frame * hop / sr * 1000.0)

    # Decay time: peak frame to first frame where rms < peak - 10 dB
    peak_db = 20 * np.log10(rms[peak_frame] + 1e-9)
    threshold_rms = 10 ** ((peak_db - 10.0) / 20)
    decay_frames = np.where(rms[peak_frame:] < threshold_rms)[0]
    if decay_frames.size > 0:
        decay_ms = float(decay_frames[0] * hop / sr * 1000.0)
    else:
        decay_ms = float((len(rms) - peak_frame) * hop / sr * 1000.0)

    # FM depth: mean absolute change in peak-frequency bin across spectrogram frames
    # (instantaneous frequency modulation per RR §2.9.2)
    if D.shape[1] > 1:
        peak_freq_per_frame = freqs[np.argmax(D, axis=0)]
        fm_depth = float(np.mean(np.abs(np.diff(peak_freq_per_frame))))
    else:
        fm_depth = 0.0

    return (
        np.array(
            [peak_freq, min_freq, freq_range, peak_amplitude, attack_ms, decay_ms],
            dtype=np.float32,
        ),
        fm_depth,
    )
