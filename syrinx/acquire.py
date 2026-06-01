"""Stage 1 — Xeno-canto API v2 download with deduplication."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

from .config import Config
from .utils import append_manifest, load_manifest, save_manifest

logger = logging.getLogger(__name__)

_ACOUSTID_AVAILABLE = False
try:
    import acoustid
    _ACOUSTID_AVAILABLE = True
except ImportError:
    logger.warning("python-acoustid not installed; chromaprint deduplication disabled")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_zebrafinch_reference(cfg: Config) -> None:
    """Download and feature-extract zebra finch reference recordings (§2.5).

    Produces ``data/reference/zebrafinch_features.npy``.  If the file already
    exists it is left untouched.  The companion label file
    ``data/reference/zebrafinch_labels.npy`` must be provided separately:
    it should contain published syllable-type labels from Tchernichovski et al.
    (2000) aligned one-to-one with the rows of the features array.

    Parameters
    ----------
    cfg:
        Pipeline configuration.

    Raises
    ------
    RuntimeError
        If no recordings can be downloaded or no syllables are segmented.
    """
    ref_dir = cfg.data_path / "reference"
    features_path = ref_dir / "zebrafinch_features.npy"

    if features_path.exists():
        logger.info("Zebra finch reference features already present: %s", features_path)
        return

    ref_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = ref_dir / "raw"
    wav_dir = ref_dir / "wav"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading zebra finch reference recordings from Xeno-canto…")
    recordings = _search_xc(cfg, "gen:Taeniopygia sp:guttata type:song q:A")
    recordings = [r for r in recordings if _meets_duration(r, cfg.min_duration_s)]
    rng = random.Random(cfg.random_seed)
    rng.shuffle(recordings)
    recordings = recordings[: cfg.zebrafinch_xc_cap]

    seen_fps: set[str] = set()
    downloaded: list[dict[str, Any]] = []
    for rec in recordings:
        result = _download_recording(
            rec, raw_dir, wav_dir, cfg, prior_fingerprints=seen_fps
        )
        if result:
            result["species"] = "Taeniopygia guttata"
            downloaded.append(result)
            if result.get("fingerprint"):
                seen_fps.add(result["fingerprint"])

    if not downloaded:
        raise RuntimeError(
            "Could not download any zebra finch recordings from Xeno-canto. "
            "Check network connectivity and the XC API."
        )
    logger.info("Downloaded %d zebra finch reference recordings", len(downloaded))

    from .segment import segment_all

    zf_syllables = segment_all(cfg, downloaded)
    if not zf_syllables:
        raise RuntimeError(
            "No syllables were segmented from the zebra finch reference recordings."
        )

    from .features import extract_features

    zf_syllables = extract_features(cfg, zf_syllables)
    if not zf_syllables:
        raise RuntimeError(
            "Feature extraction failed for all zebra finch reference syllables."
        )

    X_zf = np.vstack([s["features"] for s in zf_syllables]).astype(np.float64)
    np.save(features_path, X_zf)
    logger.info(
        "Saved zebra finch reference features: %s (%d syllables, 36 dims)",
        features_path, len(X_zf),
    )

    labels_path = ref_dir / "zebrafinch_labels.npy"
    if not labels_path.exists():
        logger.warning(
            "zebrafinch_features.npy saved (%d rows). "
            "You must also provide %s containing published syllable-type labels "
            "from Tchernichovski et al. (2000), one integer label per row.",
            len(X_zf), labels_path,
        )


def acquire(cfg: Config, dataset: str = "genus") -> list[dict[str, Any]]:
    """Download Xeno-canto recordings for the specified dataset.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    dataset:
        One of ``'genus'`` (Phylloscopus spp.) or ``'within_species'``
        (UK Willow Warbler).

    Returns
    -------
    list[dict]
        Metadata records for all successfully downloaded recordings.
    """
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    data_dir = cfg.data_path
    raw_dir = data_dir / "raw"
    grade_b_dir = raw_dir / "grade_b"
    wav_dir = data_dir / "wav"
    manifest_dir = data_dir / "manifests"
    for d in (raw_dir, grade_b_dir, wav_dir, manifest_dir):
        d.mkdir(parents=True, exist_ok=True)

    exclusion_path = manifest_dir / "exclusion_manifest.json"
    download_manifest_path = manifest_dir / "download_manifest.json"

    exclusion_set = _load_exclusion_set(exclusion_path)
    prior_manifest = load_manifest(download_manifest_path)
    prior_ids: set[str] = {r["xc_id"] for r in prior_manifest.get("records", [])}
    prior_fingerprints: set[str] = {
        r["fingerprint"] for r in prior_manifest.get("records", [])
        if r.get("fingerprint")
    }

    if dataset == "genus":
        query = cfg.xc_query_genus
        recordings = _search_xc(cfg, query)
        recordings = _filter_genus(recordings, cfg)
    elif dataset == "within_species":
        query = cfg.xc_query_within_species
        recordings = _search_xc(cfg, query)
        recordings = _filter_within_species(recordings, cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    logger.info("Found %d candidate recordings from Xeno-canto", len(recordings))

    # Pre-fetch all grade B recordings in a single query to avoid one API call
    # per species inside the loop below (which would hit rate limits at scale).
    grade_b_query = query.replace("q:A", "q:B")
    logger.info("Pre-fetching grade B recordings…")
    all_grade_b = _search_xc(cfg, grade_b_query)
    grade_b_by_species: dict[str, list[dict[str, Any]]] = {}
    for r in all_grade_b:
        grade_b_by_species.setdefault(r.get("sp", ""), []).append(r)

    # Group by species, apply recording cap per species
    by_species: dict[str, list[dict[str, Any]]] = {}
    for rec in recordings:
        sp = rec.get("sp", "unknown")
        by_species.setdefault(sp, []).append(rec)

    downloaded: list[dict[str, Any]] = []
    for species, recs in sorted(by_species.items()):
        # Count already-downloaded recordings for this species
        already = sum(
            1 for r in prior_manifest.get("records", [])
            if r.get("species") == species and not r.get("grade_b")
        )
        needed = cfg.recording_cap - already
        if needed <= 0:
            logger.info("Species %s already at cap (%d); skipping", species, cfg.recording_cap)
            continue

        # Shuffle for reproducibility then take up to needed
        rng = random.Random(cfg.random_seed + hash(species) % (2 ** 31))
        rng.shuffle(recs)
        candidates = [r for r in recs if r["id"] not in exclusion_set and r["id"] not in prior_ids]
        candidates = candidates[:needed + 20]  # over-fetch to account for failures

        count = 0
        for rec in candidates:
            if count >= needed:
                break
            result = _download_recording(
                rec, raw_dir, wav_dir, cfg,
                prior_fingerprints=prior_fingerprints,
                grade_b=False,
            )
            if result:
                downloaded.append(result)
                prior_ids.add(result["xc_id"])
                if result.get("fingerprint"):
                    prior_fingerprints.add(result["fingerprint"])
                count += 1

        # Grade B sensitivity analysis (uses pre-fetched results — no extra API call)
        grade_b_recs = [
            r for r in grade_b_by_species.get(species, [])
            if r["id"] not in exclusion_set and r["id"] not in prior_ids
        ]
        for rec in grade_b_recs[:5]:
            result = _download_recording(
                rec, grade_b_dir, wav_dir / "grade_b", cfg,
                prior_fingerprints=prior_fingerprints,
                grade_b=True,
            )
            if result:
                downloaded.append(result)

    # Persist download manifest
    manifest_data = load_manifest(download_manifest_path)
    if "records" not in manifest_data:
        manifest_data["records"] = []
    manifest_data["records"].extend(downloaded)
    save_manifest(manifest_data, download_manifest_path)

    logger.info("Acquired %d new recordings", len([d for d in downloaded if not d.get("grade_b")]))
    return downloaded


# ---------------------------------------------------------------------------
# Xeno-canto API
# ---------------------------------------------------------------------------

def _search_xc(cfg: Config, query: str) -> list[dict[str, Any]]:
    """Paginate through all Xeno-canto results for a query.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    query:
        Xeno-canto search string.
    """
    recordings: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = _xc_get(cfg, {"query": query, "page": page})
        if resp is None:
            break
        recordings.extend(resp.get("recordings", []))
        num_pages = int(resp.get("numPages", 1))
        logger.debug("XC page %d/%d (%d recordings so far)", page, num_pages, len(recordings))
        if page >= num_pages:
            break
        page += 1
    return recordings


def _xc_get(cfg: Config, params: dict[str, Any]) -> dict[str, Any] | None:
    """Make a single Xeno-canto API request with exponential backoff.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    params:
        Request parameters.
    """
    backoff = cfg.api_backoff_start_s
    for attempt in range(8):
        try:
            resp = requests.get(cfg.xeno_canto_base_url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                jitter = random.uniform(0, backoff * 0.3)
                sleep_for = min(backoff + jitter, cfg.api_backoff_max_s)
                logger.warning("Rate limited; sleeping %.1f s (attempt %d)", sleep_for, attempt)
                time.sleep(sleep_for)
                backoff = min(backoff * 2, cfg.api_backoff_max_s)
            else:
                logger.error("XC API returned %d for params %s", resp.status_code, params)
                return None
        except requests.RequestException as exc:
            jitter = random.uniform(0, backoff * 0.3)
            sleep_for = min(backoff + jitter, cfg.api_backoff_max_s)
            logger.warning("Request error: %s; retrying in %.1f s", exc, sleep_for)
            time.sleep(sleep_for)
            backoff = min(backoff * 2, cfg.api_backoff_max_s)
    logger.error("Failed to fetch XC page after 8 attempts")
    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_genus(
    recordings: list[dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    """Filter genus-scale dataset: quality A, no playback, min duration.

    Parameters
    ----------
    recordings:
        Raw recording metadata from Xeno-canto.
    cfg:
        Pipeline configuration.
    """
    kept = []
    for rec in recordings:
        if _has_playback(rec):
            continue
        if not _meets_duration(rec, cfg.min_duration_s):
            continue
        if not _in_breeding_window(rec, cfg):
            continue
        kept.append(rec)
    return kept


def _filter_within_species(
    recordings: list[dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    """Filter within-species UK dataset: bounding box, quality A, no playback.

    Parameters
    ----------
    recordings:
        Raw recording metadata from Xeno-canto.
    cfg:
        Pipeline configuration.
    """
    kept = []
    for rec in recordings:
        try:
            lat = float(rec.get("lat", "nan"))
            lon = float(rec.get("lng", "nan"))
        except (ValueError, TypeError):
            continue
        if not (cfg.xc_within_species_lat_min <= lat <= cfg.xc_within_species_lat_max):
            continue
        if not (cfg.xc_within_species_lon_min <= lon <= cfg.xc_within_species_lon_max):
            continue
        if _has_playback(rec):
            continue
        if not _meets_duration(rec, cfg.min_duration_s):
            continue
        kept.append(rec)
    return kept


def _has_playback(rec: dict[str, Any]) -> bool:
    remarks = str(rec.get("rmk", "")).lower()
    return "playback" in remarks or "lure" in remarks


def _meets_duration(rec: dict[str, Any], min_s: float) -> bool:
    try:
        length = rec.get("length", "")
        if ":" in str(length):
            parts = str(length).split(":")
            seconds = int(parts[-1]) + int(parts[-2]) * 60 if len(parts) >= 2 else float(length)
        else:
            seconds = float(length)
        return seconds >= min_s
    except (ValueError, TypeError):
        return False


def _in_breeding_window(rec: dict[str, Any], cfg: Config) -> bool:
    """Check whether a recording falls within the expected breeding season.

    Uses the union of the Palearctic (Apr–Jul) and Asian (May–Aug) breeding
    windows so that no valid recording is excluded regardless of species origin.
    Per-species window selection would be more precise but requires a maintained
    Asian Phylloscopus species list; the union is the conservative fallback
    documented in the preregistered analysis plan.

    Parameters
    ----------
    rec:
        Xeno-canto recording metadata.
    cfg:
        Pipeline configuration.
    """
    date_str = str(rec.get("date", ""))
    try:
        month = int(date_str.split("-")[1]) if "-" in date_str else 0
    except (IndexError, ValueError):
        return True  # allow through if date is unavailable
    if month == 0:
        return True
    lo_pal, hi_pal = cfg.breeding_window_palearctic
    lo_asi, hi_asi = cfg.breeding_window_asian
    lo = min(lo_pal, lo_asi)
    hi = max(hi_pal, hi_asi)
    return lo <= month <= hi


# ---------------------------------------------------------------------------
# Download and conversion
# ---------------------------------------------------------------------------

def _download_recording(
    rec: dict[str, Any],
    raw_dir: Path,
    wav_dir: Path,
    cfg: Config,
    *,
    prior_fingerprints: set[str],
    grade_b: bool = False,
) -> dict[str, Any] | None:
    """Download one recording, convert to wav, apply all deduplication safeguards.

    Parameters
    ----------
    rec:
        Xeno-canto recording metadata dict.
    raw_dir:
        Directory to save the raw .mp3 file.
    wav_dir:
        Directory to save the converted .wav file.
    cfg:
        Pipeline configuration.
    prior_fingerprints:
        Set of already-seen chromaprint fingerprints (modified in place).
    grade_b:
        If True, tag the record as grade B.

    Returns
    -------
    dict or None
        Metadata record if successful, else None.
    """
    xc_id = str(rec.get("id", ""))
    file_url = rec.get("file", "")
    if not file_url:
        file_url = f"https://xeno-canto.org/{xc_id}/download"

    sp_name = (rec.get("gen", "") + "_" + rec.get("sp", "")).strip("_").replace(" ", "_")
    mp3_path = raw_dir / f"XC{xc_id}.mp3"
    wav_path = wav_dir / f"XC{xc_id}.wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Download mp3
    if not mp3_path.exists():
        try:
            r = requests.get(file_url, timeout=60, stream=True)
            r.raise_for_status()
            with mp3_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    fh.write(chunk)
        except Exception as exc:
            logger.warning("Failed to download XC%s: %s", xc_id, exc)
            return None

    # Convert to 16 kHz mono wav
    if not wav_path.exists():
        try:
            import librosa
            y, _ = librosa.load(str(mp3_path), sr=16000, mono=True)
            sf.write(str(wav_path), y, 16000)
        except Exception as exc:
            logger.warning("Failed to convert XC%s to wav: %s", xc_id, exc)
            mp3_path.unlink(missing_ok=True)
            return None

    # Chromaprint deduplication (safeguard 3)
    fingerprint = _compute_fingerprint(wav_path)
    if fingerprint and fingerprint in prior_fingerprints:
        logger.info("Duplicate fingerprint for XC%s; discarding", xc_id)
        wav_path.unlink(missing_ok=True)
        mp3_path.unlink(missing_ok=True)
        return None

    licence = _parse_licence(rec.get("lic", ""))
    record: dict[str, Any] = {
        "xc_id": xc_id,
        "species": f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
        "subspecies": rec.get("ssp", ""),
        "country": rec.get("cnt", ""),
        "lat": _safe_float(rec.get("lat")),
        "lon": _safe_float(rec.get("lng")),
        "date": rec.get("date", ""),
        "recordist_id": rec.get("rec", ""),
        "playback_flag": _has_playback(rec),
        "licence_type": licence,
        "mp3_path": str(mp3_path),
        "wav_path": str(wav_path),
        "fingerprint": fingerprint or "",
        "grade_b": grade_b,
        "dedup_safeguards": {
            "exclusion_manifest": True,
            "cap_level": True,
            "chromaprint": bool(fingerprint),
        },
    }
    logger.debug("Downloaded XC%s (%s)", xc_id, record["species"])
    return record


def _compute_fingerprint(wav_path: Path) -> str | None:
    """Compute a chromaprint acoustic fingerprint for deduplication.

    Parameters
    ----------
    wav_path:
        Path to the 16 kHz mono wav file.
    """
    if not _ACOUSTID_AVAILABLE:
        return None
    try:
        duration, fp = acoustid.fingerprint_file(str(wav_path))
        return fp.decode() if isinstance(fp, bytes) else fp
    except Exception as exc:
        logger.debug("Fingerprint failed for %s: %s", wav_path, exc)
        return None


def _parse_licence(lic_str: str) -> str:
    """Normalise a Xeno-canto licence string to a canonical label.

    Parameters
    ----------
    lic_str:
        Raw licence string from the API.
    """
    lic = lic_str.lower()
    if "nc-sa" in lic:
        return "CC BY-NC-SA"
    if "nc-nd" in lic:
        return "CC BY-NC-ND"
    if "nc" in lic:
        return "CC BY-NC"
    if "sa" in lic:
        return "CC BY-SA"
    if "nd" in lic:
        return "CC BY-ND"
    if "by" in lic:
        return "CC BY"
    return "unknown"


def _load_exclusion_set(path: Path) -> set[str]:
    """Load Xeno-canto IDs from the exclusion manifest.

    Parameters
    ----------
    path:
        Path to the exclusion manifest JSON.
    """
    if not path.exists():
        return set()
    data = load_manifest(path)
    return set(data.get("excluded_ids", []))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
