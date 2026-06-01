"""Top-level entry point: runs all Syrinx pipeline stages in order."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from syrinx.config import load_config
from syrinx.utils import PipelineRunLog, configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv).
    """
    parser = argparse.ArgumentParser(
        description="Syrinx acoustic phylogenetics pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument(
        "--dataset",
        choices=["genus", "within_species", "both"],
        default="both",
        help="Which dataset to analyse",
    )
    parser.add_argument(
        "--cap",
        type=int,
        choices=[20, 50, 100],
        default=None,
        help="Recording cap per species (overrides config)",
    )
    parser.add_argument("--skip-acquire", action="store_true", help="Skip download stage")
    parser.add_argument("--skip-segment", action="store_true", help="Skip segmentation stage")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only; do not execute pipeline",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline.

    Parameters
    ----------
    argv:
        Argument list (defaults to sys.argv).

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Step 1: Load and validate config (must precede logging so we can use cfg.data_path)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        configure_logging(level=args.log_level)
        logging.getLogger(__name__).error("Config load failed: %s", exc)
        return 1

    log_path = cfg.data_path / "manifests" / f"pipeline_run_{timestamp}.log"
    configure_logging(level=args.log_level, log_path=log_path)

    logger = logging.getLogger(__name__)
    logger.info("Syrinx pipeline starting — dataset=%s", args.dataset)

    if args.cap is not None:
        cfg.recording_cap = args.cap
        logger.info("Recording cap overridden to %d", cfg.recording_cap)

    if args.dry_run:
        logger.info("--dry-run: config validated successfully. Exiting.")
        print(f"Config loaded OK: {args.config} (hash={cfg.hash})")
        return 0

    run_log = PipelineRunLog(
        path=cfg.data_path / "manifests" / f"pipeline_run_{timestamp}.json",
        config_hash=cfg.hash,
        random_seed=cfg.random_seed,
    )

    # Step 2: Run end-to-end simulated test (abort on failure)
    logger.info("Step 2: Running preregistered end-to-end test…")
    import pytest
    test_result = pytest.main(
        ["-v", "tests/test_pipeline_simulated.py", "--tb=short"],
        plugins=[],
    )
    if test_result != 0:
        logger.error("End-to-end test suite failed. Aborting pipeline.")
        return 1
    logger.info("End-to-end tests passed.")

    datasets = [args.dataset] if args.dataset != "both" else ["genus", "within_species"]

    for dataset in datasets:
        logger.info("=== Processing dataset: %s ===", dataset)

        recordings: list[dict] = []
        syllables: list[dict] = []

        # Step 3: Acquire
        if not args.skip_acquire:
            logger.info("Step 3: Acquiring recordings from Xeno-canto…")
            from syrinx.acquire import acquire
            try:
                recordings = acquire(cfg, dataset=dataset)
                run_log.record_stage(f"stage1_acquire_{dataset}", {"n_downloaded": len(recordings)})
            except Exception as exc:
                logger.error("Acquisition failed: %s", exc)
                return 1
        else:
            logger.info("--skip-acquire: loading existing manifests…")
            from syrinx.utils import load_manifest
            manifest = load_manifest(cfg.data_path / "manifests" / "download_manifest.json")
            recordings = manifest.get("records", [])

        # Step 4: Segment + MAO validation
        if not args.skip_segment:
            logger.info("Step 4: Segmenting recordings…")
            from syrinx.segment import run_mao_validation, segment_all
            mao_result = run_mao_validation(cfg)
            run_log.record_stage(f"stage2_mao_{dataset}", mao_result)
            try:
                syllables = segment_all(cfg, recordings)
                run_log.record_stage(f"stage2_segment_{dataset}", {"n_syllables": len(syllables)})
            except Exception as exc:
                logger.error("Segmentation failed: %s", exc)
                return 1
        else:
            logger.info("--skip-segment: loading existing syllables from features…")
            import pickle
            feat_dir = cfg.data_path / "features"
            syllables = []
            for pkl in sorted(feat_dir.glob("*_features.pkl")):
                with pkl.open("rb") as fh:
                    data = pickle.load(fh)
                syllables.extend(data.get("syllables", []))

        if len(syllables) == 0:
            logger.error("No syllables produced; cannot continue.")
            return 1

        # Step 5: Extract features
        logger.info("Step 5: Extracting acoustic features…")
        from syrinx.features import extract_features, save_features
        try:
            syllables = extract_features(cfg, syllables)
        except Exception as exc:
            logger.error("Feature extraction failed: %s", exc)
            return 1

        # Save per-species
        species_set = sorted(set(s.get("species", "") for s in syllables))
        for sp in species_set:
            sp_syllables = [s for s in syllables if s.get("species") == sp]
            save_features(sp_syllables, cfg, sp, len(syllables))
        run_log.record_stage(f"stage3_features_{dataset}", {
            "n_syllables": len(syllables),
            "n_species": len(species_set),
        })

        # Step 5b: Prepare zebra finch reference (§2.5 — must precede vocabulary)
        logger.info("Step 5b: Preparing zebra finch reference data…")
        from syrinx.acquire import prepare_zebrafinch_reference
        try:
            prepare_zebrafinch_reference(cfg)
        except Exception as exc:
            logger.error("Zebra finch reference preparation failed: %s", exc)
            return 1

        # Step 6: Vocabulary
        logger.info("Step 6: Building and validating vocabulary…")
        from syrinx.vocabulary import VocabularyValidationError, build_vocabulary
        try:
            vocabulary = build_vocabulary(cfg, syllables, run_log=run_log)
        except VocabularyValidationError as exc:
            logger.error("Vocabulary validation failed: %s", exc)
            logger.error("Diagnostic: %s", exc.diagnostic)
            return 1

        # Step 7: Substitution matrix
        logger.info("Step 7: Building substitution matrix…")
        from syrinx.substitution import build_substitution_matrix
        substitution = build_substitution_matrix(cfg, vocabulary, syllables)
        run_log.record_stage(f"stage5_substitution_{dataset}", {
            "n_clusters": substitution["n_clusters"],
            "gap_open": substitution["gap_open"],
            "gap_extend": substitution["gap_extend"],
        })

        # Step 8: Align + null model
        logger.info("Step 8: Aligning song strings…")
        from syrinx.align import PipelineGatingError, align_all
        try:
            alignment_result = align_all(
                cfg, syllables, vocabulary, substitution,
                run_log=run_log, dataset=dataset,
            )
        except PipelineGatingError as exc:
            logger.error("Alignment gating failed: %s", exc)
            logger.error("Diagnostic: %s", exc.diagnostic)
            return 1

        # Step 9: Phylogenetic trees
        logger.info("Step 9: Reconstructing phylogenetic trees…")
        from syrinx.phylo import reconstruct_trees
        phylo_result = reconstruct_trees(cfg, alignment_result, run_log=run_log)
        run_log.record_stage(f"stage7_phylo_{dataset}", {
            "n_species": len(alignment_result["species_names"]),
            "rf_distance": phylo_result.get("rf_distance"),
        })

        # Step 10: Power simulations
        logger.info("Step 10: Running power simulations…")
        from syrinx.power import run_power_simulations
        n_cells = (
            len(alignment_result["species_names"])
            if dataset == "within_species"
            else 10
        )
        power_result = run_power_simulations(cfg, n_uk_cells=n_cells, run_log=run_log)

        # Step 11: Descriptive statistics
        logger.info("Step 11: Computing descriptive statistics…")
        from syrinx.descriptive import compute_descriptives
        descriptive_result = compute_descriptives(cfg, syllables, vocabulary, run_log=run_log)

        # Step 12: Species profiles
        logger.info("Step 12: Building species profiles…")
        from syrinx.species_profile import build_species_profiles
        build_species_profiles(cfg, descriptive_result["species_stats"], alignment_result)

        # Step 13: Inferential tests (gated)
        logger.info("Step 13: Running inferential tests…")
        from syrinx.inference import PipelineGatingError as InferencePipelineGatingError
        from syrinx.inference import run_inference
        species_metadata = [
            {"species": s.get("species"), "lat": s.get("lat"), "lon": s.get("lon")}
            for s in recordings
        ]
        try:
            inference_result = run_inference(
                cfg,
                alignment_result=alignment_result,
                vocabulary=vocabulary,
                species_metadata=species_metadata,
                descriptive_result=descriptive_result,
                molecular_tree=phylo_result["reference_trees"].get("alstrom2018"),
                run_log=run_log,
                dataset=dataset,
            )
        except InferencePipelineGatingError as exc:
            logger.error("Inference gating error: %s", exc)
            logger.error("Diagnostic: %s", exc.diagnostic)
            return 1

        logger.info("Dataset %s complete.", dataset)

    run_log.finalise()
    logger.info("All pipeline stages complete. Results in %s/", cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
