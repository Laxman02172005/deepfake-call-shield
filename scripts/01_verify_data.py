"""
scripts/01_verify_data.py
=========================
Read-only dataset verification and manifest generation script for ASVspoof 2019 LA.

Verifies:
1. Expected directory structure under data/raw/LA/.
2. Protocol formatting (5 columns, valid placeholders, attack IDs, and keys).
3. Official utterance and class counts (Train: 25,380; Dev: 24,844; Eval: 71,237).
4. Strict utterance-ID and speaker-ID disjointness across official splits.
5. Presence of corresponding .flac audio files.
6. Generates full split manifests and deterministic subsets under data/manifests/.

Exits with code 0 on success, code 1 if dataset is missing or validation fails.

Usage
-----
    python scripts/01_verify_data.py
    python scripts/01_verify_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.dataset import (
    build_manifest,
    check_no_leakage,
    create_stratified_subset,
    parse_protocol,
)
from src.utils import get_logger

logger = get_logger(__name__)

OFFICIAL_COUNTS = {
    "train": {"total": 25380, "bonafide": 2580, "spoof": 22800},
    "dev": {"total": 24844, "bonafide": 2548, "spoof": 22296},
    "eval": {"total": 71237, "bonafide": 7355, "spoof": 63882},
}


def verify_dataset(cfg) -> int:
    """Run comprehensive dataset integrity verification.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    logger.info("=" * 70)
    logger.info("Deepfake Call Shield — ASVspoof 2019 LA Data Verification")
    logger.info("=" * 70)

    # 1. Directory Structure Checks
    protocol_dir = Path(cfg.dataset.protocol_dir)
    if not protocol_dir.exists():
        logger.error(
            "Dataset protocols directory not found at: %s\n\n"
            "Action required:\n"
            "  1. Download 'LA.zip' (7.12 GB) from Edinburgh DataShare (Handle 10283/3336):\n"
            "     https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\n"
            "  2. Extract contents into data/raw/LA/ such that protocol and audio directories exist.",
            protocol_dir,
        )
        return 1

    tables = {}
    for split in ("train", "dev", "eval"):
        split_file = getattr(cfg.dataset.splits, split)
        p_path = protocol_dir / split_file
        if not p_path.exists():
            logger.error("Missing protocol file for '%s' split: %s", split, p_path)
            return 1

        audio_dir = Path(getattr(cfg.dataset.audio_dirs, split))
        if not audio_dir.exists():
            logger.error("Missing audio directory for '%s' split: %s", split, audio_dir)
            return 1

        try:
            logger.info("Parsing and validating %s protocol...", split)
            table = parse_protocol(p_path, split=split)
            tables[split] = table
        except Exception as exc:
            logger.error("Protocol validation failed for '%s': %s", split, exc)
            return 1

        # Check official counts
        expected = OFFICIAL_COUNTS[split]
        bonafide_count = sum(1 for r in table if r.label == 0)
        spoof_count = sum(1 for r in table if r.label == 1)
        total_count = len(table)

        logger.info(
            "  [%s] Utterances: %d (expected: %d) | Bonafide: %d | Spoof: %d",
            split.upper(), total_count, expected["total"], bonafide_count, spoof_count
        )

        if total_count != expected["total"]:
            logger.warning(
                "Count mismatch in '%s'! Expected %d utterances, got %d",
                split, expected["total"], total_count
            )
        if bonafide_count != expected["bonafide"] or spoof_count != expected["spoof"]:
            logger.warning(
                "Class count mismatch in '%s'! Expected %d bonafide / %d spoof, got %d / %d",
                split, expected["bonafide"], expected["spoof"], bonafide_count, spoof_count
            )

    # 2. Utterance and Speaker Disjointness (Hard Assertion)
    logger.info("Verifying utterance and speaker disjointness across splits...")
    try:
        leakage_summary = check_no_leakage(tables["train"], tables["dev"], tables["eval"])
        logger.info(
            "Disjointness verified: Train (%d utts, %d spk) | Dev (%d utts, %d spk) | Eval (%d utts, %d spk)",
            leakage_summary["train_utts"], leakage_summary["train_speakers"],
            leakage_summary["dev_utts"], leakage_summary["dev_speakers"],
            leakage_summary["eval_utts"], leakage_summary["eval_speakers"],
        )
    except AssertionError as leak_err:
        logger.error("DATA LEAKAGE ASSERTION FAILED:\n%s", leak_err)
        return 1

    # 3. Audio File Existence Verification
    logger.info("Verifying audio file presence on disk...")
    for split in ("train", "dev", "eval"):
        audio_dir = Path(getattr(cfg.dataset.audio_dirs, split))
        table = tables[split]
        missing_count = 0
        for r in table:
            flac_file = audio_dir / f"{r.utt_id}.flac"
            if not flac_file.exists():
                missing_count += 1
                if missing_count <= 3:
                    logger.error("Missing audio file: %s", flac_file)

        if missing_count > 0:
            logger.error(
                "Split '%s' has %d missing audio files out of %d total!",
                split, missing_count, len(table)
            )
            return 1
        logger.info("  [%s] All %d audio files verified present on disk.", split.upper(), len(table))

    # 4. Generate Official Manifests
    logger.info("Generating split manifests in %s...", cfg.paths.manifests)
    manifest_tables = {}
    for split in ("train", "dev", "eval"):
        manifest_tables[split] = build_manifest(cfg, split=split, force=True)

    # 5. Generate Deterministic Stratified Subsets (Train & Dev only)
    logger.info("Generating deterministic stratified subsets for pilot experimentation...")
    train_subset = create_stratified_subset(manifest_tables["train"], split="train", n_samples=2000, seed=cfg.seed)
    train_sub_path = Path(cfg.paths.manifests) / "train_subset.csv"
    train_subset.to_csv(train_sub_path)
    logger.info("  Saved train subset (2000 utts) to %s", train_sub_path)

    dev_subset = create_stratified_subset(manifest_tables["dev"], split="dev", n_samples=1000, seed=cfg.seed)
    dev_sub_path = Path(cfg.paths.manifests) / "dev_subset.csv"
    dev_subset.to_csv(dev_sub_path)
    logger.info("  Saved dev subset (1000 utts) to %s", dev_sub_path)

    logger.info("=" * 70)
    logger.info("Dataset verification and manifest generation completed successfully!")
    logger.info("=" * 70)
    return 0


def main(config_path: Path | None = None) -> int:
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        logger.error("Failed to load project configuration: %s", exc)
        return 1
    return verify_dataset(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ASVspoof 2019 LA dataset and build manifests.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: configs/base.yaml)",
    )
    args = parser.parse_args()
    sys.exit(main(args.config))
