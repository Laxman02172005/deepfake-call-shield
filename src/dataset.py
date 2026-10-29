"""
src/dataset.py
==============
Protocol parsing, manifest generation, split-safe data access, and
deterministic subset sampling for the ASVspoof 2019 Logical Access (LA) dataset.

This module is the single source of truth for which utterances belong to which
split. It enforces the train / dev / eval separation and speaker disjointness
required by the experiment design.

Official 5-Column Space-Delimited Format
-----------------------------------------
Each line in official ASVspoof 2019 LA countermeasure protocol files has:
    [SPEAKER_ID] [AUDIO_FILE_NAME] [-] [SYSTEM_ID] [KEY]

- Column 0: speaker_id (e.g. 'LA_0079', 'LA_0039')
- Column 1: utt_id (e.g. 'LA_T_1138215', 'LA_D_1000265', 'LA_E_2834763')
- Column 2: placeholder (always '-' in LA scenario)
- Column 3: system_id ('-' for bonafide; 'A01'-'A06' in train/dev; 'A07'-'A19' in eval)
- Column 4: key ('bonafide' or 'spoof')

Public API
----------
parse_protocol(protocol_path, split)
    Parse an official protocol file, strictly validating format and attack IDs.
build_manifest(cfg, split, audio_dir, force)
    Generate (or load) a manifest CSV with relative audio paths.
load_manifest(cfg, split)
    Load a previously generated manifest CSV.
check_no_leakage(train_df, dev_df, eval_df)
    Enforce utterance-ID and speaker-ID disjointness across official splits.
check_speaker_disjointness(train_df, dev_df, eval_df)
    Verify that speaker sets across partitions are strictly disjoint.
create_stratified_subset(manifest_df, split, n_samples, seed, for_eval)
    Deterministically sample a subset preserving natural class and attack ratios.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Sequence, Set

from src.utils import get_logger

logger = get_logger(__name__)

SplitName = Literal["train", "dev", "eval"]

# Attack ID groupings for seen/unseen analysis
ATTACK_GROUPS: dict[str, list[str]] = {
    "seen": [f"A{i:02d}" for i in range(1, 7)],     # A01–A06 (train & dev)
    "unseen": [f"A{i:02d}" for i in range(7, 20)],  # A07–A19 (eval)
}

# Permitted spoof attack IDs per partition
ALLOWED_ATTACKS_BY_SPLIT: dict[str, set[str]] = {
    "train": {f"A{i:02d}" for i in range(1, 7)},
    "dev": {f"A{i:02d}" for i in range(1, 7)},
    "eval": {f"A{i:02d}" for i in range(7, 20)},
}


# ---------------------------------------------------------------------------
# Lightweight Table Structure (Zero external dependency requirement)
# ---------------------------------------------------------------------------

class ManifestRecord:
    """Represents a single utterance metadata row."""

    __slots__ = ("speaker_id", "utt_id", "attack_id", "key", "label", "audio_path")

    def __init__(
        self,
        speaker_id: str,
        utt_id: str,
        attack_id: str,
        key: str,
        label: int,
        audio_path: str = "",
    ):
        self.speaker_id = speaker_id
        self.utt_id = utt_id
        self.attack_id = attack_id
        self.key = key
        self.label = label
        self.audio_path = audio_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "utt_id": self.utt_id,
            "attack_id": self.attack_id,
            "key": self.key,
            "label": self.label,
            "audio_path": self.audio_path,
        }

    def __repr__(self) -> str:
        return f"ManifestRecord({self.to_dict()})"


class ManifestTable:
    """A clean, lightweight tabular dataset representation.

    Provides column indexing, row iteration, length reporting, and CSV I/O
    without requiring external libraries like pandas.
    """

    COLUMNS = ["speaker_id", "utt_id", "attack_id", "key", "label", "audio_path"]

    def __init__(self, records: Sequence[ManifestRecord]):
        self._records: list[ManifestRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self._records)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return self._records[item]
        if isinstance(item, str):
            if item not in self.COLUMNS:
                raise KeyError(f"Column '{item}' not in {self.COLUMNS}")
            return [getattr(r, item) for r in self._records]
        if isinstance(item, slice):
            return ManifestTable(self._records[item])
        raise TypeError(f"Invalid index type: {type(item)}")

    @property
    def records(self) -> list[ManifestRecord]:
        return self._records

    @property
    def columns(self) -> list[str]:
        return list(self.COLUMNS)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def to_csv(self, path: Path | str) -> None:
        """Write table contents to CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(r.to_dict())

    @classmethod
    def read_csv(cls, path: Path | str) -> ManifestTable:
        """Load a table from CSV."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        records: list[ManifestRecord] = []
        with open(p, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(
                    ManifestRecord(
                        speaker_id=row["speaker_id"],
                        utt_id=row["utt_id"],
                        attack_id=row["attack_id"],
                        key=row["key"],
                        label=int(row["label"]),
                        audio_path=row.get("audio_path", ""),
                    )
                )
        return cls(records)

    def __repr__(self) -> str:
        return f"ManifestTable({len(self)} rows)"


# ---------------------------------------------------------------------------
# Protocol Parsing
# ---------------------------------------------------------------------------

def parse_protocol(
    protocol_path: Path | str,
    split: SplitName = "train",
) -> ManifestTable:
    """Parse an official ASVspoof 2019 LA protocol file.

    Parameters
    ----------
    protocol_path:
        Path to official protocol .txt file.
    split:
        One of 'train', 'dev', 'eval'. Enforces partition-specific attack ID rules.

    Returns
    -------
    ManifestTable
        Parsed records with columns:
        [speaker_id, utt_id, attack_id, key, label, audio_path=""]

    Raises
    ------
    FileNotFoundError
        If protocol_path does not exist.
    ValueError
        If any row violates the 5-column structure, has invalid tokens,
        or contains attack IDs outside the permitted partition set.
    """
    path = Path(protocol_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    allowed_attacks = ALLOWED_ATTACKS_BY_SPLIT.get(split)
    if allowed_attacks is None:
        raise ValueError(f"Unknown split '{split}'; expected 'train', 'dev', or 'eval'")

    records: list[ManifestRecord] = []
    seen_utts: set[str] = set()

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError(
                    f"Line {line_no} in {path.name}: Expected 5 tokens, got {len(tokens)}: '{line}'"
                )

            speaker_id, utt_id, placeholder, system_id, key_str = tokens
            key = key_str.lower()

            # Column 2 must be placeholder '-'
            if placeholder != "-":
                raise ValueError(
                    f"Line {line_no} in {path.name}: Expected '-' in column 2, got '{placeholder}'"
                )

            # Column 4 must be 'bonafide' or 'spoof'
            if key not in ("bonafide", "spoof"):
                raise ValueError(
                    f"Line {line_no} in {path.name}: Invalid key '{key_str}'; expected 'bonafide' or 'spoof'"
                )

            # Attack ID and label logic
            if key == "bonafide":
                if system_id != "-":
                    raise ValueError(
                        f"Line {line_no} in {path.name}: Bonafide utterance has non-dash system_id: '{system_id}'"
                    )
                attack_id = "-"
                label = 0
            else:
                if system_id == "-":
                    raise ValueError(
                        f"Line {line_no} in {path.name}: Spoof utterance missing system_id (got '-')"
                    )
                if system_id not in allowed_attacks:
                    raise ValueError(
                        f"Line {line_no} in {path.name}: Attack ID '{system_id}' is not permitted in {split} partition "
                        f"(allowed: {sorted(allowed_attacks)})"
                    )
                attack_id = system_id
                label = 1

            # Utterance ID uniqueness check within the partition
            if utt_id in seen_utts:
                raise ValueError(
                    f"Line {line_no} in {path.name}: Duplicate utterance ID '{utt_id}' in split '{split}'"
                )
            seen_utts.add(utt_id)

            records.append(
                ManifestRecord(
                    speaker_id=speaker_id,
                    utt_id=utt_id,
                    attack_id=attack_id,
                    key=key,
                    label=label,
                    audio_path="",
                )
            )

    logger.info("Parsed %d utterances from %s [%s]", len(records), path.name, split)
    return ManifestTable(records)


# ---------------------------------------------------------------------------
# Manifest Building & Loading
# ---------------------------------------------------------------------------

def build_manifest(
    cfg: Any,
    split: SplitName,
    audio_dir: Path | None = None,
    force: bool = False,
) -> ManifestTable:
    """Build or load a manifest CSV for a given split with relative audio paths.

    Parameters
    ----------
    cfg:
        Loaded Config instance from src.config.load_config().
    split:
        'train', 'dev', or 'eval'.
    audio_dir:
        Optional directory containing .flac audio files. If None, resolves from cfg.
    force:
        If True, re-parses protocol even if manifest CSV exists.

    Returns
    -------
    ManifestTable
        Manifest table with valid relative audio_path strings.
    """
    manifest_path = Path(cfg.paths.manifests) / f"{split}.csv"
    if manifest_path.exists() and not force:
        logger.info("Loading existing manifest from %s", manifest_path)
        return load_manifest(cfg, split)

    protocol_filename = getattr(cfg.dataset.splits, split)
    protocol_path = Path(cfg.dataset.protocol_dir) / protocol_filename
    table = parse_protocol(protocol_path, split=split)

    if audio_dir is None:
        target_audio_dir = Path(getattr(cfg.dataset.audio_dirs, split))
    else:
        target_audio_dir = Path(audio_dir)

    # Compute relative path from project root (parent of configs/)
    project_root = Path(cfg.paths.manifests).parent.parent.resolve()
    try:
        rel_dir = target_audio_dir.resolve().relative_to(project_root).as_posix()
    except ValueError:
        # If outside project root or already relative, keep as clean POSIX path
        rel_dir = target_audio_dir.as_posix().lstrip("./")

    # Attach relative audio paths
    for r in table.records:
        rel_audio_path = f"{rel_dir}/{r.utt_id}.flac"
        # Strict validation: no Windows drive letter or leading root slash
        if ":" in rel_audio_path or rel_audio_path.startswith("/"):
            raise ValueError(f"Generated audio_path is not relative: {rel_audio_path}")
        r.audio_path = rel_audio_path

    table.to_csv(manifest_path)
    logger.info("Saved %s manifest (%d utterances) to %s", split, len(table), manifest_path)
    return table


def load_manifest(cfg: Any, split: SplitName) -> ManifestTable:
    """Load a previously generated manifest CSV from disk."""
    manifest_path = Path(cfg.paths.manifests) / f"{split}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest for split '{split}' not found at {manifest_path}. "
            f"Run build_manifest(cfg, '{split}') first."
        )
    return ManifestTable.read_csv(manifest_path)


# ---------------------------------------------------------------------------
# Split Leakage & Speaker Disjointness Verification
# ---------------------------------------------------------------------------

def check_no_leakage(
    train_table: ManifestTable,
    dev_table: ManifestTable,
    eval_table: ManifestTable,
) -> dict[str, int]:
    """Assert strict utterance-ID and speaker-ID disjointness across official splits.

    Raises
    ------
    AssertionError
        If any utterance ID or speaker ID appears in more than one partition.

    Returns
    -------
    dict[str, int]
        Summary counts of unique utterances and speakers per split.
    """
    train_utts = set(train_table["utt_id"])
    dev_utts = set(dev_table["utt_id"])
    eval_utts = set(eval_table["utt_id"])

    # 1. Utterance Disjointness (Hard Assertion)
    train_dev_utt = train_utts & dev_utts
    train_eval_utt = train_utts & eval_utts
    dev_eval_utt = dev_utts & eval_utts

    if train_dev_utt or train_eval_utt or dev_eval_utt:
        err_msg = ["FATAL: Utterance ID leakage detected between splits!"]
        if train_dev_utt:
            err_msg.append(f"  • Train <-> Dev overlap: {len(train_dev_utt)} utts (e.g. {sorted(train_dev_utt)[:3]})")
        if train_eval_utt:
            err_msg.append(f"  • Train <-> Eval overlap: {len(train_eval_utt)} utts (e.g. {sorted(train_eval_utt)[:3]})")
        if dev_eval_utt:
            err_msg.append(f"  • Dev <-> Eval overlap: {len(dev_eval_utt)} utts (e.g. {sorted(dev_eval_utt)[:3]})")
        raise AssertionError("\n".join(err_msg))

    # 2. Speaker Disjointness (Hard Assertion)
    speaker_info = check_speaker_disjointness(train_table, dev_table, eval_table)

    return {
        "train_utts": len(train_utts),
        "dev_utts": len(dev_utts),
        "eval_utts": len(eval_utts),
        **speaker_info,
    }


def check_speaker_disjointness(
    train_table: ManifestTable,
    dev_table: ManifestTable,
    eval_table: ManifestTable,
) -> dict[str, int]:
    """Verify that speaker IDs are strictly disjoint across partitions.

    Raises
    ------
    AssertionError
        If any speaker appears in more than one partition.
    """
    train_spk = set(train_table["speaker_id"])
    dev_spk = set(dev_table["speaker_id"])
    eval_spk = set(eval_table["speaker_id"])

    train_dev_spk = train_spk & dev_spk
    train_eval_spk = train_spk & eval_spk
    dev_eval_spk = dev_spk & eval_spk

    if train_dev_spk or train_eval_spk or dev_eval_spk:
        err_msg = ["FATAL: Speaker identity overlap detected across official splits!"]
        if train_dev_spk:
            err_msg.append(f"  • Train <-> Dev speaker overlap: {sorted(train_dev_spk)}")
        if train_eval_spk:
            err_msg.append(f"  • Train <-> Eval speaker overlap: {sorted(train_eval_spk)}")
        if dev_eval_spk:
            err_msg.append(f"  • Dev <-> Eval speaker overlap: {sorted(dev_eval_spk)}")
        raise AssertionError("\n".join(err_msg))

    logger.info(
        "Speaker disjointness verified: %d train, %d dev, %d eval speakers (all mutually exclusive)",
        len(train_spk), len(dev_spk), len(eval_spk)
    )
    return {
        "train_speakers": len(train_spk),
        "dev_speakers": len(dev_spk),
        "eval_speakers": len(eval_spk),
    }


# ---------------------------------------------------------------------------
# Deterministic Stratified Subset Sampling
# ---------------------------------------------------------------------------

def create_stratified_subset(
    manifest: ManifestTable,
    split: SplitName,
    n_samples: int,
    seed: int = 42,
    for_eval: bool = False,
) -> ManifestTable:
    """Deterministically sample a subset preserving natural class and attack ratios.

    Parameters
    ----------
    manifest:
        Input ManifestTable to sample from.
    split:
        Partition name ('train', 'dev', 'eval').
    n_samples:
        Total number of utterances desired.
    seed:
        Deterministic random seed.
    for_eval:
        Must be True if sampling from 'eval'. If False and split=='eval',
        raises ValueError to enforce evaluation isolation.

    Returns
    -------
    ManifestTable
        Deterministic subset of size n_samples.
    """
    if split == "eval" and not for_eval:
        raise ValueError(
            "Evaluation data is strictly isolated and cannot be sampled for training, "
            "layer selection, or threshold tuning. Set for_eval=True only in final reporting scripts."
        )

    if n_samples <= 0 or n_samples > len(manifest):
        raise ValueError(f"Requested {n_samples} samples, but manifest has {len(manifest)} rows.")

    rng = random.Random(seed)

    # Separate bonafide and spoof
    bonafide_records = [r for r in manifest if r.label == 0]
    spoof_records = [r for r in manifest if r.label == 1]

    # Preserve natural class representation
    total_len = len(manifest)
    n_bonafide = int(round(n_samples * (len(bonafide_records) / total_len)))
    # Ensure at least 1 bonafide if present in source
    if n_bonafide == 0 and bonafide_records:
        n_bonafide = 1
    n_spoof = n_samples - n_bonafide

    # Sample bonafide uniformly without replacement
    sampled_bonafide = rng.sample(bonafide_records, min(n_bonafide, len(bonafide_records)))

    # Stratify spoof by attack ID preserving the natural distribution
    spoof_by_attack: dict[str, list[ManifestRecord]] = {}
    for r in spoof_records:
        spoof_by_attack.setdefault(r.attack_id, []).append(r)

    attack_ids = sorted(spoof_by_attack.keys())
    sampled_spoof: list[ManifestRecord] = []

    if attack_ids and n_spoof > 0:
        base_per_attack = n_spoof // len(attack_ids)
        remainder = n_spoof % len(attack_ids)

        for idx, atk in enumerate(attack_ids):
            # Target per attack: uniform base plus deterministic remainder allocation
            quota = base_per_attack + (1 if idx < remainder else 0)
            pool = spoof_by_attack[atk]
            k = min(quota, len(pool))
            sampled_spoof.extend(rng.sample(pool, k))

    combined = sampled_bonafide + sampled_spoof
    # Deterministic final sort by utt_id
    combined.sort(key=lambda r: r.utt_id)

    logger.info(
        "Created %s subset: %d samples (%d bonafide, %d spoof across %d attacks, seed=%d)",
        split, len(combined), len(sampled_bonafide), len(sampled_spoof), len(attack_ids), seed
    )
    return ManifestTable(combined)
