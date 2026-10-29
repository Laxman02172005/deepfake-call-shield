"""
tests/test_dataset.py
=====================
Comprehensive unit tests for src/dataset.py.

All tests execute against synthetic in-memory fixtures and pytest's tmp_path.
No external dataset download is required.

Tests cover:
* Official 5-column protocol parsing for train, dev, and eval partitions.
* Rejection of malformed lines, bad tokens, invalid keys, and disallowed attack IDs.
* Hard assertion enforcement of utterance disjointness across splits.
* Hard assertion enforcement of speaker disjointness across splits.
* Relative audio path formatting in generated manifest CSV files.
* Deterministic subset sampling preserving natural class balance and attack distribution.
* Strict evaluation partition isolation guard.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.dataset import (
    ManifestRecord,
    ManifestTable,
    build_manifest,
    check_no_leakage,
    check_speaker_disjointness,
    create_stratified_subset,
    load_manifest,
    parse_protocol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def protocol_writer(tmp_path):
    """Factory fixture to write a protocol string to a temporary file."""
    def _create(filename: str, content: str) -> Path:
        p = tmp_path / filename
        p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        return p
    return _create


@pytest.fixture()
def mock_config(tmp_path):
    """Fixture providing a mock Config object conforming to src.config.Config."""
    protocols_dir = tmp_path / "protocols"
    protocols_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    audio_train = tmp_path / "audio" / "train"
    audio_dev = tmp_path / "audio" / "dev"
    audio_eval = tmp_path / "audio" / "eval"
    for d in (audio_train, audio_dev, audio_eval):
        d.mkdir(parents=True, exist_ok=True)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            manifests=manifests_dir,
            raw_data=tmp_path / "data" / "raw",
        ),
        dataset=SimpleNamespace(
            protocol_dir=protocols_dir,
            splits=SimpleNamespace(
                train="train.trn.txt",
                dev="dev.trl.txt",
                eval="eval.trl.txt",
            ),
            audio_dirs=SimpleNamespace(
                train=audio_train,
                dev=audio_dev,
                eval=audio_eval,
            ),
            sample_rate=16000,
        ),
    )
    return cfg


# ---------------------------------------------------------------------------
# Protocol Parsing Tests
# ---------------------------------------------------------------------------

class TestParseProtocol:
    def test_parse_train_official_format(self, protocol_writer):
        content = """
        LA_0079 LA_T_1138215 - - bonafide
        LA_0079 LA_T_1271820 - A06 spoof
        LA_0080 LA_T_1000137 - A01 spoof
        """
        path = protocol_writer("train.trn.txt", content)
        table = parse_protocol(path, split="train")

        assert len(table) == 3
        # Row 0: Bonafide
        assert table[0].speaker_id == "LA_0079"
        assert table[0].utt_id == "LA_T_1138215"
        assert table[0].attack_id == "-"
        assert table[0].key == "bonafide"
        assert table[0].label == 0

        # Row 1: Spoof A06
        assert table[1].speaker_id == "LA_0079"
        assert table[1].utt_id == "LA_T_1271820"
        assert table[1].attack_id == "A06"
        assert table[1].key == "spoof"
        assert table[1].label == 1

        # Row 2: Spoof A01
        assert table[2].attack_id == "A01"
        assert table[2].label == 1

    def test_parse_dev_official_format(self, protocol_writer):
        content = """
        LA_0040 LA_D_1047731 - - bonafide
        LA_0040 LA_D_1105538 - A03 spoof
        """
        path = protocol_writer("dev.trl.txt", content)
        table = parse_protocol(path, split="dev")
        assert len(table) == 2
        assert table[0].key == "bonafide"
        assert table[1].attack_id == "A03"

    def test_parse_eval_official_format(self, protocol_writer):
        content = """
        LA_0039 LA_E_2834764 - - bonafide
        LA_0039 LA_E_2834763 - A11 spoof
        LA_0045 LA_E_1000027 - A19 spoof
        """
        path = protocol_writer("eval.trl.txt", content)
        table = parse_protocol(path, split="eval")
        assert len(table) == 3
        assert table[1].attack_id == "A11"
        assert table[2].attack_id == "A19"

    def test_rejects_missing_tokens(self, protocol_writer):
        content = "LA_0079 LA_T_1138215 - bonafide"  # 4 tokens
        path = protocol_writer("bad_tokens.txt", content)
        with pytest.raises(ValueError, match="Expected 5 tokens"):
            parse_protocol(path, split="train")

    def test_rejects_invalid_placeholder(self, protocol_writer):
        content = "LA_0079 LA_T_1138215 env1 A01 spoof"  # column 2 is not '-'
        path = protocol_writer("bad_placeholder.txt", content)
        with pytest.raises(ValueError, match="Expected '-' in column 2"):
            parse_protocol(path, split="train")

    def test_rejects_invalid_key(self, protocol_writer):
        content = "LA_0079 LA_T_1138215 - - unknown_label"
        path = protocol_writer("bad_key.txt", content)
        with pytest.raises(ValueError, match="Invalid key"):
            parse_protocol(path, split="train")

    def test_rejects_bonafide_with_attack_id(self, protocol_writer):
        content = "LA_0079 LA_T_1138215 - A01 bonafide"
        path = protocol_writer("bonafide_with_attack.txt", content)
        with pytest.raises(ValueError, match="Bonafide utterance has non-dash system_id"):
            parse_protocol(path, split="train")

    def test_rejects_disallowed_attack_for_partition(self, protocol_writer):
        # A11 is an eval attack, not allowed in train
        content = "LA_0079 LA_T_1138215 - A11 spoof"
        path = protocol_writer("eval_attack_in_train.txt", content)
        with pytest.raises(ValueError, match="not permitted in train partition"):
            parse_protocol(path, split="train")

    def test_rejects_duplicate_utterance_within_split(self, protocol_writer):
        content = """
        LA_0079 LA_T_1138215 - - bonafide
        LA_0079 LA_T_1138215 - A01 spoof
        """
        path = protocol_writer("duplicate_utt.txt", content)
        with pytest.raises(ValueError, match="Duplicate utterance ID"):
            parse_protocol(path, split="train")


# ---------------------------------------------------------------------------
# Split Leakage & Speaker Disjointness Tests
# ---------------------------------------------------------------------------

class TestLeakageAndDisjointness:
    def test_clean_splits_pass(self):
        train = ManifestTable([
            ManifestRecord("spk_train_1", "utt_T_01", "-", "bonafide", 0),
            ManifestRecord("spk_train_2", "utt_T_02", "A01", "spoof", 1),
        ])
        dev = ManifestTable([
            ManifestRecord("spk_dev_1", "utt_D_01", "-", "bonafide", 0),
            ManifestRecord("spk_dev_2", "utt_D_02", "A02", "spoof", 1),
        ])
        eval_ = ManifestTable([
            ManifestRecord("spk_eval_1", "utt_E_01", "-", "bonafide", 0),
            ManifestRecord("spk_eval_2", "utt_E_02", "A11", "spoof", 1),
        ])

        summary = check_no_leakage(train, dev, eval_)
        assert summary["train_utts"] == 2
        assert summary["dev_utts"] == 2
        assert summary["eval_utts"] == 2
        assert summary["train_speakers"] == 2
        assert summary["dev_speakers"] == 2
        assert summary["eval_speakers"] == 2

    def test_utterance_overlap_raises_assertion_error(self):
        train = ManifestTable([
            ManifestRecord("spk_train_1", "utt_001", "-", "bonafide", 0),
        ])
        dev = ManifestTable([
            ManifestRecord("spk_dev_1", "utt_002", "-", "bonafide", 0),
        ])
        # Injected leak: utt_001 from train also in eval
        eval_ = ManifestTable([
            ManifestRecord("spk_eval_1", "utt_001", "A11", "spoof", 1),
        ])

        with pytest.raises(AssertionError, match="Utterance ID leakage detected"):
            check_no_leakage(train, dev, eval_)

    def test_speaker_overlap_hard_validation_failure(self):
        train = ManifestTable([
            ManifestRecord("LA_SHARED", "utt_T_01", "-", "bonafide", 0),
        ])
        dev = ManifestTable([
            ManifestRecord("LA_SHARED", "utt_D_01", "-", "bonafide", 0),  # Shared speaker
        ])
        eval_ = ManifestTable([
            ManifestRecord("LA_EVAL", "utt_E_01", "A11", "spoof", 1),
        ])

        with pytest.raises(AssertionError, match="Speaker identity overlap detected"):
            check_speaker_disjointness(train, dev, eval_)


# ---------------------------------------------------------------------------
# Manifest Building & Relative Path Tests
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_build_and_load_manifest_relative_paths(self, mock_config, protocol_writer):
        # Write mock protocol
        content = """
        LA_0079 LA_T_1000001 - - bonafide
        LA_0079 LA_T_1000002 - A02 spoof
        """
        protocol_path = mock_config.dataset.protocol_dir / "train.trn.txt"
        protocol_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")

        # Build manifest
        table = build_manifest(mock_config, split="train", force=True)
        assert len(table) == 2

        # Verify relative audio_path
        for r in table:
            assert r.audio_path.endswith(f"{r.utt_id}.flac")
            assert ":" not in r.audio_path, f"Windows drive letter found: {r.audio_path}"
            assert not r.audio_path.startswith("/"), f"Absolute root slash found: {r.audio_path}"

        # Load from disk and verify identity
        loaded = load_manifest(mock_config, split="train")
        assert len(loaded) == 2
        assert loaded[0].utt_id == "LA_T_1000001"
        assert loaded[1].attack_id == "A02"


# ---------------------------------------------------------------------------
# Deterministic Stratified Subset Tests
# ---------------------------------------------------------------------------

class TestStratifiedSubset:
    @pytest.fixture()
    def synthetic_train_manifest(self):
        records = []
        # 10 bonafide
        for i in range(10):
            records.append(ManifestRecord("spk_bonafide", f"utt_b_{i:02d}", "-", "bonafide", 0))
        # 60 spoof across A01-A06 (10 each)
        for atk_idx in range(1, 7):
            atk = f"A{atk_idx:02d}"
            for i in range(10):
                records.append(ManifestRecord(f"spk_{atk}", f"utt_{atk}_{i:02d}", atk, "spoof", 1))
        # Total = 70 records (10 bonafide, 60 spoof)
        return ManifestTable(records)

    def test_exact_determinism_across_runs(self, synthetic_train_manifest):
        sub1 = create_stratified_subset(synthetic_train_manifest, split="train", n_samples=35, seed=42)
        sub2 = create_stratified_subset(synthetic_train_manifest, split="train", n_samples=35, seed=42)

        assert len(sub1) == 35
        assert sub1["utt_id"] == sub2["utt_id"]
        assert sub1["attack_id"] == sub2["attack_id"]

    def test_preserves_class_ratio_and_attack_stratification(self, synthetic_train_manifest):
        # 70 total: 10 bonafide (14.28%), 60 spoof (85.71%)
        # Sample N=35 -> expect round(35 * 10/70) = 5 bonafide, 30 spoof
        # 30 spoof across 6 attacks -> exactly 5 per attack A01-A06
        sub = create_stratified_subset(synthetic_train_manifest, split="train", n_samples=35, seed=42)

        bonafide_count = sum(1 for r in sub if r.label == 0)
        spoof_count = sum(1 for r in sub if r.label == 1)
        assert bonafide_count == 5
        assert spoof_count == 30

        # Verify equal representation across attacks
        spoof_attacks = [r.attack_id for r in sub if r.label == 1]
        for atk_idx in range(1, 7):
            atk = f"A{atk_idx:02d}"
            assert spoof_attacks.count(atk) == 5

    def test_eval_split_isolation_guard(self, synthetic_train_manifest):
        with pytest.raises(ValueError, match="Evaluation data is strictly isolated"):
            create_stratified_subset(synthetic_train_manifest, split="eval", n_samples=20, for_eval=False)
