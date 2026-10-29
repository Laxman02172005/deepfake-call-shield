"""
tests/test_config.py
====================
Unit tests for src/config.py — the only fully implemented module in Stage 2.

Tests cover:
* Loading the real base.yaml produces a valid Config object.
* All Path fields resolve to absolute paths.
* Pydantic validation rejects invalid values (bad C, overlapping conditions,
  non-positive batch_size, etc.).
* load_config raises FileNotFoundError for a missing YAML.
* Encoder names must be unique.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_yaml_path() -> Path:
    """Return the path to the real project config."""
    return Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


@pytest.fixture()
def tmp_yaml(tmp_path):
    """Factory: write a YAML string to a temp file and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test_config.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p
    return _write


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadRealConfig:
    def test_loads_without_error(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        assert isinstance(cfg, Config)

    def test_seed_is_int(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        assert isinstance(cfg.seed, int)

    def test_paths_are_absolute(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        for field_name in cfg.paths.model_fields:
            p = getattr(cfg.paths, field_name)
            assert p.is_absolute(), f"paths.{field_name} is not absolute: {p}"

    def test_two_encoders_present(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        assert len(cfg.encoders) == 2
        names = {e.name for e in cfg.encoders}
        assert "hubert_base" in names
        assert "wav2vec2_base" in names

    def test_train_held_out_conditions_disjoint(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        train_set = set(cfg.augmentation.train_conditions)
        held_out_set = set(cfg.augmentation.held_out_conditions)
        assert train_set.isdisjoint(held_out_set), (
            f"Conditions overlap: {train_set & held_out_set}"
        )

    def test_classifier_C_candidates_positive(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        assert all(c > 0 for c in cfg.classifier.C_candidates)

    def test_config_is_frozen(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        with pytest.raises(Exception):
            cfg.seed = 999  # type: ignore[misc]

    def test_audio_dirs_are_absolute(self, base_yaml_path):
        cfg = load_config(base_yaml_path)
        for split in ("train", "dev", "eval"):
            p = getattr(cfg.dataset.audio_dirs, split)
            assert p.is_absolute(), f"audio_dirs.{split} not absolute: {p}"


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_overlapping_conditions_rejected(self, tmp_yaml, base_yaml_path):
        """train_conditions and held_out_conditions must not share an entry."""
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        # inject overlap: add 'mp3' to train_conditions
        raw["augmentation"]["train_conditions"].append("mp3")
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):  # pydantic.ValidationError
            load_config(bad_path)

    def test_negative_C_rejected(self, tmp_yaml, base_yaml_path):
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        raw["classifier"]["C_candidates"] = [-0.1, 1.0]
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):
            load_config(bad_path)

    def test_zero_batch_size_rejected(self, tmp_yaml, base_yaml_path):
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        raw["encoders"][0]["batch_size"] = 0
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):
            load_config(bad_path)

    def test_duplicate_encoder_names_rejected(self, tmp_yaml, base_yaml_path):
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        raw["encoders"][1]["name"] = raw["encoders"][0]["name"]
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):
            load_config(bad_path)

    def test_empty_encoders_rejected(self, tmp_yaml, base_yaml_path):
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        raw["encoders"] = []
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):
            load_config(bad_path)

    def test_invalid_layer_rejected(self, tmp_yaml, base_yaml_path):
        raw = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
        raw["encoders"][0]["layer"] = "all_layers"  # not a valid Literal
        bad_path = tmp_yaml(yaml.dump(raw))
        with pytest.raises(Exception):
            load_config(bad_path)
