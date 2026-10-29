"""
src/config.py
=============
Loads, validates, and exposes the project configuration from configs/base.yaml.

Design decisions
----------------
* Pydantic v2 models provide typed, validated access to every config value.
* All paths are stored as pathlib.Path objects resolved relative to the
  project root (the directory containing configs/).
* The module-level ``load_config`` function is the single entry point used
  by every script and source module.
* Config objects are frozen (immutable) after loading to prevent accidental
  mutation between stages.

Usage
-----
    from src.config import load_config
    cfg = load_config()          # reads configs/base.yaml
    print(cfg.paths.features)    # → PosixPath('data/features')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (parent of the ``configs/`` directory)."""
    # This file lives at <root>/src/config.py
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------

class PathsConfig(BaseModel, frozen=True):
    raw_data:  Path
    features:  Path
    manifests: Path
    models:    Path
    scores:    Path
    tables:    Path

    @model_validator(mode="before")
    @classmethod
    def resolve_relative(cls, values: dict) -> dict:
        root = _project_root()
        return {k: root / v for k, v in values.items()}


class SplitFilesConfig(BaseModel, frozen=True):
    train: str
    dev:   str
    eval:  str


class AudioDirsConfig(BaseModel, frozen=True):
    train: Path
    dev:   Path
    eval:  Path

    @model_validator(mode="before")
    @classmethod
    def resolve_relative(cls, values: dict) -> dict:
        root = _project_root()
        return {k: root / v for k, v in values.items()}


class DatasetConfig(BaseModel, frozen=True):
    protocol_dir: Path
    splits:       SplitFilesConfig
    audio_dirs:   AudioDirsConfig
    sample_rate:  int = Field(16000, gt=0)

    @field_validator("protocol_dir", mode="before")
    @classmethod
    def resolve_protocol_dir(cls, v: str) -> Path:
        return _project_root() / v


class EncoderConfig(BaseModel, frozen=True):
    name:       str
    hf_id:      str
    layer:      Literal["last", "weighted_sum"] = "last"
    pool:       Literal["mean"] = "mean"
    batch_size: int = Field(8, gt=0)


class NoiseConfig(BaseModel, frozen=True):
    snr_db_range: Tuple[float, float] = (10.0, 20.0)


class EchoConfig(BaseModel, frozen=True):
    delay_ms: float = 30.0
    decay:    float = 0.4


class OpusConfig(BaseModel, frozen=True):
    bitrate_kbps_range: Tuple[int, int] = (6, 16)


class Mp3Config(BaseModel, frozen=True):
    bitrate_kbps: int = 8


class G711Config(BaseModel, frozen=True):
    sample_rate: int = 8000


class AugmentationConfig(BaseModel, frozen=True):
    train_conditions:    List[str] = ["clean", "noise", "echo", "opus"]
    held_out_conditions: List[str] = ["mp3", "g711", "phone_chain"]
    noise:  NoiseConfig  = NoiseConfig()
    echo:   EchoConfig   = EchoConfig()
    opus:   OpusConfig   = OpusConfig()
    mp3:    Mp3Config    = Mp3Config()
    g711:   G711Config   = G711Config()

    @field_validator("train_conditions", "held_out_conditions", mode="before")
    @classmethod
    def no_overlap(cls, v: List[str]) -> List[str]:
        return v  # cross-field overlap checked in model_validator below

    @model_validator(mode="after")
    def conditions_disjoint(self) -> "AugmentationConfig":
        overlap = set(self.train_conditions) & set(self.held_out_conditions)
        if overlap:
            raise ValueError(
                f"train_conditions and held_out_conditions must not overlap; "
                f"found: {overlap}"
            )
        return self


class ClassifierConfig(BaseModel, frozen=True):
    type:         Literal["logistic_regression"] = "logistic_regression"
    C_candidates: List[float] = [0.001, 0.01, 0.1, 1.0]
    max_iter:     int  = Field(1000, gt=0)
    solver:       str  = "lbfgs"
    class_weight: Optional[str] = "balanced"

    @field_validator("C_candidates")
    @classmethod
    def positive_C(cls, v: List[float]) -> List[float]:
        if any(c <= 0 for c in v):
            raise ValueError("All C_candidates must be positive.")
        return v


class EvaluationConfig(BaseModel, frozen=True):
    primary_metric: Literal["EER"] = "EER"
    bootstrap_n:    int   = Field(10000, gt=0)
    ci_alpha:       float = Field(0.05, gt=0.0, lt=1.0)


class PilotConfig(BaseModel, frozen=True):
    n_files: int                        = Field(50, gt=0)
    split:   Literal["train", "dev", "eval"] = "train"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class Config(BaseModel, frozen=True):
    seed:          int               = 42
    paths:         PathsConfig
    dataset:       DatasetConfig
    encoders:      List[EncoderConfig]
    augmentation:  AugmentationConfig
    classifier:    ClassifierConfig
    evaluation:    EvaluationConfig
    pilot:         PilotConfig

    @field_validator("encoders")
    @classmethod
    def at_least_one_encoder(cls, v: List[EncoderConfig]) -> List[EncoderConfig]:
        if not v:
            raise ValueError("At least one encoder must be specified.")
        names = [e.name for e in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Encoder names must be unique; got {names}.")
        return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate the project configuration.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  When ``None`` (the default),
        resolves to ``<project_root>/configs/base.yaml``.

    Returns
    -------
    Config
        A fully validated, frozen ``Config`` instance.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    pydantic.ValidationError
        If any config value fails validation.
    """
    if path is None:
        path = _project_root() / "configs" / "base.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Have you placed configs/base.yaml in the project root?"
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return Config(**raw)
