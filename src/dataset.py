"""
src/dataset.py
==============
Protocol parsing, manifest generation, and split-safe data access for the
ASVspoof 2019 Logical Access (LA) dataset.

This module is the single source of truth for which utterances belong to
which split.  It enforces the train / dev / eval separation required by the
experiment design and raises an error if data-leakage-prone operations
(e.g., reading eval labels before the final evaluation stage) are attempted
outside their sanctioned scripts.

Public API (implemented in Stage 3)
-------------------------------------
parse_protocol(protocol_path)
    Read an ASVspoof 2019 LA protocol file and return a DataFrame with
    columns: [utt_id, speaker_id, attack_id, key, audio_path].

build_manifest(cfg, split)
    Generate (or load from cache) a manifest CSV for the requested split.

load_manifest(cfg, split)
    Load a previously saved manifest CSV as a DataFrame.

check_no_leakage(train_df, dev_df, eval_df)
    Assert that the three splits share no utterance IDs.

ATTACK_GROUPS : dict[str, list[str]]
    Maps 'seen' and 'unseen' to lists of attack IDs for the analysis stage.
    Seen attacks: A01–A06 (in train protocol).
    Unseen attacks: A07–A19 (not in train protocol).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Attack ID groupings for seen/unseen analysis
ATTACK_GROUPS: dict[str, list[str]] = {
    "seen":   [f"A{i:02d}" for i in range(1, 7)],   # A01–A06
    "unseen": [f"A{i:02d}" for i in range(7, 20)],   # A07–A19
}

SplitName = Literal["train", "dev", "eval"]


def parse_protocol(protocol_path: Path):  # -> pd.DataFrame
    """Parse an ASVspoof 2019 LA protocol file into a structured DataFrame.

    Parameters
    ----------
    protocol_path:
        Path to a protocol ``.txt`` file, e.g.
        ``ASVspoof2019.LA.cm.train.trn.txt``.

    Returns
    -------
    pd.DataFrame
        Columns: ``utt_id``, ``speaker_id``, ``attack_id``, ``key``.
        ``key`` is ``'bonafide'`` or ``'spoof'``.
        ``attack_id`` is ``'-'`` for bonafide utterances.

    Raises
    ------
    FileNotFoundError
        If ``protocol_path`` does not exist.
    ValueError
        If the protocol file has an unexpected format.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def build_manifest(cfg, split: SplitName):  # -> pd.DataFrame
    """Parse protocol + attach audio paths; save and return a manifest CSV.

    The manifest is saved to ``cfg.paths.manifests / f'{split}.csv'``.
    If the manifest already exists, it is loaded from disk rather than
    regenerated (idempotent).

    Parameters
    ----------
    cfg:
        Loaded ``Config`` instance from ``src.config.load_config()``.
    split:
        One of ``'train'``, ``'dev'``, ``'eval'``.

    Returns
    -------
    pd.DataFrame
        Manifest with columns:
        ``utt_id``, ``speaker_id``, ``attack_id``, ``key``, ``audio_path``.

    Raises
    ------
    FileNotFoundError
        If the protocol file or audio directory for the requested split
        does not exist.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def load_manifest(cfg, split: SplitName):  # -> pd.DataFrame
    """Load a previously generated manifest CSV from disk.

    Parameters
    ----------
    cfg:
        Loaded ``Config`` instance.
    split:
        One of ``'train'``, ``'dev'``, ``'eval'``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the manifest CSV has not been generated yet (run
        ``build_manifest`` first).
    """
    raise NotImplementedError("Implemented in Stage 3.")


def check_no_leakage(train_df, dev_df, eval_df) -> None:
    """Assert that the three splits share no utterance IDs.

    Parameters
    ----------
    train_df, dev_df, eval_df:
        DataFrames as returned by ``build_manifest`` / ``load_manifest``.

    Raises
    ------
    AssertionError
        If any utterance ID appears in more than one split.
    """
    raise NotImplementedError("Implemented in Stage 3.")
