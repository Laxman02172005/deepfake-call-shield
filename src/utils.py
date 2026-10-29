"""
src/utils.py
============
Shared utility functions used across all scripts and source modules.

Provides
--------
* ``seed_everything``    – reproducible seeding for Python, NumPy, and PyTorch
* ``get_logger``         – consistent structured logging
* ``probe_ffmpeg``       – hard check for required FFmpeg encoders/decoders
* ``format_duration``    – human-readable elapsed-time strings
* ``MemoryTracker``      – context manager to measure peak RAM usage

Design decisions
----------------
* torch is imported lazily so that utils.py is usable before PyTorch is
  installed (e.g., in Stage 2 smoke tests and codec checks).
* ``probe_ffmpeg`` raises ``RuntimeError`` on ANY missing codec — it never
  silently falls back, in line with working rule 6.
* All public functions are importable individually for unit testing.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` from the calling module.
    level:
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured; avoid duplicate handlers

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.  Should come from ``cfg.seed``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # NumPy not yet installed; skip

    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass  # PyTorch not yet installed; skip


# ---------------------------------------------------------------------------
# FFmpeg codec probe
# ---------------------------------------------------------------------------

# Mapping: human-readable name → (direction, ffmpeg codec string)
_REQUIRED_CODECS: dict[str, tuple[str, str]] = {
    "Opus encoder":     ("encoder", "libopus"),
    "MP3 encoder":      ("encoder", "libmp3lame"),
    "G.711 encoder":    ("encoder", "pcm_mulaw"),
    "Opus decoder":     ("decoder", "libopus"),
    "MP3 decoder":      ("decoder", "mp3"),
    "G.711 decoder":    ("decoder", "pcm_mulaw"),
}


def probe_ffmpeg(
    required: Optional[Sequence[str]] = None,
) -> dict[str, bool]:
    """Check which FFmpeg encoders/decoders are available.

    Parameters
    ----------
    required:
        Subset of codec human-readable names from ``_REQUIRED_CODECS`` to
        check.  When ``None``, all known codecs are checked.

    Returns
    -------
    dict[str, bool]
        Maps each codec name to ``True`` (available) or ``False`` (missing).

    Raises
    ------
    RuntimeError
        If ``ffmpeg`` is not on PATH, or if any requested codec is missing.
        This function **never** silently continues with a missing codec.
    """
    logger = get_logger(__name__)

    # --- Check ffmpeg is on PATH ---
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg returned non-zero exit on -version check.")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not found on PATH.\n"
            "Install it with:  winget install --id Gyan.FFmpeg --source winget\n"
            "Then restart your terminal so PATH is updated."
        )

    targets = {k: v for k, v in _REQUIRED_CODECS.items()
               if required is None or k in required}

    availability: dict[str, bool] = {}
    missing: list[str] = []

    for label, (direction, codec_str) in targets.items():
        flag = "-encoders" if direction == "encoder" else "-decoders"
        proc = subprocess.run(
            ["ffmpeg", flag],
            capture_output=True, text=True, timeout=10,
        )
        found = codec_str in proc.stdout
        availability[label] = found
        status = "✓" if found else "✗ MISSING"
        logger.info("FFmpeg %s (%s): %s", direction, codec_str, status)
        if not found:
            missing.append(f"{label} ({codec_str})")

    if missing:
        raise RuntimeError(
            "The following required FFmpeg codecs are missing:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nDo NOT proceed with augmentation until these are available."
        )

    return availability


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples
    --------
    >>> format_duration(3661)
    '1h 01m 01s'
    >>> format_duration(45.3)
    '45.3s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


def project_extraction_time(
    pilot_seconds: float,
    pilot_n: int,
    total_n: int,
) -> tuple[float, str]:
    """Project full extraction time from a pilot timing run.

    Parameters
    ----------
    pilot_seconds:
        Wall-clock seconds for the pilot run.
    pilot_n:
        Number of files processed in the pilot.
    total_n:
        Total number of files in the full dataset.

    Returns
    -------
    (projected_seconds, human_readable_string)
    """
    per_file = pilot_seconds / pilot_n
    projected = per_file * total_n
    return projected, format_duration(projected)


# ---------------------------------------------------------------------------
# Memory tracker
# ---------------------------------------------------------------------------

@contextmanager
def track_peak_ram() -> Generator[dict, None, None]:
    """Context manager that records peak RAM usage of the enclosed block.

    Yields a dict that is populated with ``peak_mb`` when the block exits.

    Example
    -------
    >>> with track_peak_ram() as mem:
    ...     data = [0] * 10_000_000
    >>> print(f"Peak RAM: {mem['peak_mb']:.1f} MB")
    """
    info: dict[str, float] = {}
    tracemalloc.start()
    try:
        yield info
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info["peak_mb"] = peak / 1024 / 1024
