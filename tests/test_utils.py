"""
tests/test_utils.py
===================
Unit tests for src/utils.py — the other fully implemented Stage 2 module.

Tests cover:
* get_logger returns a Logger, doesn't duplicate handlers on repeated calls.
* seed_everything runs without error (torch/numpy absent in Stage 2 env).
* format_duration produces correct human-readable strings.
* project_extraction_time projects correctly.
* track_peak_ram context manager records a non-zero peak.
* probe_ffmpeg raises RuntimeError when ffmpeg is not on PATH
  (patched via monkeypatch so the test doesn't depend on the system).
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.utils import (
    format_duration,
    get_logger,
    probe_ffmpeg,
    project_extraction_time,
    seed_everything,
    track_peak_ram,
)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_logger(self):
        log = get_logger("test.utils")
        assert isinstance(log, logging.Logger)

    def test_no_duplicate_handlers(self):
        log1 = get_logger("test.dedup")
        log2 = get_logger("test.dedup")
        assert log1 is log2
        assert len(log1.handlers) == 1


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_runs_without_error(self):
        """Should not raise even if torch/numpy are not installed."""
        seed_everything(42)

    def test_accepts_different_seeds(self):
        seed_everything(0)
        seed_everything(2**31 - 1)


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (0.0,    "0.0s"),
        (45.3,   "45.3s"),
        (59.9,   "59.9s"),
        (60.0,   "1m 00s"),
        (90.0,   "1m 30s"),
        (3600.0, "1h 00m 00s"),
        (3661.0, "1h 01m 01s"),
        (7322.0, "2h 02m 02s"),
    ])
    def test_format(self, seconds, expected):
        assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# project_extraction_time
# ---------------------------------------------------------------------------

class TestProjectExtractionTime:
    def test_projection_correct(self):
        projected, human = project_extraction_time(
            pilot_seconds=10.0, pilot_n=50, total_n=25000
        )
        assert projected == pytest.approx(5000.0)
        assert "1h" in human  # 5000s ≈ 1h 23m

    def test_returns_tuple(self):
        result = project_extraction_time(10.0, 10, 100)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)


# ---------------------------------------------------------------------------
# track_peak_ram
# ---------------------------------------------------------------------------

class TestTrackPeakRam:
    def test_records_nonzero_peak(self):
        with track_peak_ram() as mem:
            _ = list(range(100_000))
        assert "peak_mb" in mem
        assert mem["peak_mb"] > 0

    def test_nested_use(self):
        with track_peak_ram() as outer:
            with track_peak_ram() as inner:
                _ = [0] * 10_000
        assert outer["peak_mb"] >= 0
        assert inner["peak_mb"] >= 0


# ---------------------------------------------------------------------------
# probe_ffmpeg
# ---------------------------------------------------------------------------

class TestProbeFFmpeg:
    def test_raises_when_ffmpeg_not_on_path(self):
        """Simulate ffmpeg not found by patching subprocess.run."""
        with patch("src.utils.subprocess.run",
                   side_effect=FileNotFoundError("ffmpeg not found")):
            with pytest.raises(RuntimeError, match="ffmpeg is not found on PATH"):
                probe_ffmpeg()

    def test_raises_when_codec_missing(self):
        """Simulate ffmpeg present but a codec absent."""
        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            if "-version" in cmd:
                mock.stdout = "ffmpeg version 6.0"
            else:
                mock.stdout = ""   # no codecs listed
            return mock

        with patch("src.utils.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="required FFmpeg codecs are missing"):
                probe_ffmpeg()

    def test_returns_dict_when_all_present(self):
        """Simulate ffmpeg with all codecs available."""
        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = (
                "ffmpeg version 6.0\n"
                " E libopus\n E libmp3lame\n E pcm_mulaw\n"
                " D libopus\n D mp3\n D pcm_mulaw\n"
            )
            return mock

        with patch("src.utils.subprocess.run", side_effect=fake_run):
            result = probe_ffmpeg()

        assert isinstance(result, dict)
        assert all(result.values()), "All codecs should be True when present"
