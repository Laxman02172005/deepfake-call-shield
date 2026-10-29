"""
scripts/02_check_codecs.py
==========================
Probe FFmpeg for all required audio codecs.

Exits with code 0 if all codecs are present, code 1 if any are missing.
Never silently continues with a missing codec.

Usage
-----
    python scripts/02_check_codecs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, probe_ffmpeg

logger = get_logger(__name__)


def main() -> int:
    """Check all required FFmpeg codecs and report results.

    Returns
    -------
    int
        0 if all codecs found, 1 if any are missing.
    """
    logger.info("Probing FFmpeg for required codecs...")
    try:
        availability = probe_ffmpeg()
        for codec, ok in availability.items():
            status = "PASS" if ok else "FAIL"
            logger.info("  [%s] %s", status, codec)
        logger.info("All required FFmpeg codecs are available.")
        return 0
    except RuntimeError as exc:
        logger.error("Codec check failed:\n%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
