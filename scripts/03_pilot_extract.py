"""
scripts/03_pilot_extract.py
===========================
Pilot extraction timing script.

Runs a frozen-encoder forward pass on a small sample of N files from the
train split, measures wall-clock time and peak RAM, projects to the full
dataset, and prints a PASS / WARN / FAIL verdict.

Full extraction (script 04) will not start unless this script exits with
code 0 (PASS) or the user explicitly overrides with --force.

Thresholds
----------
PASS  : projected time < 2 h  AND  estimated disk < 10 GB
WARN  : 2–6 h  OR  10–20 GB   (prompts user confirmation)
FAIL  : projected time > 6 h  OR  estimated disk > 20 GB

Usage
-----
    python scripts/03_pilot_extract.py
    python scripts/03_pilot_extract.py --encoder hubert_base --n-files 50
    python scripts/03_pilot_extract.py --force   # skip verdict gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)

PASS_TIME_H  = 2.0
WARN_TIME_H  = 6.0
PASS_DISK_GB = 10.0
WARN_DISK_GB = 20.0


def main(encoder_name: str | None, n_files: int, force: bool) -> int:
    """Run pilot extraction and print verdict.

    Returns
    -------
    int
        0 = PASS, 2 = WARN (user must confirm), 1 = FAIL.
    """
    raise NotImplementedError("Implemented in Stage 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot extraction timing estimate.")
    parser.add_argument("--encoder",  type=str,  default=None,
                        help="Encoder name from config (default: first in list)")
    parser.add_argument("--n-files",  type=int,  default=None,
                        help="Number of files for pilot (default: cfg.pilot.n_files)")
    parser.add_argument("--force",    action="store_true",
                        help="Skip the PASS/FAIL gate and proceed anyway")
    args = parser.parse_args()
    sys.exit(main(args.encoder, args.n_files, args.force))
