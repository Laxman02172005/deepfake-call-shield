"""
scripts/01_verify_data.py
=========================
Read-only dataset verification script.

Checks that the ASVspoof 2019 LA directory structure, protocol files, and
audio files are present and well-formed.  Writes a manifest CSV for each
split to data/manifests/.

Exits with code 0 on success, code 1 if any check fails.

Usage
-----
    python scripts/01_verify_data.py
    python scripts/01_verify_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so src/ is importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)


def main(config_path: Path | None = None) -> int:
    """Run all data verification checks.

    Returns
    -------
    int
        0 on success, 1 on any failure.
    """
    raise NotImplementedError("Implemented in Stage 3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ASVspoof 2019 LA dataset.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config YAML (default: configs/base.yaml)")
    args = parser.parse_args()
    sys.exit(main(args.config))
