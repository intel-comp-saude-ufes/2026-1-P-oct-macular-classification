#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular.data import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a patient-safe OCT manifest.")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, summary = build_manifest(
        raw_data_dir=args.raw_data_dir,
        manifest_path=args.manifest,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    print(f"Manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

