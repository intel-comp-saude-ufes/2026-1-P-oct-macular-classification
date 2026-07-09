#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare data and train all configured models.")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/hog_logreg.yaml",
            "configs/simple_cnn.yaml",
            "configs/mobilenetv3.yaml",
        ],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not args.skip_prepare:
        run(
            [
                sys.executable,
                str(root / "scripts" / "prepare_data.py"),
                "--raw-data-dir",
                args.raw_data_dir,
                "--manifest",
                args.manifest,
            ]
        )
    for config in args.configs:
        run(
            [
                sys.executable,
                str(root / "scripts" / "train.py"),
                "--config",
                config,
                "--manifest",
                args.manifest,
            ]
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

