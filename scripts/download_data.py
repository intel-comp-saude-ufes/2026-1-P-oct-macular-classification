#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

MENDELEY_URL = "https://data.mendeley.com/datasets/rscbjbr9sj/3"


def unpack_archive(archive: Path, raw_data_dir: Path) -> None:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(raw_data_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare local raw data folder for the Kermany OCT dataset."
    )
    parser.add_argument("--raw-data-dir", default="data/raw", help="Destination for raw files.")
    parser.add_argument(
        "--archive",
        help="Path to a manually downloaded archive, for example OCT2017.zip.",
    )
    parser.add_argument(
        "--direct-url",
        help="Optional direct archive URL. The Mendeley landing page is not a stable direct file URL.",
    )
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        unpack_archive(archive, raw_data_dir)
        print(f"Extracted {archive} into {raw_data_dir}")
        return 0

    if args.direct_url:
        filename = args.direct_url.rstrip("/").split("/")[-1] or "dataset.zip"
        archive = raw_data_dir / filename
        urllib.request.urlretrieve(args.direct_url, archive)
        unpack_archive(archive, raw_data_dir)
        print(f"Downloaded and extracted {archive} into {raw_data_dir}")
        return 0

    print("Manual download required for the default dataset page.")
    print(f"1. Open: {MENDELEY_URL}")
    print("2. Download the OCT archive, usually named OCT2017.zip.")
    print(f"3. Run: python scripts/download_data.py --archive /path/OCT2017.zip --raw-data-dir {raw_data_dir}")
    print(f"Raw directory is ready for manual extraction: {raw_data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

