from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from oct_macular import EXPECTED_CLASSES

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label: str
    patient_id: str
    source_split: str


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def extract_patient_id(image_path: Path, label: str) -> str:
    """Extract Kermany-style patient id from names like CNV-1016042-1.jpeg."""
    stem = image_path.stem
    parts = [part for part in re.split(r"[-_\s]+", stem) if part]
    if len(parts) >= 2 and parts[0].upper() in EXPECTED_CLASSES:
        return parts[1]
    if len(parts) >= 2 and parts[0].upper() == label.upper():
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return stem


def _child_class_dirs(directory: Path) -> dict[str, Path]:
    child_dirs = {
        child.name.upper(): child
        for child in directory.iterdir()
        if child.is_dir()
    }
    return {
        label: child_dirs[label]
        for label in EXPECTED_CLASSES
        if label in child_dirs
    }


def discover_split_roots(raw_data_dir: str | Path) -> list[tuple[Path, str, dict[str, Path]]]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    roots: list[tuple[Path, str, dict[str, Path]]] = []
    for dirpath, dirnames, _ in os.walk(raw_data_dir):
        path = Path(dirpath)
        class_dirs = _child_class_dirs(path)
        if set(class_dirs) != set(EXPECTED_CLASSES):
            continue
        split = SPLIT_ALIASES.get(path.name.lower(), "unsplit")
        roots.append((path, split, class_dirs))
        dirnames[:] = []
    return sorted(roots, key=lambda item: str(item[0]))


def collect_records(raw_data_dir: str | Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_paths: set[Path] = set()
    split_roots = discover_split_roots(raw_data_dir)
    if not split_roots:
        raise ValueError(
            "No OCT split root found. Expected directories containing all "
            f"classes: {', '.join(EXPECTED_CLASSES)}"
        )

    for _, split, class_dirs in split_roots:
        for label in EXPECTED_CLASSES:
            for image_path in sorted(class_dirs[label].rglob("*")):
                if not is_image_file(image_path):
                    continue
                resolved = image_path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                records.append(
                    ImageRecord(
                        image_path=resolved,
                        label=label,
                        patient_id=extract_patient_id(image_path, label),
                        source_split=split,
                    )
                )
    return records


def _distribution_score(labels: np.ndarray, train_idx: np.ndarray, heldout_idx: np.ndarray) -> float:
    all_counts = Counter(labels.tolist())
    train_counts = Counter(labels[train_idx].tolist())
    heldout_counts = Counter(labels[heldout_idx].tolist())
    score = 0.0
    for label, total in all_counts.items():
        target = total / len(labels)
        score += abs((train_counts[label] / max(1, len(train_idx))) - target)
        score += abs((heldout_counts[label] / max(1, len(heldout_idx))) - target)
    missing = set(all_counts) - set(heldout_counts)
    return score + 10.0 * len(missing)


def group_split_indices(
    records: list[ImageRecord],
    heldout_size: float,
    seed: int,
    attempts: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < heldout_size < 1:
        raise ValueError("heldout_size must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required to split")

    labels = np.array([record.label for record in records])
    groups = np.array([record.patient_id for record in records])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two patient groups are required to split")

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=heldout_size,
            random_state=seed + offset,
        )
        train_idx, heldout_idx = next(splitter.split(labels, labels, groups))
        score = _distribution_score(labels, train_idx, heldout_idx)
        if best is None or score < best[0]:
            best = (score, train_idx, heldout_idx)
            if score < 1e-9:
                break
    if best is None:
        raise RuntimeError("Could not create a group-aware split")
    return np.sort(best[1]), np.sort(best[2])


def _assign(records: list[ImageRecord], split: str) -> list[dict[str, str]]:
    return [
        {
            "image_path": str(record.image_path),
            "label": record.label,
            "patient_id": record.patient_id,
            "split": split,
        }
        for record in records
    ]


def validate_manifest(df: pd.DataFrame) -> None:
    required = {"image_path", "label", "patient_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    invalid_labels = sorted(set(df["label"]) - set(EXPECTED_CLASSES))
    if invalid_labels:
        raise ValueError(f"Manifest contains invalid labels: {invalid_labels}")

    invalid_splits = sorted(set(df["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Manifest contains invalid splits: {invalid_splits}")

    missing_paths = [path for path in df["image_path"] if not Path(path).exists()]
    if missing_paths:
        raise ValueError(f"Manifest contains missing image paths, first: {missing_paths[0]}")

    split_patients = {
        split: set(df.loc[df["split"] == split, "patient_id"])
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": split_patients["train"] & split_patients["val"],
        "train_test": split_patients["train"] & split_patients["test"],
        "val_test": split_patients["val"] & split_patients["test"],
    }
    leaking = {name: values for name, values in overlaps.items() if values}
    if leaking:
        examples = {name: sorted(values)[:5] for name, values in leaking.items()}
        raise ValueError(f"Patient overlap across splits detected: {examples}")


def write_summary(df: pd.DataFrame, summary_path: str | Path) -> dict[str, object]:
    summary = {
        "num_images": int(len(df)),
        "classes": EXPECTED_CLASSES,
        "splits": {
            split: {
                "num_images": int(len(part)),
                "num_patients": int(part["patient_id"].nunique()),
                "labels": {
                    label: int(count)
                    for label, count in part["label"].value_counts().sort_index().items()
                },
            }
            for split, part in df.groupby("split", sort=True)
        },
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_manifest(
    raw_data_dir: str | Path,
    manifest_path: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    records = collect_records(raw_data_dir)
    by_source = defaultdict(list)
    for record in records:
        by_source[record.source_split].append(record)

    rows: list[dict[str, str]] = []
    has_official_train_test = bool(by_source["train"]) and bool(by_source["test"])
    if has_official_train_test:
        train_pool = by_source["train"] + by_source["val"]
        train_idx, val_idx = group_split_indices(train_pool, val_size, seed)
        rows.extend(_assign([train_pool[index] for index in train_idx], "train"))
        rows.extend(_assign([train_pool[index] for index in val_idx], "val"))
        rows.extend(_assign(by_source["test"], "test"))
    else:
        train_val_idx, test_idx = group_split_indices(records, test_size, seed)
        train_val = [records[index] for index in train_val_idx]
        train_idx, val_idx = group_split_indices(train_val, val_size, seed + 10_000)
        rows.extend(_assign([train_val[index] for index in train_idx], "train"))
        rows.extend(_assign([train_val[index] for index in val_idx], "val"))
        rows.extend(_assign([records[index] for index in test_idx], "test"))

    df = pd.DataFrame(rows, columns=["image_path", "label", "patient_id", "split"])
    df = df.sort_values(["split", "label", "patient_id", "image_path"]).reset_index(drop=True)
    validate_manifest(df)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    summary = write_summary(df, manifest_path.with_name("summary.json"))
    return df, summary

