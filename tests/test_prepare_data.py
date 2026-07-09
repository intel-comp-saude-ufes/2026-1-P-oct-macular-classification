from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular import EXPECTED_CLASSES
from oct_macular.data import build_manifest


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(path)


def test_build_manifest_preserves_official_test_and_blocks_patient_overlap(tmp_path: Path):
    raw = tmp_path / "raw" / "OCT2017"
    for label in EXPECTED_CLASSES:
        for patient in range(4):
            make_image(raw / "train" / label / f"{label}-{label.lower()}train{patient}-1.jpeg")
        for patient in range(2):
            make_image(raw / "test" / label / f"{label}-{label.lower()}test{patient}-1.jpeg")

    make_image(tmp_path / "raw" / "chest_xray" / "train" / "NORMAL" / "NORMAL-chest-1.jpeg")
    make_image(tmp_path / "raw" / "chest_xray" / "train" / "PNEUMONIA" / "PNEUMONIA-chest-1.jpeg")

    manifest_path = tmp_path / "processed" / "manifest.csv"
    df, summary = build_manifest(raw.parent, manifest_path, val_size=0.25, seed=7)

    assert manifest_path.exists()
    assert set(df["label"]) == set(EXPECTED_CLASSES)
    assert not df["image_path"].str.contains("chest_xray").any()
    assert summary["num_images"] == len(df)

    train_patients = set(df.loc[df["split"] == "train", "patient_id"])
    val_patients = set(df.loc[df["split"] == "val", "patient_id"])
    test_patients = set(df.loc[df["split"] == "test", "patient_id"])
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
    assert all(patient.endswith("test0") or patient.endswith("test1") for patient in test_patients)

    reloaded = pd.read_csv(manifest_path)
    assert list(reloaded.columns) == ["image_path", "label", "patient_id", "split"]

