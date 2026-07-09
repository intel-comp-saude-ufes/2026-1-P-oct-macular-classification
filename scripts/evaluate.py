#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular.config import load_yaml, resolve_path
from oct_macular.data import validate_manifest
from oct_macular.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report,
    save_metrics,
)
from oct_macular.models import ManifestImageDataset, build_transforms, load_checkpoint_model

from train import class_names_from_config, encode_labels, load_hog_features


@torch.no_grad()
def predict_model(model, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    for inputs, labels in loader:
        scores = torch.softmax(model(inputs.to(device)), dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(torch.argmax(scores, dim=1).cpu().numpy().tolist())
        y_score.extend(scores.cpu().numpy().tolist())
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_score, dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved OCT run.")
    parser.add_argument("--run", required=True, help="Run directory under outputs.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    run_dir = resolve_path(args.run)
    config = load_yaml(run_dir / "config.yaml")
    class_names = class_names_from_config(config)
    manifest_path = resolve_path(config["data"]["manifest"])
    df = pd.read_csv(manifest_path)
    validate_manifest(df)
    rows = df[df["split"] == args.split]

    model_name = config["model"]["name"]
    if model_name == "hog_logreg":
        model = joblib.load(run_dir / "model.joblib")
        x = load_hog_features(rows, config)
        y_true = encode_labels(rows["label"], class_names)
        y_score = model.predict_proba(x)
        y_pred = np.argmax(y_score, axis=1)
    else:
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        device = torch.device(device_name)
        model, _ = load_checkpoint_model(run_dir / "best_model.pt", device)
        image_size = int(config.get("data", {}).get("image_size", 224))
        imagenet_norm = model_name == "mobilenetv3_small"
        class_to_idx = {label: index for index, label in enumerate(class_names)}
        dataset = ManifestImageDataset(
            rows,
            class_to_idx,
            transform=build_transforms(image_size, train=False, imagenet_norm=imagenet_norm),
        )
        loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("batch_size", 32)), shuffle=False)
        y_true, y_pred, y_score = predict_model(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, y_score, class_names)
    prefix = f"{args.split}_standalone"
    plot_confusion_matrix(y_true, y_pred, class_names, run_dir / f"{prefix}_confusion_matrix.png")
    plot_roc_curve(y_true, y_score, class_names, run_dir / f"{prefix}_roc_curve.png")
    save_classification_report(y_true, y_pred, class_names, run_dir / f"{prefix}_classification_report.txt")
    save_metrics(metrics, run_dir / f"{prefix}_metrics.json")
    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())

