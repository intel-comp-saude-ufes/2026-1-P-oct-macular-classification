#!/usr/bin/env python
"""Compara estatisticamente dois modelos treinados no MESMO conjunto de teste.

Aplica o teste de McNemar exato (predições pareadas) e intervalos de confiança
por bootstrap para acurácia e F1 macro de cada modelo, além do IC da diferença
entre eles. Exemplo:

    python scripts/compare_models.py \
        --run-a outputs/<run_hog> \
        --run-b outputs/<run_cnn> \
        --split test

As predições dos dois runs precisam vir do mesmo manifesto/split, para que o
pareamento por amostra seja válido.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular.config import load_yaml, resolve_path
from oct_macular.stats import (
    accuracy,
    bootstrap_metric_ci,
    macro_f1,
    mcnemar_exact,
    paired_bootstrap_diff,
)


def get_predictions(run_dir: Path, split: str, device_arg: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reproduz as predições de um run salvo (espelha scripts/evaluate.py)."""
    import pandas as pd

    from oct_macular.data import validate_manifest

    # Import tardio de train.py para reaproveitar utilitários sem custo se não usado.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import class_names_from_config, encode_labels, load_hog_features

    config = load_yaml(run_dir / "config.yaml")
    class_names = class_names_from_config(config)
    manifest_path = resolve_path(config["data"]["manifest"])
    df = pd.read_csv(manifest_path)
    validate_manifest(df)
    rows = df[df["split"] == split].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Split '{split}' vazio no manifesto {manifest_path}.")

    y_true = encode_labels(rows["label"], class_names)
    model_name = config["model"]["name"]

    if model_name == "hog_logreg":
        import joblib

        model = joblib.load(run_dir / "model.joblib")
        x = load_hog_features(rows, config)
        y_pred = np.argmax(model.predict_proba(x), axis=1)
        return y_true, y_pred.astype(np.int64), class_names

    import torch
    from torch.utils.data import DataLoader

    from oct_macular.models import ManifestImageDataset, build_transforms, load_checkpoint_model

    device_name = "cuda" if device_arg == "auto" and torch.cuda.is_available() else device_arg
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
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 32)),
        shuffle=False,
    )
    preds: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return y_true, np.asarray(preds, dtype=np.int64), class_names


def _fmt(ci: dict[str, float]) -> str:
    return f"{ci['point']:.3f}  (IC95% {ci['lo']:.3f} – {ci['hi']:.3f})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparação estatística de dois runs.")
    parser.add_argument("--run-a", required=True, help="Pasta do primeiro run (baseline).")
    parser.add_argument("--run-b", required=True, help="Pasta do segundo run (candidato).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--bootstrap", type=int, default=2000, help="Nº de réplicas de bootstrap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", help="Caminho do JSON de saída (padrão: dentro de run-b).")
    args = parser.parse_args()

    run_a = resolve_path(args.run_a)
    run_b = resolve_path(args.run_b)
    name_a = load_yaml(run_a / "config.yaml").get("experiment", {}).get("name", run_a.name)
    name_b = load_yaml(run_b / "config.yaml").get("experiment", {}).get("name", run_b.name)

    y_true_a, y_pred_a, classes_a = get_predictions(run_a, args.split, args.device)
    y_true_b, y_pred_b, classes_b = get_predictions(run_b, args.split, args.device)

    if classes_a != classes_b:
        raise ValueError("Os dois runs usam conjuntos de classes diferentes.")
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            "Os rótulos verdadeiros não coincidem entre os runs. "
            "As predições não estão pareadas (manifesto/split diferentes)."
        )

    y_true = y_true_a
    num_classes = len(classes_a)
    f1_metric = macro_f1(num_classes)

    mcnemar = mcnemar_exact(y_true == y_pred_a, y_true == y_pred_b)
    acc_a = bootstrap_metric_ci(y_true, y_pred_a, accuracy, args.bootstrap, seed=args.seed)
    acc_b = bootstrap_metric_ci(y_true, y_pred_b, accuracy, args.bootstrap, seed=args.seed)
    f1_a = bootstrap_metric_ci(y_true, y_pred_a, f1_metric, args.bootstrap, seed=args.seed)
    f1_b = bootstrap_metric_ci(y_true, y_pred_b, f1_metric, args.bootstrap, seed=args.seed)
    acc_diff = paired_bootstrap_diff(y_true, y_pred_a, y_pred_b, accuracy, args.bootstrap, seed=args.seed)
    f1_diff = paired_bootstrap_diff(y_true, y_pred_a, y_pred_b, f1_metric, args.bootstrap, seed=args.seed)

    report = {
        "split": args.split,
        "n_samples": int(len(y_true)),
        "classes": classes_a,
        "model_a": {"name": name_a, "run": str(run_a), "accuracy": acc_a, "macro_f1": f1_a},
        "model_b": {"name": name_b, "run": str(run_b), "accuracy": acc_b, "macro_f1": f1_b},
        "mcnemar": mcnemar,
        "difference_b_minus_a": {"accuracy": acc_diff, "macro_f1": f1_diff},
    }

    output_path = Path(args.output) if args.output else run_b / f"compare_vs_{name_a}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    sig = "SIM" if mcnemar["p_value"] < 0.05 else "NÃO"
    print("=" * 60)
    print(f"Comparação no split '{args.split}'  (n = {len(y_true)} amostras)")
    print(f"  A = {name_a}")
    print(f"  B = {name_b}")
    print("-" * 60)
    print(f"Acurácia  A : {_fmt(acc_a)}")
    print(f"Acurácia  B : {_fmt(acc_b)}")
    print(f"F1 macro  A : {_fmt(f1_a)}")
    print(f"F1 macro  B : {_fmt(f1_b)}")
    print("-" * 60)
    print(
        f"Diferença de acurácia (B-A): {acc_diff['point']:+.3f} "
        f"(IC95% {acc_diff['lo']:+.3f} – {acc_diff['hi']:+.3f}); "
        f"P(B>A) = {acc_diff['prob_b_better']:.3f}"
    )
    print(
        f"Diferença de F1 macro (B-A): {f1_diff['point']:+.3f} "
        f"(IC95% {f1_diff['lo']:+.3f} – {f1_diff['hi']:+.3f}); "
        f"P(B>A) = {f1_diff['prob_b_better']:.3f}"
    )
    print("-" * 60)
    print(
        f"McNemar exato: A acerta/B erra = {mcnemar['a_correct_b_wrong']}, "
        f"A erra/B acerta = {mcnemar['a_wrong_b_correct']}, "
        f"p = {mcnemar['p_value']:.3e}  ->  diferença significativa (p<0,05)? {sig}"
    )
    print("=" * 60)
    print(f"Relatório salvo em {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
