"""Comparação estatística entre dois classificadores no mesmo conjunto de teste.

O núcleo aqui depende apenas de ``numpy`` e da biblioteca padrão, para poder ser
testado sem carregar ``torch``/``scikit-learn``. As duas ferramentas centrais são:

- Teste de McNemar (exato, binomial): apropriado para comparar dois modelos
  avaliados nas MESMAS amostras de teste (predições pareadas, acerto/erro).
- Intervalos de confiança por *bootstrap*: quantificam a precisão de uma métrica
  e da diferença entre modelos reamostrando o conjunto de teste com reposição.
"""
from __future__ import annotations

from math import comb
from typing import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_f1(num_classes: int) -> Metric:
    """Fábrica de métrica F1 macro em numpy puro (sem scikit-learn)."""

    def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        f1s = []
        for cls in range(num_classes):
            tp = int(np.sum((y_pred == cls) & (y_true == cls)))
            fp = int(np.sum((y_pred == cls) & (y_true != cls)))
            fn = int(np.sum((y_pred != cls) & (y_true == cls)))
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
        return float(np.mean(f1s)) if f1s else 0.0

    return _macro_f1


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float | int]:
    """Teste de McNemar exato (binomial de duas caudas) sobre predições pareadas.

    ``correct_a`` e ``correct_b`` são vetores booleanos de acerto por amostra,
    alinhados (mesma amostra, mesma posição). Foca nos casos em que os modelos
    discordam:

    - ``n10``: A acerta e B erra.
    - ``n01``: A erra e B acerta.

    Sob a hipótese nula (mesma taxa de erro), cada discordância é uma moeda justa.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("Vetores de acerto precisam ter o mesmo tamanho.")

    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n = n10 + n01

    if n == 0:
        p_value = 1.0
    else:
        k = min(n10, n01)
        tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p_value = min(1.0, 2.0 * tail)

    # Estatística qui-quadrado com correção de continuidade (referência).
    chi2 = ((abs(n10 - n01) - 1) ** 2) / n if n > 0 else 0.0

    return {
        "n_discordant": n,
        "a_correct_b_wrong": n10,
        "a_wrong_b_correct": n01,
        "chi2_cc": float(chi2),
        "p_value": float(p_value),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """IC percentil por *bootstrap* de uma métrica para um único modelo."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = metric(y_true, y_pred)
    if n == 0:
        return {"point": point, "lo": point, "hi": point}
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric(y_true[idx], y_pred[idx])
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return {"point": float(point), "lo": lo, "hi": hi}


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: Metric,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """IC por *bootstrap* da diferença (B - A), reamostrando as mesmas posições.

    Reusa o mesmo reamostramento para os dois modelos (pareado), o que reduz a
    variância da diferença. ``prob_b_better`` é a fração de réplicas em que B
    supera A — uma leitura direta da robustez da vantagem.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = metric(y_true, y_pred_b) - metric(y_true, y_pred_a)
    if n == 0:
        return {"point": point, "lo": point, "hi": point, "prob_b_better": 0.0}
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        diffs[i] = metric(yt, y_pred_b[idx]) - metric(yt, y_pred_a[idx])
    lo = float(np.percentile(diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "point": float(point),
        "lo": lo,
        "hi": hi,
        "prob_b_better": float(np.mean(diffs > 0)),
    }
