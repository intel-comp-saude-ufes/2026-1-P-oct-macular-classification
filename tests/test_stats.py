from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular.stats import (
    accuracy,
    bootstrap_metric_ci,
    macro_f1,
    mcnemar_exact,
    paired_bootstrap_diff,
)


def test_accuracy_basic():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 0])
    assert accuracy(y_true, y_pred) == 0.75


def test_macro_f1_perfect_and_known():
    f1 = macro_f1(2)
    y_true = np.array([0, 0, 1, 1])
    assert f1(y_true, y_true) == 1.0
    # Prever tudo classe 0: classe 0 tem prec=0.5/rec=1 (F1=2/3), classe 1 F1=0.
    # Macro = (2/3 + 0) / 2 = 1/3 (mesmo valor do scikit-learn).
    y_pred = np.array([0, 0, 0, 0])
    assert abs(f1(y_true, y_pred) - 1 / 3) < 1e-12


def test_mcnemar_no_disagreement_is_nonsignificant():
    correct = np.array([True, True, False, False])
    result = mcnemar_exact(correct, correct)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_exact_known_value():
    # 10 discordâncias, todas a favor de B: p = 2 * 0.5**10 = 0.001953125
    correct_a = np.array([False] * 10 + [True] * 5)
    correct_b = np.array([True] * 10 + [True] * 5)
    result = mcnemar_exact(correct_a, correct_b)
    assert result["a_wrong_b_correct"] == 10
    assert result["a_correct_b_wrong"] == 0
    assert abs(result["p_value"] - 2 * (0.5 ** 10)) < 1e-12
    assert result["p_value"] < 0.05


def test_mcnemar_symmetric_is_nonsignificant():
    correct_a = np.array([True] * 5 + [False] * 5)
    correct_b = np.array([False] * 5 + [True] * 5)
    result = mcnemar_exact(correct_a, correct_b)
    assert result["a_correct_b_wrong"] == 5
    assert result["a_wrong_b_correct"] == 5
    assert result["p_value"] == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=500)
    y_pred = y_true.copy()
    flip = rng.random(500) < 0.2
    y_pred[flip] = (y_pred[flip] + 1) % 4
    ci = bootstrap_metric_ci(y_true, y_pred, accuracy, n_boot=500, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0.7 < ci["point"] < 0.9


def test_paired_bootstrap_detects_clear_winner():
    rng = np.random.default_rng(2)
    n = 400
    y_true = rng.integers(0, 4, size=n)
    y_pred_a = y_true.copy()
    a_flip = rng.random(n) < 0.4  # baseline ruim
    y_pred_a[a_flip] = (y_pred_a[a_flip] + 1) % 4
    y_pred_b = y_true.copy()
    b_flip = rng.random(n) < 0.05  # candidato bom
    y_pred_b[b_flip] = (y_pred_b[b_flip] + 1) % 4
    diff = paired_bootstrap_diff(y_true, y_pred_a, y_pred_b, accuracy, n_boot=500, seed=3)
    assert diff["point"] > 0
    assert diff["lo"] > 0  # IC da diferença exclui zero
    assert diff["prob_b_better"] > 0.99
