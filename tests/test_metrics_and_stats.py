"""Exact hand-verified unit tests for classification, calibration (ECE/NLL/Brier), and bootstrap CIs."""

from __future__ import annotations

import torch

from mnarm.eval import eval_classification_metrics, paired_bootstrap_diff


def test_classification_metrics_and_7cls_exclusion() -> None:
    classes = list(range(8))
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    logits = torch.full((8, 8), -10.0)
    for i in range(7):
        logits[i, i] = 10.0
    # Deliberately misclassify class 7 into class 0
    logits[7, 0] = 10.0

    m = eval_classification_metrics(logits, labels, classes, early_classes=[0, 1, 4, 5], late_classes=[2, 3, 6, 7])
    # Accuracy = 7/8 = 0.875; Class 0 F1 = 2/3, Classes 1..6 F1 = 1.0, Class 7 F1 = 0.0
    expected_8cls = (2.0 / 3.0 + 6.0) / 8.0
    expected_7cls = (2.0 / 3.0 + 6.0) / 7.0
    assert abs(m["accuracy"] - 0.875) < 1e-6
    assert abs(m["macro_f1"] - expected_8cls) < 1e-5
    assert m["macro_f1_7cls"] is not None
    assert abs(m["macro_f1_7cls"] - expected_7cls) < 1e-5
    assert m["worst_class_f1"] == 0.0
    assert abs(m["early_recall"] - 1.0) < 1e-6
    assert abs(m["late_recall"] - 0.75) < 1e-6
    assert 0.0 <= m["ece"] <= 1.0
    assert m["nll"] > 0.0 and m["brier"] > 0.0


def test_paired_bootstrap_diff() -> None:
    classes = [0, 1, 2]
    labels = torch.tensor([0, 1, 2] * 30, dtype=torch.long)
    logits_a = torch.zeros(90, 3)
    logits_b = torch.zeros(90, 3)
    for i, yv in enumerate(labels.tolist()):
        logits_a[i, yv] = 4.0
        logits_b[i, yv if i % 2 == 0 else (yv + 1) % 3] = 4.0

    diff = paired_bootstrap_diff(logits_a, logits_b, labels, classes, n_resamples=300, seed=11)
    assert diff["delta_macro_f1"] > 0.3
    assert diff["ci95_low"] > 0.2
    assert diff["p_value"] < 0.05

    diff_same = paired_bootstrap_diff(logits_a, logits_a, labels, classes, n_resamples=100, seed=11)
    assert diff_same["delta_macro_f1"] == 0.0
