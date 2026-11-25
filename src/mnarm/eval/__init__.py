"""Evaluation metrics, calibration errors, cone geometry diagnostics, bootstrap stats, and profiler."""

from mnarm.eval.metrics import (
    compute_ece,
    eval_classification_metrics,
    eval_retrieval_r1,
    eval_view_suite,
    measure_cone_drift,
    paired_bootstrap_diff,
)
from mnarm.eval.profiler import Profiler

__all__ = [
    "Profiler",
    "compute_ece",
    "eval_classification_metrics",
    "eval_retrieval_r1",
    "eval_view_suite",
    "measure_cone_drift",
    "paired_bootstrap_diff",
]
