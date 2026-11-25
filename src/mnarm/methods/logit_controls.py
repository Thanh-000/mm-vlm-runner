"""Zero-leakage logit debiasing controls (unlabeled centering and convex L-BFGS calibration bias)."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def apply_bias(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply additive per-class logit offset in float64 precision."""
    b = bias.to(device=logits.device, dtype=torch.float64)
    if logits.shape[-1] != b.numel():
        raise ValueError(f"Logit/bias class count mismatch: {logits.shape[-1]} vs {b.numel()}")
    return (logits.double() + b).to(logits.dtype)


def fit_unlabeled_center(scores: torch.Tensor) -> dict[str, Any]:
    """Fit label-free per-class centering offset (-mean logit per class across calibration inputs)."""
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("Invalid unlabeled calibration score tensor")
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite unlabeled calibration logits")
    b = (-scores.detach().cpu().double().mean(dim=0)).clone()
    return {"bias": b, "n_samples": int(scores.shape[0]), "labels_used": False}


def fit_cal_bias(scores: torch.Tensor, labels: torch.Tensor, classes: list[int]) -> dict[str, Any]:
    """Fit convex class-balanced additive bias via L-BFGS strong Wolfe (with Adam fallback)."""
    if scores.ndim != 2 or scores.shape[1] != len(classes) or len(labels) != scores.shape[0]:
        raise ValueError("Invalid calibration score/label shape")
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite calibration logits")
    cls_pos = {int(c): i for i, c in enumerate(classes)}
    with torch.inference_mode(False), torch.enable_grad():
        c = len(classes)
        targets = torch.tensor([cls_pos[int(y)] for y in labels.detach().cpu().tolist()], dtype=torch.long)
        z = scores.detach().cpu().double().clone()
        counts = torch.bincount(targets, minlength=c).clamp_min(1)
        weights = 1.0 / counts[targets].double()
        weights = weights / weights.sum().clamp_min(1e-12)

        def _objective(vec: torch.Tensor) -> torch.Tensor:
            return (F.cross_entropy(z + vec, targets, reduction="none") * weights).sum() + 0.01 * vec.square().sum()

        b = torch.zeros(c, dtype=torch.float64, requires_grad=True)
        lbfgs = torch.optim.LBFGS(
            [b], lr=1.0, max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe"
        )

        def _closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = _objective(b)
            loss.backward()
            return loss

        try:
            lbfgs.step(_closure)
        except Exception:
            pass

        def _check(vec: torch.Tensor) -> tuple[bool, float]:
            loss = _objective(vec)
            if not torch.isfinite(loss) or not torch.isfinite(vec).all():
                return False, float("inf")
            g = torch.autograd.grad(loss, vec)[0]
            return (bool(torch.isfinite(g).all()), float(g.abs().max().item())) if torch.isfinite(g).all() else (False, float("inf"))

        ok, grad_inf = _check(b)
        if not ok or grad_inf > 1e-4:
            b = torch.zeros(c, dtype=torch.float64, requires_grad=True)
            adam = torch.optim.Adam([b], lr=0.05)
            for _ in range(1000):
                adam.zero_grad(set_to_none=True)
                loss = _objective(b)
                loss.backward()
                adam.step()
            ok, grad_inf = _check(b)
        if not ok or grad_inf > 1e-3:
            raise ValueError(f"Calibration bias failed to converge (grad_inf={grad_inf})")
        return {"bias": b.detach().cpu().clone(), "n_samples": int(len(targets)), "grad_inf": grad_inf, "labels_used": True}
