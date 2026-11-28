"""Zero-leakage logit controls (unlabeled centering, GR-CLIP 3-mean gap removal, convex L-BFGS cal_bias)."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from mnarm.models.vlm import l2_normalize


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


def fit_gr_clip_gap(cal_feats_t: torch.Tensor | None, cal_feats_i: torch.Tensor | None) -> dict[str, Any]:
    """Fit zero-label GR-CLIP (NeurIPS 2025, arXiv:2507.19054 Alg. 1) modality document means."""
    has_t = cal_feats_t is not None and cal_feats_t.ndim == 2 and cal_feats_t.shape[0] >= 1
    has_i = cal_feats_i is not None and cal_feats_i.ndim == 2 and cal_feats_i.shape[0] >= 1
    d = (
        int(cal_feats_t.shape[-1])
        if (cal_feats_t is not None and cal_feats_t.ndim == 2)
        else (int(cal_feats_i.shape[-1]) if (cal_feats_i is not None and cal_feats_i.ndim == 2) else 1)
    )
    doc_text_mean = l2_normalize(cal_feats_t).detach().cpu().float().mean(dim=0) if has_t else torch.zeros(d, dtype=torch.float32)  # type: ignore[arg-type]
    doc_image_mean = l2_normalize(cal_feats_i).detach().cpu().float().mean(dim=0) if has_i else torch.zeros(d, dtype=torch.float32)  # type: ignore[arg-type]
    return {
        "doc_text_mean": doc_text_mean,
        "doc_image_mean": doc_image_mean,
        "has_t": bool(has_t),
        "has_i": bool(has_i),
        "active": bool(has_t or has_i),
        "labels_used": False,
        "n_t": int(cal_feats_t.shape[0]) if has_t else 0,  # type: ignore[union-attr]
        "n_i": int(cal_feats_i.shape[0]) if has_i else 0,  # type: ignore[union-attr]
    }


def gr_clip_logits(
    dev_t: torch.Tensor,
    dev_i: torch.Tensor,
    protos_full: torch.Tensor,
    seen_classes: list[int],
    scale: float,
    gap_op: dict[str, Any],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute verbatim GR-CLIP (MixedModalitySearch Alg. 1) 3-mean centered logits on all views."""
    cols = torch.tensor([int(c) for c in seen_classes], dtype=torch.long)
    p_all_n = l2_normalize(protos_full.detach().cpu().float())
    q_norm = p_all_n.index_select(0, cols)
    ut = l2_normalize(dev_t.detach().cpu().float())
    ui = l2_normalize(dev_i.detach().cpu().float())
    if gap_op.get("active", False):
        q_mean = p_all_n.mean(dim=0, keepdim=True) if (len(seen_classes) <= 2 and p_all_n.shape[0] > 2) else q_norm.mean(dim=0, keepdim=True)
        q_gr = l2_normalize(q_norm - q_mean)
        dt_c = ut - gap_op["doc_text_mean"].detach().cpu().float().unsqueeze(0) if gap_op.get("has_t", False) else ut
        di_c = ui - gap_op["doc_image_mean"].detach().cpu().float().unsqueeze(0) if gap_op.get("has_i", False) else ui
        e_t = l2_normalize(dt_c)
        e_i = l2_normalize(di_c)
        e_pair = l2_normalize(float(alpha) * dt_c + (1.0 - float(alpha)) * di_c)
        return {
            "text": (float(scale) * (e_t @ q_gr.T)).float(),
            "image": (float(scale) * (e_i @ q_gr.T)).float(),
            "paired": (float(scale) * (e_pair @ q_gr.T)).float(),
        }
    zt = (float(scale) * (ut @ q_norm.T)).float()
    zi = (float(scale) * (ui @ q_norm.T)).float()
    return {"text": zt, "image": zi, "paired": 0.5 * (zt + zi)}


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
