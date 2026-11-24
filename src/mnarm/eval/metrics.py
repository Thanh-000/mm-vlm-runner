"""ICCV classification, calibration (ECE/NLL/Brier), cone geometry, and paired bootstrap metrics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from mnarm.models.vlm import l2_normalize


def compute_ece(probs: torch.Tensor, y_local: torch.Tensor, n_bins: int = 15) -> float:
    """Compute top-1 Expected Calibration Error with equal-width confidence bins."""
    if probs.shape[0] == 0:
        return 0.0
    conf, pred = probs.max(dim=-1)
    correct = (pred == y_local).float()
    ece = 0.0
    n = float(probs.shape[0])
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    for b_idx in range(n_bins):
        lo, hi = edges[b_idx], edges[b_idx + 1]
        in_bin = (conf >= lo) & (conf <= hi) if b_idx == 0 else (conf > lo) & (conf <= hi)
        cnt = int(in_bin.sum().item())
        if cnt > 0:
            ece += (float(cnt) / n) * abs(float(correct[in_bin].mean().item()) - float(conf[in_bin].mean().item()))
    return float(ece)


def _macro_f1_from_cm(cm: torch.Tensor, class_indices: list[int]) -> float:
    denom = (cm.sum(dim=0) + cm.sum(dim=1)).double()
    tp = cm.diag().double()
    f1_per_cls = torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1.0), torch.zeros_like(tp))
    return float(f1_per_cls[class_indices].mean().item())


def eval_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    early_classes: list[int] | None = None,
    late_classes: list[int] | None = None,
) -> dict[str, Any]:
    """Compute Macro-F1 (all & 7-cls), Accuracy, Early/Late Recall, Worst-Class F1, ECE, NLL, and Brier."""
    cls_list = [int(c) for c in classes]
    z = logits.detach().cpu().float()
    y = labels.detach().cpu().long()
    if z.shape != (len(y), len(cls_list)) or not torch.isfinite(z).all():
        raise ValueError(f"Invalid logits shape/finiteness: {tuple(z.shape)} vs n={len(y)}, k={len(cls_list)}")

    cls_map = {c: i for i, c in enumerate(cls_list)}
    y_local = torch.tensor([cls_map[int(v)] for v in y.tolist()], dtype=torch.long)
    probs = F.softmax(z, dim=-1)
    nll = float(F.cross_entropy(z, y_local).item())
    y_oh = F.one_hot(y_local, num_classes=len(cls_list)).float()
    brier = float((probs - y_oh).square().sum(dim=-1).mean().item())
    ece = compute_ece(probs, y_local, n_bins=15)

    pred_local = z.argmax(dim=-1)
    k = len(cls_list)
    cm = torch.bincount(k * y_local + pred_local, minlength=k * k).reshape(k, k)
    denom = (cm.sum(dim=0) + cm.sum(dim=1)).double()
    tp = cm.diag().double()
    per_cls_f1 = [float(2.0 * tp[i] / denom[i]) if denom[i] > 0 else 0.0 for i in range(k)]
    macro_f1 = sum(per_cls_f1) / max(k, 1)
    worst_class_f1 = min(per_cls_f1) if per_cls_f1 else 0.0
    acc = float((cm.diag().sum() / max(int(cm.sum().item()), 1)).item())

    recalls = {cls_list[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else None) for i in range(k)}
    early_vals = [recalls[c] for c in (early_classes or []) if recalls.get(c) is not None]
    late_vals = [recalls[c] for c in (late_classes or []) if recalls.get(c) is not None]
    early_rec = sum(early_vals) / len(early_vals) if early_vals else None
    late_rec = sum(late_vals) / len(late_vals) if late_vals else None

    cls_7 = [i for i, c in enumerate(cls_list) if c != 7]
    macro_f1_7cls = (sum(per_cls_f1[i] for i in cls_7) / len(cls_7)) if (len(cls_list) == 8 and 7 in cls_map and cls_7) else None

    return {
        "n": int(len(y)),
        "macro_f1": float(macro_f1),
        "macro_f1_7cls": float(macro_f1_7cls) if macro_f1_7cls is not None else None,
        "accuracy": float(acc),
        "early_recall": float(early_rec) if early_rec is not None else None,
        "late_recall": float(late_rec) if late_rec is not None else None,
        "worst_class_f1": float(worst_class_f1),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier),
    }


def eval_view_suite(
    logits_by_view: dict[str, torch.Tensor],
    labels: torch.Tensor,
    classes: list[int],
    early_classes: list[int] | None = None,
    late_classes: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate text, image, and paired views plus 3-view mean and worst-view summaries."""
    views = {
        vw: eval_classification_metrics(logits_by_view[vw], labels, classes, early_classes, late_classes)
        for vw in ("text", "image", "paired")
        if vw in logits_by_view
    }
    f1_vals = [views[vw]["macro_f1"] for vw in ("text", "image", "paired") if vw in views]
    summary = {
        "view3_mean_f1": sum(f1_vals) / max(len(f1_vals), 1),
        "worst_view_f1": min(f1_vals) if f1_vals else 0.0,
    }
    if "paired" in views:
        er = views["paired"]["early_recall"]
        lr = views["paired"]["late_recall"]
        if er is not None and lr is not None:
            summary["paired_worst_group_recall"] = min(er, lr)
    return {**views, "summary": summary}


def measure_cone_drift(u_adapted: torch.Tensor | None, u_base: torch.Tensor | None) -> dict[str, Any]:
    """Measure global cone-center angular drift (deg) and spread radii r_0, r_m on unit sphere."""
    if u_adapted is None or u_base is None or u_adapted.shape[0] == 0:
        return {"theta_deg": 0.0, "r_0": None, "r_m": None, "kappa_raw": 1.0, "top1_ev": 0.0}

    u0 = l2_normalize(u_base).to(dtype=torch.float64, device="cpu")
    um = l2_normalize(u_adapted).to(dtype=torch.float64, device="cpu")
    m0 = u0.mean(dim=0)
    mm = um.mean(dim=0)
    r0 = float(m0.norm().item())
    rm = float(mm.norm().item())
    mu0 = m0 / max(r0, 1e-12)
    mum = mm / max(rm, 1e-12)
    cos_t = max(-1.0, min(1.0, float(torch.dot(mu0, mum).item())))
    theta_deg = math.degrees(math.acos(cos_t))

    r0_c = min(max(r0, 1e-6), 1.0 - 1e-6)
    rm_c = min(max(rm, 1e-6), 1.0 - 1e-6)
    kap_0 = r0_c / math.sqrt(1.0 - r0_c * r0_c)
    kap_m = rm_c / math.sqrt(1.0 - rm_c * rm_c)

    svals = torch.linalg.svdvals(um - u0)
    s2 = svals.square()
    top1_ev = float((s2[0] / s2.sum().clamp_min(1e-24)).item()) if s2.numel() > 0 else 0.0
    return {
        "theta_deg": float(theta_deg),
        "r_0": float(r0),
        "r_m": float(rm),
        "kappa_raw": float(kap_0 / max(kap_m, 1e-12)),
        "top1_ev": float(top1_ev),
    }


def paired_bootstrap_diff(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    n_resamples: int = 1000,
    seed: int = 11,
) -> dict[str, float]:
    """Paired bootstrap 95% CI and two-sided p-value for Macro-F1 difference (A minus B)."""
    cls_map = {int(c): i for i, c in enumerate(classes)}
    k = len(classes)
    y_loc = torch.tensor([cls_map[int(v)] for v in labels.detach().cpu().tolist()], dtype=torch.long)
    pred_a = logits_a.detach().cpu().argmax(dim=-1)
    pred_b = logits_b.detach().cpu().argmax(dim=-1)
    n = int(y_loc.shape[0])
    idx_all = list(range(k))

    def _f1(pred: torch.Tensor, ys: torch.Tensor) -> float:
        cm = torch.bincount(k * ys + pred, minlength=k * k).reshape(k, k)
        return _macro_f1_from_cm(cm, idx_all)

    point_diff = _f1(pred_a, y_loc) - _f1(pred_b, y_loc)
    gen = torch.Generator().manual_seed(int(seed))
    boot_diffs: list[float] = []
    for _ in range(int(n_resamples)):
        idx = torch.randint(0, n, (n,), generator=gen)
        ys = y_loc.index_select(0, idx)
        boot_diffs.append(_f1(pred_a.index_select(0, idx), ys) - _f1(pred_b.index_select(0, idx), ys))

    boot_t = torch.tensor(boot_diffs, dtype=torch.float64).sort().values
    ci_lo = float(torch.quantile(boot_t, 0.025).item())
    ci_hi = float(torch.quantile(boot_t, 0.975).item())
    p_pos = float((boot_t <= 0.0).float().mean().item())
    p_neg = float((boot_t >= 0.0).float().mean().item())
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "delta_macro_f1": float(point_diff),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_value": float(p_two),
        "n_resamples": int(n_resamples),
    }
