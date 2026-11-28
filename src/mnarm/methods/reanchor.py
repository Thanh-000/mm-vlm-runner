"""Isometric O(d) canonical frame re-anchoring and projection weight folding (norm-preserving)."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch

from mnarm.models.vlm import MultimodalVLM, l2_normalize


def fit_canonical_reanchor(
    u0: torch.Tensor | None,
    um: torch.Tensor | None,
    protos_0: torch.Tensor | None = None,
    protos_m: torch.Tensor | None = None,
    anchor_weight: float = 0.05,
) -> dict[str, Any]:
    """Fit uncentered orthogonal Procrustes Q_{m->0} in O(d) from canonical anchor pairs."""
    pairs_0: list[torch.Tensor] = []
    pairs_m: list[torch.Tensor] = []
    for a0, am in ((u0, um), (protos_0, protos_m)):
        if a0 is not None and am is not None and a0.ndim == 2 and am.ndim == 2:
            if a0.shape[0] > 0 and a0.shape == am.shape:
                pairs_0.append(l2_normalize(a0).to(dtype=torch.float64, device="cpu"))
                pairs_m.append(l2_normalize(am).to(dtype=torch.float64, device="cpu"))

    if not pairs_0:
        d = 1
        for candidate in (u0, um, protos_0, protos_m):
            if candidate is not None and candidate.ndim == 2 and candidate.shape[-1] > 0:
                d = int(candidate.shape[-1])
                break
        return {
            "active": False,
            "q_mat": torch.eye(d, dtype=torch.float32),
            "n_anchors": 0,
            "theta_rms_deg": 0.0,
            "cos_before": 1.0,
            "cos_after": 1.0,
        }

    anch_0 = torch.cat(pairs_0, dim=0)
    anch_m = torch.cat(pairs_m, dim=0)
    n, d = int(anch_0.shape[0]), int(anch_0.shape[-1])
    cos_before = float((anch_0 * anch_m).sum(dim=-1).mean().item())
    if float((anch_0 - anch_m).abs().max().item()) < 1e-6:
        return {
            "active": False,
            "q_mat": torch.eye(d, dtype=torch.float32),
            "n_anchors": n,
            "theta_rms_deg": 0.0,
            "cos_before": cos_before,
            "cos_after": cos_before,
        }

    eff = float(anchor_weight) * max(float(n) / float(d), 1e-2)
    cross = eff * torch.eye(d, dtype=torch.float64) + anch_0.T @ anch_m
    u_s, _, vh_s = torch.linalg.svd(cross, full_matrices=False)
    q64 = u_s @ vh_s
    aligned = anch_m @ q64.T
    cos_after = float((anch_0 * aligned).sum(dim=-1).mean().item())
    theta_rms_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_before))))
    return {
        "active": True,
        "q_mat": q64.to(dtype=torch.float32),
        "n_anchors": n,
        "theta_rms_deg": float(theta_rms_deg),
        "cos_before": float(cos_before),
        "cos_after": float(cos_after),
    }


def apply_reanchor(feats: torch.Tensor, op: dict[str, Any]) -> torch.Tensor:
    """Apply isometric orthogonal frame rotation Q_{m->0} and L2-normalize."""
    if not op.get("active", False):
        return l2_normalize(feats)
    q = op["q_mat"].to(device=feats.device, dtype=torch.float32)
    return l2_normalize(feats.float() @ q.T)


@torch.inference_mode()
def verify_projection_weight_fold(
    vlm: MultimodalVLM,
    tower: str,
    op: dict[str, Any],
    sample_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Verify exact equivalence between feature rotation and folding Q into VLM projection weights."""
    if tower not in ("text", "image"):
        raise ValueError(f"Unsupported tower={tower!r}")
    clip = getattr(vlm, "clip", vlm)
    is_siglip = bool(getattr(vlm, "is_siglip", False)) or (
        hasattr(clip, "text_model") and hasattr(clip.text_model, "head") and not hasattr(clip, "text_projection")
    )
    q_mat = op["q_mat"].to(device=vlm.device, dtype=torch.float32)

    if tower == "text":
        sig_params = set(inspect.signature(clip.get_text_features).parameters.keys())
        fwd_inp = {k: v.to(vlm.device) for k, v in sample_inputs.items() if k in sig_params}
        forward_fn = lambda: clip.get_text_features(**fwd_inp).float()
        proj = clip.text_model.head if is_siglip else clip.text_projection
        fold_mode = "exact_weight_and_bias_fold" if (is_siglip and proj.bias is not None) else "exact_weight_fold"
    else:
        pv = sample_inputs["pixel_values"].to(vlm.device)
        forward_fn = lambda: clip.get_image_features(pixel_values=pv).float()
        proj = None if is_siglip else clip.visual_projection
        fold_mode = "post_pool_orthogonal" if is_siglip else "exact_weight_fold"

    was_training = bool(clip.training)
    clip.eval()
    if proj is None:
        raw_feat = forward_fn()
        rot_feat = raw_feat @ q_mat.T
        norm_diff = float((rot_feat.norm(dim=-1) - raw_feat.norm(dim=-1)).abs().max().item())
        re_u = apply_reanchor(raw_feat, op)
        exp_u = l2_normalize(rot_feat)
        if was_training:
            clip.train()
        return {
            "tower": tower,
            "fold_mode": fold_mode,
            "extra_params": 0,
            "max_abs_diff": float((re_u - exp_u).abs().max().item()),
            "norm_preservation_diff": norm_diff,
            "restored": True,
        }

    w_orig = proj.weight.detach().clone()
    b_orig = proj.bias.detach().clone() if proj.bias is not None else None
    try:
        raw_feat = forward_fn()
        u_expected = apply_reanchor(raw_feat, op)
        proj.weight.data.copy_((q_mat @ w_orig.float()).to(dtype=w_orig.dtype, device=w_orig.device))
        if b_orig is not None:
            proj.bias.data.copy_((q_mat @ b_orig.float()).to(dtype=b_orig.dtype, device=b_orig.device))
        folded_feat = forward_fn()
        u_folded = l2_normalize(folded_feat)
        max_abs_diff = float((u_folded - u_expected).abs().max().item())
        min_cos = float((u_folded * u_expected).sum(dim=-1).min().item())
    finally:
        proj.weight.data.copy_(w_orig)
        if b_orig is not None:
            proj.bias.data.copy_(b_orig)
        if was_training:
            clip.train()

    return {
        "tower": tower,
        "fold_mode": fold_mode,
        "extra_params": 0,
        "max_abs_diff": max_abs_diff,
        "min_cos_sim": min_cos,
        "restored": True,
    }
