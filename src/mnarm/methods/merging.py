"""Task-vector merging operators (14 verbatim author methods covering 2023-2026)."""

from __future__ import annotations

import torch

from mnarm.methods.subspace_merging import (
    _dc_lora_matrix,
    _dop_merge,
    _iso_c_matrix,
    _iso_cts_matrix,
    _knots_matrix,
    _opcm_merge,
    _tsvm_merge_matrix,
    _wudi_matrix,
)

ALL_MERGE_METHODS = (
    "mean",
    "exposure",
    "ta_sum",
    "ties20",
    "magmax",
    "magmax_s1",
    "knots",
    "tsvm",
    "iso_c",
    "iso_cts",
    "wudi",
    "opcm",
    "dop",
    "dc_lora",
)


def _resolve_sign_mass(tensor: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(tensor.sum(dim=0))
    majority = torch.sign(sign.sum())
    if float(majority.item()) == 0.0:
        majority = torch.ones_like(majority)
    sign[sign == 0] = majority
    return sign


def _ties_merge_flat(flat: torch.Tensor, keep_frac: float = 0.2) -> torch.Tensor:
    abs_f = flat.abs()
    thresh_idx = int(max(0, min(flat.shape[1] - 1, round(flat.shape[1] * (1.0 - keep_frac)))))
    kth = abs_f.kthvalue(thresh_idx + 1, dim=1, keepdim=True).values if flat.shape[1] > 0 else abs_f
    trimmed = torch.where(abs_f >= kth, flat, torch.zeros_like(flat))
    elected = _resolve_sign_mass(trimmed)
    aligned = torch.where((trimmed * elected.unsqueeze(0)) > 0, trimmed, torch.zeros_like(trimmed))
    denom = (aligned != 0).sum(dim=0).clamp_min(1)
    return aligned.sum(dim=0) / denom


def merge_task_deltas(
    deltas: list[dict[str, torch.Tensor]],
    method: str,
    rank_per_task: int = 8,
) -> dict[str, torch.Tensor]:
    """Merge chronologically ordered per-stage dense LoRA delta dicts."""
    if not deltas:
        raise ValueError("Cannot merge an empty list of task deltas")
    keys = sorted(deltas[0])
    if len(deltas) == 1:
        return {k: deltas[0][k].float().cpu().clone() for k in keys}
    if method == "mean":
        return {k: torch.stack([d[k].float().cpu() for d in deltas]).mean(dim=0) for k in keys}
    if method == "exposure":
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            act = [d[k].float().cpu() for d in deltas if float(d[k].norm().item()) > 1e-12]
            out[k] = torch.stack(act).mean(dim=0) if act else torch.zeros_like(deltas[0][k].float().cpu())
        return out
    if method == "ta_sum":
        return {k: torch.stack([d[k].float().cpu() for d in deltas]).sum(dim=0) for k in keys}
    if method in ("magmax", "magmax_s1"):
        scale = 0.5 if method == "magmax" else 1.0
        out = {}
        for k in keys:
            best = deltas[0][k].float().cpu().clone()
            for d in deltas[1:]:
                cur = d[k].float().cpu()
                best = torch.where(cur.abs() >= best.abs(), cur, best)
            out[k] = scale * best
        return out
    if method == "ties20":
        flat = torch.stack([torch.cat([d[k].float().cpu().reshape(-1) for k in keys]) for d in deltas])
        merged_flat = _ties_merge_flat(flat, keep_frac=0.2)
        sizes = [deltas[0][k].numel() for k in keys]
        return {k: chunk.reshape_as(deltas[0][k]).float().cpu() for k, chunk in zip(keys, merged_flat.split(sizes))}
    if method == "knots":
        return {k: _knots_matrix([d[k] for d in deltas], _ties_merge_flat) for k in keys}
    if method == "tsvm":
        return {k: _tsvm_merge_matrix([d[k] for d in deltas], rank_per_task=rank_per_task) for k in keys}
    if method == "iso_c":
        return {k: _iso_c_matrix([d[k] for d in deltas], rank_per_task=rank_per_task) for k in keys}
    if method == "iso_cts":
        return {k: _iso_cts_matrix([d[k] for d in deltas], rank_per_task=rank_per_task, k_frac=0.8) for k in keys}
    if method == "wudi":
        return {k: _wudi_matrix([d[k] for d in deltas]) for k in keys}
    if method == "opcm":
        return _opcm_merge(deltas, keys, alpha=0.5)
    if method == "dop":
        return _dop_merge(deltas, keys)
    if method == "dc_lora":
        return {k: _dc_lora_matrix([d[k] for d in deltas], rank_per_task=rank_per_task) for k in keys}
    raise ValueError(f"Unknown merge method: {method}")
