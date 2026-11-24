"""Task-vector merging operators (Mean, Task Arithmetic, TIES-20, MagMax, TSV-M)."""

from __future__ import annotations

import torch


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


def _tsvm_merge_matrix(mats: list[torch.Tensor]) -> torch.Tensor:
    active = [m.float().cpu() for m in mats if float(m.norm().item()) > 1e-12]
    if not active:
        return torch.zeros_like(mats[0].float().cpu())
    if len(active) == 1:
        return active[0].clone()
    rk = max(1, min(active[0].shape) // len(active))

    u_parts, s_parts, v_parts = [], [], []
    with torch.no_grad():
        for mat in active:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            keep = min(rk, s.shape[0])
            u_parts.append(u[:, :keep])
            s_parts.append(s[:keep])
            v_parts.append(vh[:keep, :].T)
        u_cat = torch.cat(u_parts, dim=1)
        s_cat = torch.cat(s_parts, dim=0)
        v_cat = torch.cat(v_parts, dim=1)
        uu, _, uhu = torch.linalg.svd(u_cat, full_matrices=False)
        uv, _, uhv = torch.linalg.svd(v_cat, full_matrices=False)
        return ((uu @ uhu) * s_cat.unsqueeze(0)) @ (uv @ uhv).T


def merge_task_deltas(deltas: list[dict[str, torch.Tensor]], method: str) -> dict[str, torch.Tensor]:
    """Merge chronologically ordered per-stage dense LoRA delta dicts."""
    if not deltas:
        raise ValueError("Cannot merge an empty list of task deltas")
    keys = sorted(deltas[0])
    if len(deltas) == 1:
        return {k: deltas[0][k].float().cpu().clone() for k in keys}

    if method == "mean":
        return {k: torch.stack([d[k].float().cpu() for d in deltas]).mean(dim=0) for k in keys}
    if method == "ta_sum":
        return {k: torch.stack([d[k].float().cpu() for d in deltas]).sum(dim=0) for k in keys}
    if method == "magmax_s1":
        out = {}
        for k in keys:
            stk = torch.stack([d[k].float().cpu() for d in deltas], dim=0)
            idx = stk.abs().argmax(dim=0, keepdim=True)
            out[k] = stk.gather(0, idx).squeeze(0)
        return out
    if method == "ties20":
        flat = torch.stack([torch.cat([d[k].float().cpu().reshape(-1) for k in keys]) for d in deltas])
        merged_flat = _ties_merge_flat(flat, keep_frac=0.2)
        sizes = [deltas[0][k].numel() for k in keys]
        return {k: chunk.reshape_as(deltas[0][k]).float().cpu() for k, chunk in zip(keys, merged_flat.split(sizes))}
    if method == "tsvm":
        return {k: _tsvm_merge_matrix([d[k] for d in deltas]) for k in keys}
    raise ValueError(f"Unknown merge method: {method}")
