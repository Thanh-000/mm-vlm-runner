"""Subspace LoRA merging operators (KnOTS, TSV-M, ISO-C, ISO-CTS, DC-LoRA) + iterative re-exports."""

from __future__ import annotations

from typing import Callable

import torch

from mnarm.methods.iterative_merging import (
    _dop_merge,
    _dop_merge_pair,
    _opcm_merge,
    _prep_active,
    _wudi_matrix,
    _wudi_merge,
)


def _svd_dev() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tsvm_merge_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim TSV-M on LoRA deltas (DC-Merge/TSV-M: truncate each task to active LoRA rank)."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev = _svd_dev()
    n = len(active)
    min_dim = min(active[0].shape)
    r_lora = max(1, int(rank_per_task))
    rk = max(1, min(r_lora, min_dim // n)) if n * r_lora > min_dim else max(1, min(r_lora, min_dim))
    u_parts, s_parts, v_parts = [], [], []
    with torch.no_grad():
        for mat in active:
            u, s, vh = torch.linalg.svd(mat.to(dev), full_matrices=False)
            keep = min(rk, int(s.numel()))
            u_parts.append(u[:, :keep])
            s_parts.append(s[:keep])
            v_parts.append(vh[:keep, :].T)
        uu, _, uhu = torch.linalg.svd(torch.cat(u_parts, dim=1), full_matrices=False)
        uv, _, uhv = torch.linalg.svd(torch.cat(v_parts, dim=1), full_matrices=False)
        return (((uu @ uhu) * torch.cat(s_parts, dim=0).unsqueeze(0)) @ (uv @ uhv).T).float().cpu()


def _iso_c_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim ISO-C (ICML 2025) for low-rank LoRA: truncate to active rank before spectrum flattening."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev = _svd_dev()
    combined = torch.stack([m.to(dev) for m in active], dim=0).sum(dim=0)
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(combined, full_matrices=False)
        pos = int((s > 1e-6).sum().item())
        r_act = max(1, min(len(active) * max(1, int(rank_per_task)), pos, int(s.numel())))
        s_bar = float(s[:r_act].mean().item())
        return ((u[:, :r_act] * s_bar) @ vh[:r_act, :]).float().cpu()


def _iso_cts_matrix(mats: list[torch.Tensor], rank_per_task: int = 8, k_frac: float = 0.8) -> torch.Tensor:
    """Verbatim ISO-CTS (ICML 2025 & DC-Merge LoRA): common + residual task-specific subspaces."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev = _svd_dev()
    n = len(active)
    act_d = [m.to(dev) for m in active]
    delta_ta = torch.stack(act_d, dim=0).sum(dim=0)
    min_dim = min(delta_ta.shape)
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(delta_ta, full_matrices=False)
        pos = int((s > 1e-6).sum().item())
        r_tot = max(1, min(n * max(1, int(rank_per_task)), min_dim, pos, int(s.numel())))
        common_s = int(r_tot * float(k_frac))
        ts_tot = int(round((r_tot - common_s) / float(n))) * n
        common_s = r_tot - ts_tot
        n_dims_per_task = (r_tot - common_s) // n
        if n_dims_per_task < 1 or common_s < 1:
            return _iso_c_matrix(active, rank_per_task=rank_per_task)

        u_cm, s_cm, vh_cm = u[:, :common_s], s[:common_s], vh[:common_s, :]
        u_ts_list, s_ts_list, vh_ts_list = [], [], []
        for mat in act_d:
            delta_hat = mat - u_cm @ (u_cm.T @ mat)
            u_h, s_h, vh_h = torch.linalg.svd(delta_hat, full_matrices=False)
            u_ts_list.append(u_h[:, :n_dims_per_task])
            s_ts_list.append(s_h[:n_dims_per_task])
            vh_ts_list.append(vh_h[:n_dims_per_task, :])

        u_mixed = torch.cat([*u_ts_list, u_cm], dim=1)
        s_mixed = torch.cat([*s_ts_list, s_cm], dim=0)
        vh_mixed = torch.cat([*vh_ts_list, vh_cm], dim=0)

        u_u, _, vh_u = torch.linalg.svd(u_mixed, full_matrices=False)
        u_v, _, vh_v = torch.linalg.svd(vh_mixed, full_matrices=False)
        u_orth = u_u @ vh_u
        vh_orth = u_v @ vh_v
        s_bar = float(s_mixed.mean().item())
        return ((u_orth * s_bar) @ vh_orth).float().cpu()


def _knots_matrix(
    mats: list[torch.Tensor],
    ties_fn: Callable[[torch.Tensor, float], torch.Tensor],
    svd_tol: float = 1e-5,
    top_k: float = 0.20,
    scale: float = 0.6,
) -> torch.Tensor:
    """Verbatim KnOTS-TIES (ICLR 2025): joint horizontal SVD alignment + TIES on shared subspace coefficients."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev = _svd_dev()
    n = len(active)
    _, c_dim = active[0].shape
    with torch.no_grad():
        m_cat = torch.cat([m.double().to(dev) for m in active], dim=1)
        u64, s64, vh64 = torch.linalg.svd(m_cat, full_matrices=False)
        keep = s64 > float(svd_tol)
        if int(keep.sum().item()) == 0:
            return torch.zeros_like(active[0])
        u_k = u64[:, keep].float().cpu()
        s_k = s64[keep].float().cpu()
        vh_k = vh64[keep, :].float().cpu()
        sv_tasks = [(s_k.unsqueeze(1) * vh_k[:, i * c_dim : (i + 1) * c_dim]).reshape(-1) for i in range(n)]
        merged_sv = ties_fn(torch.stack(sv_tasks, dim=0), float(top_k)).reshape(int(s_k.numel()), c_dim)
        return (float(scale) * (u_k @ merged_sv)).float().cpu()


def _dc_lora_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim DC-Merge LoRA (CVPR 2026: avg smoothing + cover space + top-0.1% TIES + block diag)."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev = _svd_dev()
    n = len(active)
    min_dim = min(active[0].shape)
    lr = max(1, min(int(rank_per_task), min_dim // n)) if n * int(rank_per_task) > min_dim else int(rank_per_task)
    smoothed_vecs: list[torch.Tensor] = []
    with torch.no_grad():
        sum_u = sum_v = None
        for i, mat in enumerate(active):
            u, s, vh = torch.linalg.svd(mat.to(dev), full_matrices=False)
            if sum_u is None:
                sum_u = torch.zeros_like(u)
                sum_v = torch.zeros_like(vh)
            sum_u[:, i * lr : (i + 1) * lr] = u[:, :lr]
            sum_v[i * lr : (i + 1) * lr, :] = vh[:lr, :]
            orig_e = s[:lr]
            smoothed_e = orig_e.sum() * (torch.ones_like(orig_e) / float(len(orig_e)))
            smoothed_vecs.append((u[:, :lr] * smoothed_e.unsqueeze(0)) @ vh[:lr, :])
        assert sum_u is not None and sum_v is not None
        u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
        cover_u = (u_u @ vh_u)[:, : n * lr]
        cover_vt = (u_v @ vh_v)[: n * lr, :]
        ms = [cover_u.T @ sv @ cover_vt.T for sv in smoothed_vecs]

        filtered = []
        for t in ms:
            flat_abs = t.abs().view(-1)
            k_keep = max(1, int(1e-3 * flat_abs.numel()))
            thresh = torch.topk(flat_abs, k_keep).values.min()
            filtered.append(t * (t.abs() >= thresh))
        stk = torch.stack(filtered, dim=0)
        summed = stk.sum(dim=0)
        mask = torch.sign(stk) == torch.sign(summed).unsqueeze(0)
        agg = (stk * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
        agg = torch.where(summed == 0, torch.zeros_like(agg), agg)

        mask_m = torch.zeros_like(agg)
        for i in range(n):
            mask_m[i * lr : (i + 1) * lr, i * lr : (i + 1) * lr] = 1.0
        return (cover_u @ (agg * mask_m) @ cover_vt).float().cpu()


__all__ = [
    "_dc_lora_matrix",
    "_dop_merge",
    "_dop_merge_pair",
    "_iso_c_matrix",
    "_iso_cts_matrix",
    "_knots_matrix",
    "_opcm_merge",
    "_prep_active",
    "_tsvm_merge_matrix",
    "_wudi_matrix",
    "_wudi_merge",
]
