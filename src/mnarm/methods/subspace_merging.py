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


def _tower_group(key: str) -> str:
    if key == "t" or key.startswith(("text_model.", "t.")):
        return "text"
    if key == "v" or key.startswith(("vision_model.", "v.")):
        return "vision"
    return "shared"


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


def _norm_cap(out: torch.Tensor, active: list[torch.Tensor], only_overshoot: bool = True) -> torch.Tensor:
    avg_norm = sum(float(m.norm().item()) for m in active) / float(len(active))
    cur_norm = float(out.norm().item())
    thresh = max(avg_norm, 1e-12) if only_overshoot else 1e-12
    if cur_norm > thresh and avg_norm > 1e-12:
        out = out * (avg_norm / cur_norm)
    return out.float().cpu()


def _iso_c_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim ISO-C (ICML 2025) for low-rank LoRA with overshoot energy cap."""
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
        return _norm_cap((u[:, :r_act] * s_bar) @ vh[:r_act, :], active, only_overshoot=True)


def _iso_cts_matrix(mats: list[torch.Tensor], rank_per_task: int = 8, k_frac: float = 0.8) -> torch.Tensor:
    """Verbatim ISO-CTS (ICML 2025 & DC-Merge LoRA) with overshoot energy cap."""
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

        u_u, _, vh_u = torch.linalg.svd(torch.cat([*u_ts_list, u_cm], dim=1), full_matrices=False)
        u_v, _, vh_v = torch.linalg.svd(torch.cat([*vh_ts_list, vh_cm], dim=0), full_matrices=False)
        s_bar = float(torch.cat([*s_ts_list, s_cm], dim=0).mean().item())
        return _norm_cap(((u_u @ vh_u) * s_bar) @ (u_v @ vh_v), active, only_overshoot=True)


def _knots_merge(
    deltas: list[dict[str, torch.Tensor]], keys: list[str], ties_fn: Callable[[torch.Tensor, float], torch.Tensor],
    svd_tol: float = 1e-5, top_k: float = 0.20, scale: float = 0.6,
) -> dict[str, torch.Tensor]:
    """Verbatim KnOTS-TIES (ICLR 2025): joint SVD alignment + global cross-layer TIES per active tower."""
    dev, out = _svd_dev(), {}
    groups: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for k in keys:
        mats = [d[k] for d in deltas]
        sig = tuple(i for i, m in enumerate(mats) if float(m.norm().item()) > 1e-12)
        if len(sig) <= 1 or mats[0].ndim != 2:
            _, fallback = _prep_active(mats)
            assert fallback is not None
            out[k] = fallback
        else:
            groups.setdefault((_tower_group(k), sig), []).append(k)
    with torch.no_grad():
        for (_, sig), gkeys in groups.items():
            items: list[tuple[str, torch.Tensor, int, int]] = []
            task_vecs: list[list[torch.Tensor]] = [[] for _ in sig]
            sizes: list[int] = []
            for k in gkeys:
                c_dim = int(deltas[0][k].shape[1])
                m_cat = torch.cat([deltas[i][k].double().to(dev) for i in sig], dim=1)
                u64, s64, vh64 = torch.linalg.svd(m_cat, full_matrices=False)
                keep = s64 > float(svd_tol)
                r_k = int(keep.sum().item())
                if r_k == 0:
                    out[k] = torch.zeros_like(deltas[0][k].float().cpu())
                    continue
                u_k, s_k, vh_k = u64[:, keep].float().cpu(), s64[keep].float().cpu(), vh64[keep, :].float().cpu()
                items.append((k, u_k, r_k, c_dim))
                sizes.append(r_k * c_dim)
                for j in range(len(sig)):
                    task_vecs[j].append((s_k.unsqueeze(1) * vh_k[:, j * c_dim : (j + 1) * c_dim]).reshape(-1))
            if not items:
                continue
            global_sv = torch.stack([torch.cat(tv, dim=0) for tv in task_vecs], dim=0)
            merged_flat = ties_fn(global_sv, float(top_k))
            for (k, u_k, r_k, c_dim), chunk in zip(items, merged_flat.split(sizes)):
                out[k] = (float(scale) * (u_k @ chunk.reshape(r_k, c_dim))).float().cpu()
    return out


def _knots_matrix(
    mats: list[torch.Tensor], ties_fn: Callable[[torch.Tensor, float], torch.Tensor],
    svd_tol: float = 1e-5, top_k: float = 0.20, scale: float = 0.6,
) -> torch.Tensor:
    """Single-layer KnOTS-TIES wrapper delegating to _knots_merge."""
    return _knots_merge([{"k": m} for m in mats], ["k"], ties_fn, svd_tol=svd_tol, top_k=top_k, scale=scale)["k"]


def _dc_lora_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim DC-Merge LoRA (CVPR 2026) with compact active subspace, rank-floor TIES, and energy norm."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    dev, n, min_dim = _svd_dev(), len(active), min(active[0].shape)
    lr = max(1, min(int(rank_per_task), min_dim // n)) if n * int(rank_per_task) > min_dim else int(rank_per_task)
    u_list, v_list, smoothed_vecs = [], [], []
    with torch.no_grad():
        for mat in active:
            u, s, vh = torch.linalg.svd(mat.to(dev), full_matrices=False)
            u_list.append(u[:, :lr])
            v_list.append(vh[:lr, :])
            smoothed_e = s[:lr].mean().expand_as(s[:lr])
            smoothed_vecs.append((u[:, :lr] * smoothed_e.unsqueeze(0)) @ vh[:lr, :])
        u_u, _, vh_u = torch.linalg.svd(torch.cat(u_list, dim=1), full_matrices=False)
        u_v, _, vh_v = torch.linalg.svd(torch.cat(v_list, dim=0), full_matrices=False)
        cover_u, cover_vt = u_u @ vh_u, u_v @ vh_v
        ms = [cover_u.T @ sv @ cover_vt.T for sv in smoothed_vecs]

        filtered = []
        for t in ms:
            flat_abs = t.abs().view(-1)
            k_keep = min(int(flat_abs.numel()), max(lr, int(1e-3 * flat_abs.numel())))
            thresh = torch.topk(flat_abs, k_keep).values.min()
            filtered.append(t * (t.abs() >= thresh))
        stk = torch.stack(filtered, dim=0)
        summed = stk.sum(dim=0)
        mask = torch.sign(stk) == torch.sign(summed).unsqueeze(0)
        agg = torch.where(summed == 0, torch.zeros_like(summed), (stk * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1))

        mask_m = torch.zeros_like(agg)
        for i in range(n):
            mask_m[i * lr : (i + 1) * lr, i * lr : (i + 1) * lr] = 1.0
        return _norm_cap(cover_u @ (agg * mask_m) @ cover_vt, active, only_overshoot=False)


__all__ = [
    "_dc_lora_matrix", "_dop_merge", "_dop_merge_pair", "_iso_c_matrix", "_iso_cts_matrix",
    "_knots_matrix", "_knots_merge", "_opcm_merge", "_prep_active", "_tower_group",
    "_tsvm_merge_matrix", "_wudi_matrix", "_wudi_merge",
]
