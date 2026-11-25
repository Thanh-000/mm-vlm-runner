"""Subspace & iterative LoRA model merging operators (KnOTS, TSV-M, ISO-C, ISO-CTS, WUDI, OPCM, DOP, DC-LoRA)."""

from __future__ import annotations

from typing import Callable

import torch


def _prep_active(mats: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    active = [m.detach().float().cpu().clone() for m in mats if float(m.norm().item()) > 1e-12]
    if not active:
        return [], torch.zeros_like(mats[0].float().cpu())
    if len(active) == 1:
        return [], active[0].clone()
    if active[0].ndim != 2:
        return [], torch.stack(active, dim=0).mean(dim=0)
    return active, None


def _tsvm_merge_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim TSV-M on LoRA deltas (DC-Merge/TSV-M: truncate each task to active LoRA rank)."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    n = len(active)
    min_dim = min(active[0].shape)
    r_lora = max(1, int(rank_per_task))
    rk = max(1, min(r_lora, min_dim // n)) if n * r_lora > min_dim else max(1, min(r_lora, min_dim))
    u_parts, s_parts, v_parts = [], [], []
    with torch.no_grad():
        for mat in active:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
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
    combined = torch.stack(active, dim=0).sum(dim=0)
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
    n = len(active)
    delta_ta = torch.stack(active, dim=0).sum(dim=0)
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
        for mat in active:
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
    n = len(active)
    _, c_dim = active[0].shape
    with torch.no_grad():
        m_cat = torch.cat([m.double() for m in active], dim=1)
        u64, s64, vh64 = torch.linalg.svd(m_cat, full_matrices=False)
        keep = s64 > float(svd_tol)
        if int(keep.sum().item()) == 0:
            return torch.zeros_like(active[0])
        u_k = u64[:, keep].float()
        s_k = s64[keep].float()
        vh_k = vh64[keep, :].float()
        sv_tasks = [(s_k.unsqueeze(1) * vh_k[:, i * c_dim : (i + 1) * c_dim]).reshape(-1) for i in range(n)]
        merged_sv = ties_fn(torch.stack(sv_tasks, dim=0), float(top_k)).reshape(int(s_k.numel()), c_dim)
        return (float(scale) * (u_k @ merged_sv)).float().cpu()


def _wudi_merge(
    deltas: list[dict[str, torch.Tensor]],
    keys: list[str],
    iter_num: int = 300,
    lr: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Batched verbatim WUDI-Merging grouped by active stages & tensor shape."""
    out: dict[str, torch.Tensor] = {}
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {}
    for k in keys:
        mats = [d[k] for d in deltas]
        sig = tuple(i for i, m in enumerate(mats) if float(m.norm().item()) > 1e-12)
        if len(sig) <= 1 or mats[0].ndim != 2:
            _, fallback = _prep_active(mats)
            assert fallback is not None
            out[k] = fallback
        else:
            groups.setdefault((sig, tuple(int(s) for s in mats[0].shape)), []).append(k)
    if not groups:
        return out
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode(False), torch.enable_grad():
        for (sig, _), gkeys in groups.items():
            vectors = torch.stack(
                [torch.stack([deltas[i][k].detach().float().cpu().clone() for i in sig], dim=0) for k in gkeys],
                dim=0,
            ).to(device=dev, dtype=torch.float32)
            b_sz, t_act = vectors.shape[0], vectors.shape[1]
            merging_vector = torch.nn.Parameter(vectors.sum(dim=1))
            opt = torch.optim.Adam([merging_vector], lr=float(lr), weight_decay=0.0)
            l2_norms = torch.square(torch.norm(vectors.reshape(b_sz, t_act, -1), p=2, dim=-1)).clamp_min(1e-24)
            for _ in range(int(iter_num)):
                disturbing = merging_vector.unsqueeze(1) - vectors
                inner = torch.matmul(disturbing, vectors.transpose(-1, -2))
                loss = torch.sum(torch.square(inner) / l2_norms.unsqueeze(-1).unsqueeze(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            res = merging_vector.detach().cpu()
            for idx, k in enumerate(gkeys):
                out[k] = res[idx].clone()
    return out


def _wudi_matrix(mats: list[torch.Tensor], iter_num: int = 300, lr: float = 1e-5) -> torch.Tensor:
    """Single-layer verbatim WUDI-Merging wrapper delegating to batched _wudi_merge."""
    return _wudi_merge([{"k": m} for m in mats], ["k"], iter_num=iter_num, lr=lr)["k"]


def _dc_lora_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim DC-Merge LoRA (CVPR 2026: avg smoothing + cover space + top-0.1% TIES + block diag)."""
    active, fallback = _prep_active(mats)
    if fallback is not None:
        return fallback
    n = len(active)
    min_dim = min(active[0].shape)
    lr = max(1, min(int(rank_per_task), min_dim // n)) if n * int(rank_per_task) > min_dim else int(rank_per_task)
    smoothed_vecs: list[torch.Tensor] = []
    with torch.no_grad():
        sum_u = sum_v = None
        for i, mat in enumerate(active):
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
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


def _opcm_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str], alpha: float = 0.5) -> dict[str, torch.Tensor]:
    """Verbatim OPCM (arXiv:2501.09522) continual SVD projection across stage task deltas."""
    merged = {k: deltas[0][k].float().cpu().clone() for k in keys}
    norm0 = float(torch.cat([merged[k].reshape(-1) for k in keys]).norm().item())
    all_norms = [norm0]
    prev_lambda = 1.0
    with torch.no_grad():
        for nxt in deltas[1:]:
            t_vec = {k: nxt[k].float().cpu() for k in keys}
            all_norms.append(float(torch.cat([t_vec[k].reshape(-1) for k in keys]).norm().item()))
            avg_norm = sum(all_norms) / float(len(all_norms))
            for k in keys:
                prev_m, task_m = merged[k], t_vec[k]
                if float(task_m.norm().item()) <= 1e-12:
                    cleaned = task_m
                elif float(prev_m.norm().item()) <= 1e-12:
                    cleaned = task_m
                elif prev_m.ndim == 2:
                    u, s, vh = torch.linalg.svd(prev_m, full_matrices=False)
                    s_sum = s.sum().clamp_min(1e-12)
                    split_rank = int((s.cumsum(dim=0) / s_sum > float(alpha)).float().argmax().item())
                    proj = u.T @ task_m @ vh.T
                    proj.diag().fill_(0.0)
                    proj[:split_rank, :split_rank] = 0.0
                    cleaned = u @ proj @ vh
                else:
                    cleaned = task_m
                merged[k] = prev_lambda * prev_m + cleaned
            cur_norm = float(torch.cat([merged[k].reshape(-1) for k in keys]).norm().item())
            if cur_norm > 1e-12 and avg_norm > 1e-12:
                prev_lambda = cur_norm / avg_norm
                scale = avg_norm / cur_norm
                for k in keys:
                    merged[k] = merged[k] * scale
            else:
                prev_lambda = 1.0
    return merged


def _dop_merge_pair(
    d0: torch.Tensor,
    d1: torch.Tensor,
    lr: float = 1e-4,
    num_steps: int = 200,
    ema_beta: float = 0.999,
    alpha: float = 0.8,
    svd_eps: float = 0.99999,
) -> torch.Tensor:
    """Verbatim DOP (NeurIPS 2025, EnnengYang/DOP dop.yaml): dual SVD projections + 2-task MGDA."""
    d0_f, d1_f = d0.detach().float().cpu(), d1.detach().float().cpu()
    if float(d1_f.norm().item()) <= 1e-12:
        return d0_f.clone()
    if float(d0_f.norm().item()) <= 1e-12:
        return d1_f.clone()
    if d0_f.ndim != 2:
        return 0.5 * (d0_f + d1_f)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        d0_d, d1_d = d0_f.to(dev), d1_f.to(dev)
        factors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for mat in (d0_d, d1_d):
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            pos = max(1, int((s > 1e-7).sum().item()))
            ratio = s.cumsum(dim=0) / s.sum().clamp_min(1e-12)
            k = max(1, min(int(torch.searchsorted(ratio, float(svd_eps)).item()) + 1, pos, int(s.numel())))
            factors.append((u[:, :k], s[:k], vh[:k, :]))

        w = 0.5 * (d0_d + d1_d)
        m_adam = torch.zeros_like(w)
        v_adam = torch.zeros_like(w)
        ema_c = torch.tensor(float(alpha), dtype=w.dtype, device=dev)
        (u0, s0, vh0), (u1, s1, vh1) = factors

        for step in range(1, int(num_steps) + 1):
            diff0 = w - d0_d
            p0_u = s0.unsqueeze(1) * (u0.T @ diff0)
            p0_v = (diff0 @ vh0.T) * s0.unsqueeze(0)
            loss0 = (p0_u.square().sum() + p0_v.square().sum()).clamp_min(1e-12)
            grad0 = 2.0 * (u0 @ (s0.unsqueeze(1) * p0_u) + (p0_v * s0.unsqueeze(0)) @ vh0)

            diff1 = w - d1_d
            p1_u = s1.unsqueeze(1) * (u1.T @ diff1)
            p1_v = (diff1 @ vh1.T) * s1.unsqueeze(0)
            loss1 = (p1_u.square().sum() + p1_v.square().sum()).clamp_min(1e-12)
            grad1 = 2.0 * (u1 @ (s1.unsqueeze(1) * p1_u) + (p1_v * s1.unsqueeze(0)) @ vh1)

            g0_n = grad0 / loss0
            g1_n = grad1 / loss1
            v11 = (g0_n * g0_n).sum()
            v12 = (g0_n * g1_n).sum()
            v22 = (g1_n * g1_n).sum()
            gamma_mid = (-1.0 * ((v12 - v22) / (v11 + v22 - 2.0 * v12).clamp_min(1e-12))).clamp(0.001, 0.999)
            gamma = torch.where(
                v12 >= v11,
                torch.full_like(v12, 0.999),
                torch.where(v12 >= v22, torch.full_like(v12, 0.001), gamma_mid),
            )

            ema_c = float(ema_beta) * ema_c + (1.0 - float(ema_beta)) * gamma
            grad = ema_c * grad0 + (1.0 - ema_c) * grad1
            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * grad.square()
            m_hat = m_adam / (1.0 - (0.9 ** step))
            v_hat = v_adam / (1.0 - (0.999 ** step))
            w = w - float(lr) * m_hat / (v_hat.sqrt() + 1e-8)
        return w.detach().cpu()


def _dop_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str]) -> dict[str, torch.Tensor]:
    merged = {k: deltas[0][k].float().cpu().clone() for k in keys}
    for nxt in deltas[1:]:
        for k in keys:
            merged[k] = _dop_merge_pair(merged[k], nxt[k])
    return merged
