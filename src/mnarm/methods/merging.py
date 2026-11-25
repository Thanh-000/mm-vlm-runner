"""Task-vector merging operators (Mean, Exposure, TA, TIES, MagMax, TSV-M, ISO-C, OPCM, DOP, DC-LoRA)."""

from __future__ import annotations

import torch

ALL_MERGE_METHODS = (
    "mean",
    "exposure",
    "ta_sum",
    "ties20",
    "magmax",
    "magmax_s1",
    "tsvm",
    "iso_c",
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
        uu, _, uhu = torch.linalg.svd(torch.cat(u_parts, dim=1), full_matrices=False)
        uv, _, uhv = torch.linalg.svd(torch.cat(v_parts, dim=1), full_matrices=False)
        return ((uu @ uhu) * torch.cat(s_parts, dim=0).unsqueeze(0)) @ (uv @ uhv).T


def _opcm_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str], alpha: float = 0.5) -> dict[str, torch.Tensor]:
    """Verbatim-parity OPCM continual SVD projection across stage task deltas."""
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


def _dc_lora_matrix(mats: list[torch.Tensor], rank_per_task: int = 8) -> torch.Tensor:
    """Verbatim-parity DC-Merge LoRA (avg smoothing + cover space + top-0.1% TIES + block diag)."""
    active = [m.float().cpu() for m in mats if float(m.norm().item()) > 1e-12]
    if not active:
        return torch.zeros_like(mats[0].float().cpu())
    if len(active) == 1:
        return active[0].clone()
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
        return cover_u @ (agg * mask_m) @ cover_vt


def _iso_c_matrix(mats: list[torch.Tensor]) -> torch.Tensor:
    """Verbatim-parity ISO-C (ICML 2025): equalize singular value spectrum of summed active deltas."""
    active = [m.float().cpu() for m in mats if float(m.norm().item()) > 1e-12]
    if not active:
        return torch.zeros_like(mats[0].float().cpu())
    if len(active) == 1:
        return active[0].clone()
    combined = torch.stack(active, dim=0).sum(dim=0)
    if combined.ndim != 2:
        return combined / float(len(active))
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(combined, full_matrices=False)
        s_iso = torch.full_like(s, float(s.mean().item()))
        return (u * s_iso.unsqueeze(0)) @ vh


def _dop_merge_pair(
    d0: torch.Tensor,
    d1: torch.Tensor,
    lr: float = 1e-3,
    num_steps: int = 80,
    ema_beta: float = 0.99,
    alpha: float = 0.5,
    svd_eps: float = 0.99,
) -> torch.Tensor:
    """Continual DOP (NeurIPS 2025): dual SVD subspace projections with closed-form 2-task MGDA + Adam."""
    d0_f, d1_f = d0.float().cpu(), d1.float().cpu()
    if float(d1_f.norm().item()) <= 1e-12:
        return d0_f.clone()
    if float(d0_f.norm().item()) <= 1e-12:
        return d1_f.clone()
    if d0_f.ndim != 2:
        return 0.5 * (d0_f + d1_f)

    with torch.no_grad():
        factors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for mat in (d0_f, d1_f):
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            ratio = s.cumsum(dim=0) / s.sum().clamp_min(1e-12)
            k = max(1, min(int(torch.searchsorted(ratio, float(svd_eps)).item()) + 1, int(s.numel())))
            factors.append((u[:, :k], s[:k], vh[:k, :]))

        w = 0.5 * (d0_f + d1_f)
        m_adam = torch.zeros_like(w)
        v_adam = torch.zeros_like(w)
        ema_c = float(alpha)
        (u0, s0, vh0), (u1, s1, vh1) = factors

        for step in range(1, int(num_steps) + 1):
            diff0 = w - d0_f
            p0_u = s0.unsqueeze(1) * (u0.T @ diff0)
            p0_v = (diff0 @ vh0.T) * s0.unsqueeze(0)
            loss0 = float((p0_u.square().sum() + p0_v.square().sum()).item())
            grad0 = 2.0 * (u0 @ (s0.unsqueeze(1) * p0_u) + (p0_v * s0.unsqueeze(0)) @ vh0)

            diff1 = w - d1_f
            p1_u = s1.unsqueeze(1) * (u1.T @ diff1)
            p1_v = (diff1 @ vh1.T) * s1.unsqueeze(0)
            loss1 = float((p1_u.square().sum() + p1_v.square().sum()).item())
            grad1 = 2.0 * (u1 @ (s1.unsqueeze(1) * p1_u) + (p1_v * s1.unsqueeze(0)) @ vh1)

            g0_n = grad0 / max(loss0, 1e-12)
            g1_n = grad1 / max(loss1, 1e-12)
            v11 = float((g0_n * g0_n).sum().item())
            v12 = float((g0_n * g1_n).sum().item())
            v22 = float((g1_n * g1_n).sum().item())
            if v12 >= v11:
                gamma = 0.999
            elif v12 >= v22:
                gamma = 0.001
            else:
                gamma = min(0.999, max(0.001, -1.0 * ((v12 - v22) / max(v11 + v22 - 2.0 * v12, 1e-12))))

            ema_c = float(ema_beta) * ema_c + (1.0 - float(ema_beta)) * gamma
            grad = ema_c * grad0 + (1.0 - ema_c) * grad1
            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * grad.square()
            m_hat = m_adam / (1.0 - (0.9 ** step))
            v_hat = v_adam / (1.0 - (0.999 ** step))
            w = w - float(lr) * m_hat / (v_hat.sqrt() + 1e-8)
        return w


def _dop_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str]) -> dict[str, torch.Tensor]:
    merged = {k: deltas[0][k].float().cpu().clone() for k in keys}
    for nxt in deltas[1:]:
        for k in keys:
            merged[k] = _dop_merge_pair(merged[k], nxt[k])
    return merged


def merge_task_deltas(deltas: list[dict[str, torch.Tensor]], method: str) -> dict[str, torch.Tensor]:
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
    if method == "tsvm":
        return {k: _tsvm_merge_matrix([d[k] for d in deltas]) for k in keys}
    if method == "iso_c":
        return {k: _iso_c_matrix([d[k] for d in deltas]) for k in keys}
    if method == "opcm":
        return _opcm_merge(deltas, keys, alpha=0.5)
    if method == "dop":
        return _dop_merge(deltas, keys)
    if method == "dc_lora":
        return {k: _dc_lora_matrix([d[k] for d in deltas], rank_per_task=8) for k in keys}
    raise ValueError(f"Unknown merge method: {method}")
