"""Iterative and batched LoRA merging operators (WUDI, OPCM, and batched GPU DOP)."""

from __future__ import annotations

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


def _wudi_merge(
    deltas: list[dict[str, torch.Tensor]],
    keys: list[str],
    iter_num: int = 300,
    lr: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Batched verbatim WUDI-Merging grouped by active stages and tensor shape."""
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


def _opcm_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str], alpha: float = 0.5) -> dict[str, torch.Tensor]:
    """Verbatim OPCM (arXiv:2501.09522) continual SVD projection across stage task deltas."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    merged = {k: deltas[0][k].float().to(dev).clone() for k in keys}
    norm0 = float(torch.cat([merged[k].reshape(-1) for k in keys]).norm().item())
    all_norms = [norm0]
    prev_lambda = 1.0
    with torch.no_grad():
        for nxt in deltas[1:]:
            t_vec = {k: nxt[k].float().to(dev) for k in keys}
            all_norms.append(float(torch.cat([t_vec[k].reshape(-1) for k in keys]).norm().item()))
            avg_norm = sum(all_norms) / float(len(all_norms))
            for k in keys:
                prev_m, task_m = merged[k], t_vec[k]
                if float(task_m.norm().item()) <= 1e-12 or float(prev_m.norm().item()) <= 1e-12:
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
    return {k: merged[k].detach().cpu() for k in keys}


def _dop_factor_truncated(mat: torch.Tensor, svd_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    pos = max(1, int((s > 1e-7).sum().item()))
    ratio = s.cumsum(dim=0) / s.sum().clamp_min(1e-12)
    k = max(1, min(int(torch.searchsorted(ratio, float(svd_eps)).item()) + 1, pos, int(s.numel())))
    return u[:, :k], s[:k], vh[:k, :]


def _dop_pad_factors(
    facs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    d_out: int,
    d_in: int,
    dev: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_max = max(f[1].numel() for f in facs)
    b = len(facs)
    u_b = torch.zeros((b, d_out, k_max), dtype=dtype, device=dev)
    s_b = torch.zeros((b, k_max), dtype=dtype, device=dev)
    vh_b = torch.zeros((b, k_max, d_in), dtype=dtype, device=dev)
    for i, (ui, si, vhi) in enumerate(facs):
        k_i = si.numel()
        u_b[i, :, :k_i] = ui
        s_b[i, :k_i] = si
        vh_b[i, :k_i, :] = vhi
    return u_b, s_b, vh_b


def _dop_merge_pair_batch(
    mats0: list[torch.Tensor],
    mats1: list[torch.Tensor],
    lr: float = 1e-4,
    num_steps: int = 200,
    ema_beta: float = 0.999,
    alpha: float = 0.8,
    svd_eps: float = 0.99999,
) -> torch.Tensor:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        d0_d = torch.stack([m.detach().float() for m in mats0], dim=0).to(dev)
        d1_d = torch.stack([m.detach().float() for m in mats1], dim=0).to(dev)
        b_sz, d_out, d_in = d0_d.shape
        fac0 = [_dop_factor_truncated(d0_d[i], svd_eps) for i in range(b_sz)]
        fac1 = [_dop_factor_truncated(d1_d[i], svd_eps) for i in range(b_sz)]
        u0, s0, vh0 = _dop_pad_factors(fac0, d_out, d_in, dev, d0_d.dtype)
        u1, s1, vh1 = _dop_pad_factors(fac1, d_out, d_in, dev, d0_d.dtype)
        u0t, vh0t, u1t, vh1t = u0.transpose(1, 2), vh0.transpose(1, 2), u1.transpose(1, 2), vh1.transpose(1, 2)

        w = 0.5 * (d0_d + d1_d)
        m_adam, v_adam = torch.zeros_like(w), torch.zeros_like(w)
        ema_c = torch.full((b_sz, 1, 1), float(alpha), dtype=w.dtype, device=dev)
        for step in range(1, int(num_steps) + 1):
            diff0 = w - d0_d
            p0_u = s0.unsqueeze(-1) * torch.bmm(u0t, diff0)
            p0_v = torch.bmm(diff0, vh0t) * s0.unsqueeze(1)
            loss0 = (p0_u.square().sum(dim=(1, 2)) + p0_v.square().sum(dim=(1, 2))).clamp_min(1e-12)
            grad0 = 2.0 * (torch.bmm(u0, s0.unsqueeze(-1) * p0_u) + torch.bmm(p0_v * s0.unsqueeze(1), vh0))

            diff1 = w - d1_d
            p1_u = s1.unsqueeze(-1) * torch.bmm(u1t, diff1)
            p1_v = torch.bmm(diff1, vh1t) * s1.unsqueeze(1)
            loss1 = (p1_u.square().sum(dim=(1, 2)) + p1_v.square().sum(dim=(1, 2))).clamp_min(1e-12)
            grad1 = 2.0 * (torch.bmm(u1, s1.unsqueeze(-1) * p1_u) + torch.bmm(p1_v * s1.unsqueeze(1), vh1))

            g0_n = grad0 / loss0.view(b_sz, 1, 1)
            g1_n = grad1 / loss1.view(b_sz, 1, 1)
            v11 = (g0_n * g0_n).sum(dim=(1, 2))
            v12 = (g0_n * g1_n).sum(dim=(1, 2))
            v22 = (g1_n * g1_n).sum(dim=(1, 2))
            gamma_mid = (-1.0 * ((v12 - v22) / (v11 + v22 - 2.0 * v12).clamp_min(1e-12))).clamp(0.001, 0.999)
            gamma = torch.where(v12 >= v11, 0.999, torch.where(v12 >= v22, 0.001, gamma_mid)).view(b_sz, 1, 1)

            ema_c = float(ema_beta) * ema_c + (1.0 - float(ema_beta)) * gamma
            grad = ema_c * grad0 + (1.0 - ema_c) * grad1
            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * grad.square()
            w = w - float(lr) * (m_adam / (1.0 - (0.9 ** step))) / ((v_adam / (1.0 - (0.999 ** step))).sqrt() + 1e-8)
        return w.detach().cpu()


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
    return _dop_merge_pair_batch([d0_f], [d1_f], lr=lr, num_steps=num_steps, ema_beta=ema_beta, alpha=alpha, svd_eps=svd_eps)[0]


def _dop_merge(deltas: list[dict[str, torch.Tensor]], keys: list[str]) -> dict[str, torch.Tensor]:
    """Batched GPU/CPU DOP merging across all layers per stage transition."""
    merged = {k: deltas[0][k].float().cpu().clone() for k in keys}
    for nxt in deltas[1:]:
        groups: dict[tuple[int, ...], list[str]] = {}
        for k in keys:
            d0, d1 = merged[k], nxt[k].detach().float().cpu()
            if float(d1.norm().item()) <= 1e-12:
                continue
            if float(d0.norm().item()) <= 1e-12:
                merged[k] = d1.clone()
            elif d0.ndim != 2:
                merged[k] = 0.5 * (d0 + d1)
            else:
                groups.setdefault(tuple(int(s) for s in d0.shape), []).append(k)
        for gkeys in groups.values():
            res = _dop_merge_pair_batch([merged[k] for k in gkeys], [nxt[k] for k in gkeys])
            for idx, k in enumerate(gkeys):
                merged[k] = res[idx].clone()
    return merged
