"""Spherical sufficient statistics, O(d^2) cross-moments, and conjugate Empirical-Bayes estimators."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from mnarm.models.vlm import l2_normalize


def _shrink_sample_cov(sample_cov: torch.Tensor, n_samples: int, dim: int, shrinkage: float | None = None, jitter: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply closed-form OAS shrinkage to a within-class sample covariance and invert via Cholesky."""
    eye = torch.eye(dim, dtype=torch.float64)
    tr_s = float(torch.trace(sample_cov).item())
    if tr_s <= 0.0 or n_samples <= 1:
        return eye, eye, 1.0
    mu_iso = max(tr_s / float(dim), 1e-8)

    if shrinkage is None:
        tr_s2 = float(sample_cov.square().sum().item())
        num = (1.0 - 2.0 / float(dim)) * tr_s2 + tr_s * tr_s
        den = (float(n_samples) + 1.0 - 2.0 / float(dim)) * (tr_s2 - (tr_s * tr_s) / float(dim))
        rho = 1.0 if den <= 1e-30 else min(1.0, max(0.0, num / den))
    else:
        rho = min(1.0, max(0.0, float(shrinkage)))

    cov = (1.0 - rho) * sample_cov + rho * mu_iso * eye + float(jitter) * eye
    return cov, torch.cholesky_inverse(torch.linalg.cholesky(cov)), float(rho)


def _oas_shrinkage_covariance(features: torch.Tensor, labels: torch.Tensor, obs_classes: list[int], shrinkage: float | None = None, jitter: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute pooled within-class covariance with closed-form OAS shrinkage."""
    u = features.to(dtype=torch.float64, device="cpu")
    n, d = u.shape
    eye = torch.eye(d, dtype=torch.float64)
    resid_parts = [u[labels == int(c)] - u[labels == int(c)].mean(dim=0, keepdim=True) for c in obs_classes if bool((labels == int(c)).any())]
    if not resid_parts or n <= len(obs_classes):
        return eye, eye, 1.0
    resid = torch.cat(resid_parts, dim=0)
    sample_cov = (resid.T @ resid) / float(max(n - len(obs_classes), 1))
    return _shrink_sample_cov(sample_cov, n_samples=n, dim=d, shrinkage=shrinkage, jitter=jitter)


def _identity_anchored_procrustes(src: torch.Tensor, dst: torch.Tensor, anchor_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit scale-aware mean-shifted orthogonal Procrustes R mapping centered src -> dst."""
    d = src.shape[-1]
    src64, dst64 = src.to(dtype=torch.float64, device="cpu"), dst.to(dtype=torch.float64, device="cpu")
    src_mean, dst_mean = src64.mean(dim=0), dst64.mean(dim=0)
    src_c, dst_c = src64 - src_mean.unsqueeze(0), dst64 - dst_mean.unsqueeze(0)
    if src64.shape[0] < 2 or float(src_c.norm().item()) < 1e-8 or float(dst_c.norm().item()) < 1e-8:
        return torch.eye(d, dtype=torch.float64), src_mean, dst_mean
    eff_anchor = float(anchor_weight) * max(float(src64.shape[0]) / float(d), 1e-3)
    u_mat, _, vh_mat = torch.linalg.svd(eff_anchor * torch.eye(d, dtype=torch.float64) + (dst_c.T @ src_c), full_matrices=False)
    return u_mat @ vh_mat, src_mean, dst_mean


def _procrustes_from_cross_moments(
    n_pairs: int,
    sum_src: torch.Tensor,
    sum_dst: torch.Tensor,
    cross_dst_src: torch.Tensor,
    anchor_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact O(d^2) sufficient-statistics orthogonal Procrustes with zero stored samples."""
    d = int(sum_src.numel())
    n_f = float(max(int(n_pairs), 1))
    src_mean = sum_src.to(dtype=torch.float64, device="cpu") / n_f
    dst_mean = sum_dst.to(dtype=torch.float64, device="cpu") / n_f
    eye = torch.eye(d, dtype=torch.float64)
    if n_pairs < 2:
        return eye, src_mean, dst_mean
    cov = cross_dst_src.to(dtype=torch.float64, device="cpu") - n_f * torch.outer(dst_mean, src_mean)
    if float(cov.norm().item()) < 1e-10:
        return eye, src_mean, dst_mean
    eff_anchor = float(anchor_weight) * max(n_f / float(d), 1e-3)
    u_mat, _, vh_mat = torch.linalg.svd(eff_anchor * eye + cov, full_matrices=False)
    return u_mat @ vh_mat, src_mean, dst_mean


def _safe_precision_from_cov(cov: torch.Tensor, is_partial: bool, partial_cov_mode: str, sigma2_trans: float = 0.0) -> torch.Tensor:
    """Prevent precision nullspace explosion on partially observed towers (S_m != Y)."""
    d = cov.shape[0]
    if not is_partial or partial_cov_mode == "raw_oas":
        return torch.cholesky_inverse(torch.linalg.cholesky(cov))
    if partial_cov_mode == "isotropic_vmf":
        return torch.eye(d, dtype=torch.float64)
    tr = max(float(torch.trace(cov).item()), 1e-12)
    lam_eff = max(float(cov.square().sum().item()) / tr, 1e-6)
    if partial_cov_mode == "marginal_epistemic":
        iso_var = max(float(sigma2_trans), lam_eff)
        cov_marg = cov + iso_var * torch.eye(d, dtype=torch.float64)
        return torch.cholesky_inverse(torch.linalg.cholesky(cov_marg))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    lam_safe = eigvals.clamp_min(lam_eff)
    return (eigvecs / lam_safe.unsqueeze(0)) @ eigvecs.T


def _equalize_mahalanobis_centroids(mu_stack: torch.Tensor, prec: torch.Tensor, obs_indices: list[int]) -> torch.Tensor:
    """Equalize quadratic Mahalanobis norms so missing classes incur zero bias penalty."""
    m_norms = (mu_stack * (mu_stack @ prec)).sum(dim=-1).clamp_min(1e-12).sqrt()
    target_m = m_norms[obs_indices].mean() if obs_indices else m_norms.mean()
    return mu_stack * (target_m / m_norms).unsqueeze(-1)


def _scale_to_norm(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    return (vec / vec.norm().clamp_min(1e-12)) * float(target_norm)


def _stratified_folds(labels: list[int], k_folds: int) -> list[list[int]]:
    by_cls: dict[int, list[int]] = defaultdict(list)
    for idx, yv in enumerate(labels):
        by_cls[int(yv)].append(idx)
    folds: list[list[int]] = [[] for _ in range(k_folds)]
    for cls in sorted(by_cls):
        for pos, row_idx in enumerate(by_cls[cls]):
            folds[pos % k_folds].append(row_idx)
    return [sorted(f) for f in folds]


def _fit_oof_tau(oof_scores: torch.Tensor, oof_y_local: torch.Tensor, num_obs: int, balanced: bool = True) -> float:
    """Fit 1 scalar log-temperature tau > 0 in exact nat-scale via OOF NLL (class-balanced or empirical)."""
    if oof_scores.numel() == 0 or num_obs < 2:
        return 1.0
    if balanced:
        counts = torch.bincount(oof_y_local, minlength=num_obs).to(torch.float64).clamp_min(1.0)
        sw = 1.0 / counts[oof_y_local]
        sw = sw / sw.sum().clamp_min(1e-12)
    else:
        sw = torch.full((oof_y_local.shape[0],), 1.0 / float(max(oof_y_local.shape[0], 1)), dtype=torch.float64)
    z64 = oof_scores.to(torch.float64)
    spread = float((z64.max(dim=-1).values - z64.median(dim=-1).values).mean().clamp_min(1e-4).item())
    init_tau = min(max(2.0 / spread, 1e-4), 5.0)
    log_tau = torch.tensor(math.log(init_tau), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_tau], lr=1.0, max_iter=60, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tau = torch.exp(torch.clamp(log_tau, -12.0, 6.0))
        loss = (sw * F.cross_entropy(tau * z64, oof_y_local, reduction="none")).sum() + 1e-3 * (log_tau - math.log(init_tau)).square()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(torch.clamp(log_tau.detach(), -12.0, 6.0)).item())


def _fit_direct_oof_tau(
    u: torch.Tensor,
    y: torch.Tensor,
    obs: list[int],
    shrinkage: float | None,
    is_partial: bool,
    partial_cov_mode: str,
    sigma2_trans: float,
    calib_folds: int,
) -> float:
    """Estimate OOF temperature tau_m using exact inference precision (with sigma2_trans)."""
    counts = [int((y == c).sum().item()) for c in obs]
    if len(obs) < 2 or not counts or min(counts) < 2:
        return 1.0
    folds = _stratified_folds([int(v) for v in y.tolist()], min(calib_folds, min(counts)))
    obs_map = {c: i for i, c in enumerate(obs)}
    oof_scores_list, oof_y_list, all_idx, obs_rng = [], [], set(range(int(u.shape[0]))), list(range(len(obs)))
    for f_idx in folds:
        tr_idx = sorted(all_idx - set(f_idx))
        u_tr, y_tr = u[tr_idx], y[tr_idx]
        cov_f, _, _ = _oas_shrinkage_covariance(u_tr, y_tr, obs, shrinkage=shrinkage)
        prec_f = _safe_precision_from_cov(cov_f, is_partial, partial_cov_mode, sigma2_trans=sigma2_trans)
        mu_mat = torch.stack([u_tr[y_tr == c].mean(dim=0) for c in obs], dim=0)
        if is_partial and partial_cov_mode != "raw_oas":
            mu_mat = _equalize_mahalanobis_centroids(mu_mat, prec_f, obs_rng)
        w_f = mu_mat @ prec_f
        oof_scores_list.append(u[f_idx] @ w_f.T + (-0.5 * (mu_mat * w_f).sum(dim=-1)).unsqueeze(0))
        oof_y_list.append(torch.tensor([obs_map[int(v)] for v in y[f_idx].tolist()], dtype=torch.long))
    return _fit_oof_tau(torch.cat(oof_scores_list, dim=0), torch.cat(oof_y_list, dim=0), len(obs))


def _fit_stat_oof_tau(
    centroids: dict[int, torch.Tensor],
    counts: dict[int, int],
    raw_sample_cov: torch.Tensor | None,
    prec: torch.Tensor,
    is_partial: bool,
    partial_cov_mode: str,
    calib_folds: int = 4,
    seed: int = 1729,
) -> float:
    """Calibrate nat-scale tau_m purely from canonical sufficient statistics (0 exemplars, 0 Stage-4 probes)."""
    obs = sorted(centroids)
    k_cls = len(obs)
    if k_cls < 2:
        return 1.0
    d = int(centroids[obs[0]].numel())
    n_folds = max(2, int(calib_folds))
    g = torch.Generator().manual_seed(int(seed) + 37 * k_cls + d)
    if raw_sample_cov is not None:
        eig_v, eig_u = torch.linalg.eigh(raw_sample_cov.to(dtype=torch.float64, device="cpu"))
        sqrt_cov = eig_u * eig_v.clamp_min(0.0).sqrt().unsqueeze(0)
    else:
        r_bar = sum(float(centroids[c].norm().item()) for c in obs) / float(k_cls)
        iso_std = math.sqrt(max(1.0 - r_bar * r_bar, 1e-4) / float(d))
        sqrt_cov = iso_std * torch.eye(d, dtype=torch.float64)

    tot_n = max(sum(max(int(counts.get(c, 1)), 1) for c in obs), k_cls)
    target_m = min(max(tot_n, 48 * k_cls), 960)
    m_per = {c: max(n_folds, int(round(target_m * max(int(counts.get(c, 1)), 1) / float(tot_n)))) for c in obs}
    oof_scores, oof_y, obs_rng = [], [], list(range(k_cls))
    for _ in range(n_folds):
        mu_fold = []
        for c in obs:
            n_c_tr = max(float(counts.get(c, 10)) * float(n_folds - 1) / float(n_folds), 1.0)
            eps_mu = (torch.randn(d, dtype=torch.float64, generator=g) @ sqrt_cov.T) / math.sqrt(n_c_tr)
            mu_fold.append(centroids[c].to(dtype=torch.float64, device="cpu") + eps_mu)
        mu_f = torch.stack(mu_fold, dim=0)
        if is_partial and partial_cov_mode != "raw_oas":
            mu_f = _equalize_mahalanobis_centroids(mu_f, prec, obs_rng)
        w_f = mu_f @ prec
        b_f = -0.5 * (mu_f * w_f).sum(dim=-1)
        xs_f, ys_f = [], []
        for idx_c, c in enumerate(obs):
            mc = max(1, m_per[c] // n_folds)
            z = torch.randn(mc, d, dtype=torch.float64, generator=g) @ sqrt_cov.T
            xs_f.append(l2_normalize((centroids[c].to(dtype=torch.float64, device="cpu").unsqueeze(0) + z).float()).to(torch.float64))
            ys_f.append(torch.full((mc,), idx_c, dtype=torch.long))
        xf_cat = torch.cat(xs_f, dim=0)
        oof_scores.append(xf_cat @ w_f.T + b_f.unsqueeze(0))
        oof_y.append(torch.cat(ys_f, dim=0))
    return _fit_oof_tau(torch.cat(oof_scores, dim=0), torch.cat(oof_y, dim=0), k_cls, balanced=False)


def _dir(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm().clamp_min(1e-12)


def _estimate_conjugate_weights(
    obs: list[int],
    centroids_m: dict[int, torch.Tensor],
    p_all: torch.Tensor,
    other_centroids: dict[int, torch.Tensor],
    cross_map: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    anchor_weight: float = 1.0,
) -> dict[str, float]:
    """Estimate cone-adjusted vMF sensor reliability weights (w_P, w_CM) and sigma^2_trans."""
    if not obs:
        return {"w_p": 1.0, "w_cm": 0.0, "r_p_eff": 0.0, "r_cm_eff": 0.0, "base_cos": 0.0, "sigma2_trans": 1e-4}
    d = int(p_all.shape[-1])
    mean_norm = sum(float(centroids_m[c].norm().item()) for c in obs) / float(len(obs))
    obs_dirs = torch.stack([_dir(centroids_m[c]) for c in obs], dim=0)
    base_cos = (
        float(((obs_dirs @ obs_dirs.T).sum().item() - len(obs)) / float(len(obs) * (len(obs) - 1)))
        if len(obs) >= 2 else 0.0
    )

    cos_p_list: list[float] = []
    for cj in obs:
        tgt_d = _dir(centroids_m[cj])
        if len(obs) >= 3:
            sub = [c for c in obs if c != cj]
            rp_l, pb_l, mb_l = _identity_anchored_procrustes(
                torch.stack([p_all[c] for c in sub], dim=0),
                torch.stack([centroids_m[c] for c in sub], dim=0),
                anchor_weight=max(anchor_weight, 1.0),
            )
            pred_p = _dir(mb_l + rp_l @ (p_all[cj] - pb_l))
        elif len(obs) == 2:
            other_c = obs[1] if cj == obs[0] else obs[0]
            u_o, _, vh_o = torch.linalg.svd(max(anchor_weight, 1.0) * torch.eye(d, dtype=torch.float64) + torch.outer(centroids_m[other_c], p_all[other_c]), full_matrices=False)
            pred_p = _dir((u_o @ vh_o) @ p_all[cj])
        else:
            pred_p = _dir(p_all[cj])
        cos_p_list.append(float((pred_p @ tgt_d).item()))
    r_p_raw = sum(cos_p_list) / float(len(cos_p_list))
    r_p_eff = min(max((r_p_raw - base_cos) / max(1.0 - base_cos, 1e-4), 0.0), 0.995)
    w_p = max(r_p_eff / (1.0 - r_p_eff * r_p_eff), 1e-3)

    co_obs = [c for c in obs if c in other_centroids]
    w_cm, r_cm_eff = 0.0, 0.0
    cos_cm_list: list[float] = []
    if cross_map is not None and co_obs:
        rot_cm, src_m, dst_m = cross_map
        for cj in co_obs:
            tgt_d = _dir(centroids_m[cj])
            c_map = _dir(dst_m + rot_cm @ (other_centroids[cj] - src_m))
            if len(co_obs) >= 3:
                sub = [c for c in co_obs if c != cj]
                rl, sl, dl = _identity_anchored_procrustes(
                    torch.stack([other_centroids[c] for c in sub], dim=0),
                    torch.stack([centroids_m[c] for c in sub], dim=0),
                    anchor_weight=max(anchor_weight, 1.0),
                )
                c_loo = _dir(dl + rl @ (other_centroids[cj] - sl))
                cos_cm_list.append(0.5 * float((c_map @ tgt_d).item()) + 0.5 * float((c_loo @ tgt_d).item()))
            else:
                cos_cm_list.append(float((c_map @ tgt_d).item()))
        r_cm_raw = sum(cos_cm_list) / float(len(cos_cm_list))
        r_cm_eff = min(max((r_cm_raw - base_cos) / max(1.0 - base_cos, 1e-4), 0.0), 0.995)
        w_cm = max(r_cm_eff / (1.0 - r_cm_eff * r_cm_eff), 1e-3)

    rp_f, pb_f, mb_f = _identity_anchored_procrustes(
        torch.stack([p_all[c] for c in obs], dim=0),
        torch.stack([centroids_m[c] for c in obs], dim=0),
        anchor_weight=max(anchor_weight, 1.0),
    )
    sq_errs: list[float] = []
    for cj in obs:
        cp = _dir(mb_f + rp_f @ (p_all[cj] - pb_f))
        if cross_map is not None and cj in other_centroids:
            ccm = _dir(cross_map[2] + cross_map[0] @ (other_centroids[cj] - cross_map[1]))
            comb = _scale_to_norm(w_p * cp + w_cm * ccm, mean_norm)
        else:
            comb = _scale_to_norm(cp, mean_norm)
        sq_errs.append(float((centroids_m[cj] - comb).square().sum().item()) / float(d))
    sigma2_trans = max(sum(sq_errs) / float(len(sq_errs)), 1e-6)
    return {
        "w_p": float(w_p),
        "w_cm": float(w_cm),
        "r_p_eff": float(r_p_eff),
        "r_cm_eff": float(r_cm_eff),
        "base_cos": float(base_cos),
        "sigma2_trans": float(sigma2_trans),
    }
