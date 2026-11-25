"""Self-Normalized Shrinkage-LDA Density + Modality-Gap Transport with zero D_cal labels."""

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


def _safe_precision_from_cov(cov: torch.Tensor, is_partial: bool, partial_cov_mode: str) -> torch.Tensor:
    """Prevent precision nullspace explosion on partially observed towers (S_m != Y)."""
    d = cov.shape[0]
    if not is_partial or partial_cov_mode == "raw_oas":
        return torch.cholesky_inverse(torch.linalg.cholesky(cov))
    if partial_cov_mode == "isotropic_vmf":
        return torch.eye(d, dtype=torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    tr = max(float(torch.trace(cov).item()), 1e-12)
    lam_eff = max(float(cov.square().sum().item()) / tr, 1e-6)
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


def _fit_oof_tau(oof_scores: torch.Tensor, oof_y_local: torch.Tensor, num_obs: int) -> float:
    """Fit 1 scalar log-temperature tau > 0 in exact nat-scale via class-balanced NLL on OOF scores."""
    if oof_scores.numel() == 0 or num_obs < 2:
        return 1.0
    counts = torch.bincount(oof_y_local, minlength=num_obs).to(torch.float64).clamp_min(1.0)
    sw = 1.0 / counts[oof_y_local]
    sw = sw / sw.sum().clamp_min(1e-12)
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


class SelfNormalizedBayesTransportHead:
    """Zero-D_cal-label generative Shrinkage-LDA classifier with cross-modal gap transport."""

    def __init__(self, bridge_mode: str = "hybrid_bridge", shrinkage: float | None = None, calib_folds: int = 5, anchor_weight: float = 1.0, partial_cov_mode: str = "spectral_floor"):
        if bridge_mode not in {"prompt_bridge", "cross_modal_transport", "hybrid_bridge", "none_zero_shot"}:
            raise ValueError(f"Unsupported bridge_mode={bridge_mode!r}")
        if partial_cov_mode not in {"spectral_floor", "isotropic_vmf", "raw_oas"}:
            raise ValueError(f"Unsupported partial_cov_mode={partial_cov_mode!r}")
        self.bridge_mode, self.partial_cov_mode = bridge_mode, partial_cov_mode
        self.shrinkage, self.calib_folds, self.anchor_weight = shrinkage, int(calib_folds), float(anchor_weight)
        self.seen_classes: list[int] = []
        self.weights: dict[str, torch.Tensor] = {}
        self.biases: dict[str, torch.Tensor] = {}
        self.taus: dict[str, float] = {}
        self.diagnostics: dict[str, Any] = {}

    def _fit_single_tower_stats(self, feats: torch.Tensor | None, labels: torch.Tensor | None, n_seen_total: int) -> dict[str, Any]:
        if feats is None or labels is None or feats.shape[0] == 0:
            return {"observed_classes": [], "centroids": {}, "mean_norm": 0.0}
        u = l2_normalize(feats).to(dtype=torch.float64, device="cpu")
        y = labels.detach().cpu().long()
        obs = sorted({int(v) for v in y.tolist()})
        is_partial = len(obs) < int(n_seen_total)
        centroids = {c: u[y == c].mean(dim=0) for c in obs}
        mean_norm = sum(float(mu.norm().item()) for mu in centroids.values()) / max(len(centroids), 1)
        cov, _, rho = _oas_shrinkage_covariance(u, y, obs, shrinkage=self.shrinkage)
        precision = _safe_precision_from_cov(cov, is_partial, self.partial_cov_mode)

        tau_m, counts = 1.0, [int((y == c).sum().item()) for c in obs]
        if len(obs) >= 2 and min(counts) >= 2:
            folds = _stratified_folds([int(v) for v in y.tolist()], min(self.calib_folds, min(counts)))
            obs_map = {c: i for i, c in enumerate(obs)}
            oof_scores_list, oof_y_list, all_idx, obs_rng = [], [], set(range(int(u.shape[0]))), list(range(len(obs)))
            for f_idx in folds:
                tr_idx = sorted(all_idx - set(f_idx))
                u_tr, y_tr = u[tr_idx], y[tr_idx]
                cov_f, _, _ = _oas_shrinkage_covariance(u_tr, y_tr, obs, shrinkage=self.shrinkage)
                prec_f = _safe_precision_from_cov(cov_f, is_partial, self.partial_cov_mode)
                mu_mat = torch.stack([u_tr[y_tr == c].mean(dim=0) for c in obs], dim=0)
                if is_partial and self.partial_cov_mode != "raw_oas":
                    mu_mat = _equalize_mahalanobis_centroids(mu_mat, prec_f, obs_rng)
                w_f = mu_mat @ prec_f
                oof_scores_list.append(u[f_idx] @ w_f.T + (-0.5 * (mu_mat * w_f).sum(dim=-1)).unsqueeze(0))
                oof_y_list.append(torch.tensor([obs_map[int(v)] for v in y[f_idx].tolist()], dtype=torch.long))
            tau_m = _fit_oof_tau(torch.cat(oof_scores_list, dim=0), torch.cat(oof_y_list, dim=0), len(obs))

        return {"observed_classes": obs, "centroids": centroids, "mean_norm": float(mean_norm), "precision": precision, "rho": float(rho), "tau": float(tau_m), "cov_trace": float(torch.trace(cov).item())}

    def _assemble_towers(self, raw_stats: dict[str, dict[str, Any]], paired_feats_t: torch.Tensor | None, paired_feats_i: torch.Tensor | None, protos_full: torch.Tensor, seen_classes: list[int], fallback_scale: float) -> "SelfNormalizedBayesTransportHead":
        self.seen_classes = [int(c) for c in seen_classes]
        p_all = l2_normalize(protos_full).to(dtype=torch.float64, device="cpu")
        has_paired = paired_feats_t is not None and paired_feats_i is not None and paired_feats_t.shape[0] >= 2 and paired_feats_i.shape[0] == paired_feats_t.shape[0]
        cross_maps: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if has_paired:
            pt_n, pi_n = l2_normalize(paired_feats_t).to(dtype=torch.float64, device="cpu"), l2_normalize(paired_feats_i).to(dtype=torch.float64, device="cpu")
            cross_maps["image_to_text"] = _identity_anchored_procrustes(pi_n, pt_n, self.anchor_weight)
            cross_maps["text_to_image"] = _identity_anchored_procrustes(pt_n, pi_n, self.anchor_weight)

        self.diagnostics = {"bridge_mode": self.bridge_mode, "partial_cov_mode": self.partial_cov_mode, "has_paired_transport": has_paired, "towers": {}}
        for tw in ("text", "image"):
            other_tw = "image" if tw == "text" else "text"
            st, other_st = raw_stats[tw], raw_stats[other_tw]
            obs = st["observed_classes"]
            if not obs:
                mu_mat = torch.stack([p_all[c] for c in self.seen_classes], dim=0)
                tau_m = float(fallback_scale) / max(math.sqrt(float(len(self.seen_classes))), 1.0)
                self.weights[tw], self.biases[tw], self.taus[tw] = mu_mat.float(), (-0.5 * (mu_mat * mu_mat).sum(dim=-1)).float(), float(tau_m)
                self.diagnostics["towers"][tw] = {"observed": [], "transported": list(self.seen_classes), "tau": tau_m, "rho": 1.0, "is_partial": True, "partial_cov_mode": self.partial_cov_mode}
                continue

            mean_norm, prec, obs_set = max(st["mean_norm"], 1e-4), st["precision"], set(obs)
            is_partial = len(obs) < len(self.seen_classes)
            rot_p, p_bar, m_bar = _identity_anchored_procrustes(torch.stack([p_all[c] for c in obs], dim=0), torch.stack([st["centroids"][c] for c in obs], dim=0), anchor_weight=max(self.anchor_weight, 1.0))

            t_key = f"{other_tw}_to_{tw}"
            mu_by_class, transported_cls, obs_local_idx = [], [], []
            for idx, cls in enumerate(self.seen_classes):
                if cls in obs_set:
                    obs_local_idx.append(idx)
                    mu_by_class.append(st["centroids"][cls])
                    continue
                transported_cls.append(cls)
                if self.bridge_mode == "none_zero_shot":
                    mu_est = _scale_to_norm(p_all[cls], mean_norm)
                else:
                    cand_p = _scale_to_norm(m_bar + rot_p @ (p_all[cls] - p_bar), mean_norm)
                    cand_cm = None
                    if t_key in cross_maps and cls in other_st["centroids"]:
                        rot_cm, src_m, dst_m = cross_maps[t_key]
                        cand_cm = _scale_to_norm(dst_m + rot_cm @ (other_st["centroids"][cls] - src_m), mean_norm)
                    if self.bridge_mode == "prompt_bridge" or cand_cm is None:
                        mu_est = cand_p
                    elif self.bridge_mode == "cross_modal_transport":
                        mu_est = cand_cm
                    else:
                        mu_est = _scale_to_norm(0.5 * (cand_p / cand_p.norm().clamp_min(1e-12)) + 0.5 * (cand_cm / cand_cm.norm().clamp_min(1e-12)), mean_norm)
                mu_by_class.append(mu_est)

            mu_stack = torch.stack(mu_by_class, dim=0)
            if is_partial and self.partial_cov_mode != "raw_oas":
                mu_stack = _equalize_mahalanobis_centroids(mu_stack, prec, obs_local_idx)
            w_mat = mu_stack @ prec
            self.weights[tw], self.biases[tw], self.taus[tw] = w_mat.float(), (-0.5 * (mu_stack * w_mat).sum(dim=-1)).float(), float(st["tau"])
            self.diagnostics["towers"][tw] = {"observed": obs, "transported": transported_cls, "is_partial": is_partial, "partial_cov_mode": self.partial_cov_mode, "mean_centroid_norm": float(mean_norm), "rho": float(st["rho"]), "tau": float(st["tau"])}
        return self

    def fit(self, train_feats_t: torch.Tensor | None, train_y_t: torch.Tensor | None, train_feats_i: torch.Tensor | None, train_y_i: torch.Tensor | None, paired_feats_t: torch.Tensor | None, paired_feats_i: torch.Tensor | None, protos_full: torch.Tensor, seen_classes: list[int], fallback_scale: float = 20.0) -> "SelfNormalizedBayesTransportHead":
        """Fit class-conditional densities + modality-gap bridges using training features only."""
        n_seen = len(seen_classes)
        raw_stats = {"text": self._fit_single_tower_stats(train_feats_t, train_y_t, n_seen), "image": self._fit_single_tower_stats(train_feats_i, train_y_i, n_seen)}
        return self._assemble_towers(raw_stats, paired_feats_t, paired_feats_i, protos_full, seen_classes, fallback_scale)

    def fit_from_sufficient_stats(self, tower_stats: dict[str, dict[str, Any]], paired_feats_t: torch.Tensor | None, paired_feats_i: torch.Tensor | None, protos_full: torch.Tensor, seen_classes: list[int], fallback_scale: float = 20.0) -> "SelfNormalizedBayesTransportHead":
        """Fit from exemplar-free transported centroids, within-class scatter matrices, and OOF taus."""
        raw_stats: dict[str, dict[str, Any]] = {}
        d = int(protos_full.shape[-1])
        for tw in ("text", "image"):
            st = tower_stats.get(tw, {})
            centroids = st.get("centroids", {})
            obs = sorted(centroids)
            if not obs:
                raw_stats[tw] = {"observed_classes": [], "centroids": {}, "mean_norm": 0.0}
                continue
            is_partial = len(obs) < len(seen_classes)
            mean_norm = sum(float(mu.norm().item()) for mu in centroids.values()) / len(obs)
            scatter, dof = st.get("within_scatter"), int(st.get("scatter_dof", 0))
            n_tot = int(st.get("total_samples", dof + len(obs)))
            if scatter is not None and dof > 0:
                cov, _, rho = _shrink_sample_cov(scatter.to(dtype=torch.float64, device="cpu") / float(dof), n_samples=n_tot, dim=d, shrinkage=self.shrinkage)
                prec = _safe_precision_from_cov(cov, is_partial, self.partial_cov_mode)
            else:
                cov, prec, rho = torch.eye(d, dtype=torch.float64), torch.eye(d, dtype=torch.float64), 1.0

            tau_m, probe_x, probe_y = float(st.get("tau", 0.0)), st.get("probe_x"), st.get("probe_y")
            if tau_m <= 0.0 and probe_x is not None and probe_y is not None:
                p_obs = sorted({int(v) for v in probe_y.tolist()})
                if len(p_obs) >= 2:
                    p_u, p_map = l2_normalize(probe_x).to(dtype=torch.float64, device="cpu"), {c: i for i, c in enumerate(p_obs)}
                    mu_p = torch.stack([centroids[c] for c in p_obs], dim=0)
                    if is_partial and self.partial_cov_mode != "raw_oas":
                        mu_p = _equalize_mahalanobis_centroids(mu_p, prec, list(range(len(p_obs))))
                    w_p = mu_p @ prec
                    tau_m = _fit_oof_tau(p_u @ w_p.T + (-0.5 * (mu_p * w_p).sum(dim=-1)).unsqueeze(0), torch.tensor([p_map[int(v)] for v in probe_y.tolist()], dtype=torch.long), len(p_obs))
                else:
                    tau_m = 0.05
            elif tau_m <= 0.0:
                tau_m = 0.05

            raw_stats[tw] = {"observed_classes": obs, "centroids": centroids, "mean_norm": float(mean_norm), "precision": prec, "rho": float(rho), "tau": float(tau_m), "cov_trace": float(torch.trace(cov).item())}
        return self._assemble_towers(raw_stats, paired_feats_t, paired_feats_i, protos_full, seen_classes, fallback_scale)

    def score_tower(self, feats: torch.Tensor, tower: str) -> torch.Tensor:
        u = l2_normalize(feats).detach().cpu().float()
        return float(self.taus[tower]) * (u @ self.weights[tower].T + self.biases[tower].unsqueeze(0))

    def predict_views(self, feats_t: torch.Tensor | None, feats_i: torch.Tensor | None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if feats_t is not None:
            out["text"] = self.score_tower(feats_t, "text")
        if feats_i is not None:
            out["image"] = self.score_tower(feats_i, "image")
        if "text" in out and "image" in out:
            out["paired"] = out["text"] + out["image"]
        elif "text" in out:
            out["paired"] = out["text"]
        elif "image" in out:
            out["paired"] = out["image"]
        return out
