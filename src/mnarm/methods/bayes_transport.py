"""Self-Normalized Empirical-Bayes Density + Conjugate Cross-Modal Transport (zero D_cal labels)."""

from __future__ import annotations

import math
from typing import Any

import torch

from mnarm.methods.spherical_stats import (
    _dir,
    _equalize_mahalanobis_centroids,
    _estimate_conjugate_weights,
    _fit_direct_oof_tau,
    _fit_stat_oof_tau,
    _identity_anchored_procrustes,
    _oas_shrinkage_covariance,
    _procrustes_from_cross_moments,
    _safe_precision_from_cov,
    _scale_to_norm,
    _shrink_sample_cov,
)
from mnarm.models.vlm import l2_normalize

VALID_BRIDGE_MODES = ("conjugate_bayes", "prompt_bridge", "cross_modal_transport", "hybrid_bridge", "none_zero_shot")
VALID_PARTIAL_COV_MODES = ("marginal_epistemic", "spectral_floor", "isotropic_vmf", "raw_oas")


class SelfNormalizedBayesTransportHead:
    """Zero-D_cal-label generative Shrinkage-LDA / vMF classifier with conjugate cross-modal bridge."""

    def __init__(
        self,
        bridge_mode: str = "conjugate_bayes",
        shrinkage: float | None = None,
        calib_folds: int = 5,
        anchor_weight: float = 1.0,
        partial_cov_mode: str = "marginal_epistemic",
    ) -> None:
        if bridge_mode not in VALID_BRIDGE_MODES:
            raise ValueError(f"Unsupported bridge_mode={bridge_mode!r}")
        if partial_cov_mode not in VALID_PARTIAL_COV_MODES:
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
            return {"observed_classes": [], "centroids": {}, "counts": {}, "mean_norm": 0.0}
        u = l2_normalize(feats).to(dtype=torch.float64, device="cpu")
        y = labels.detach().cpu().long()
        obs = sorted({int(v) for v in y.tolist()})
        is_partial = len(obs) < int(n_seen_total)
        centroids = {c: u[y == c].mean(dim=0) for c in obs}
        counts_map = {c: int((y == c).sum().item()) for c in obs}
        mean_norm = sum(float(mu.norm().item()) for mu in centroids.values()) / max(len(centroids), 1)
        cov, _, rho = _oas_shrinkage_covariance(u, y, obs, shrinkage=self.shrinkage)
        precision = _safe_precision_from_cov(cov, is_partial, self.partial_cov_mode)
        return {
            "observed_classes": obs,
            "centroids": centroids,
            "counts": counts_map,
            "mean_norm": float(mean_norm),
            "cov": cov,
            "precision": precision,
            "rho": float(rho),
            "tau": 0.0,
            "oof_u": u,
            "oof_y": y,
            "cov_trace": float(torch.trace(cov).item()),
        }

    def _assemble_towers(
        self,
        raw_stats: dict[str, dict[str, Any]],
        paired_feats_t: torch.Tensor | None,
        paired_feats_i: torch.Tensor | None,
        protos_full: torch.Tensor,
        seen_classes: list[int],
        fallback_scale: float,
        paired_moments: dict[str, Any] | None = None,
    ) -> "SelfNormalizedBayesTransportHead":
        self.seen_classes = [int(c) for c in seen_classes]
        p_all = l2_normalize(protos_full).to(dtype=torch.float64, device="cpu")
        cross_maps: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        has_moments = paired_moments is not None and int(paired_moments.get("n_pairs", 0)) >= 2
        has_samples = (
            paired_feats_t is not None and paired_feats_i is not None
            and paired_feats_t.shape[0] >= 2 and paired_feats_i.shape[0] == paired_feats_t.shape[0]
        )
        if has_moments:
            assert paired_moments is not None
            n_p = int(paired_moments["n_pairs"])
            s_t, s_i, c_ti = paired_moments["sum_t"], paired_moments["sum_i"], paired_moments["cross_ti"]
            cross_maps["image_to_text"] = _procrustes_from_cross_moments(n_p, s_i, s_t, c_ti, self.anchor_weight)
            cross_maps["text_to_image"] = _procrustes_from_cross_moments(n_p, s_t, s_i, c_ti.T, self.anchor_weight)
        elif has_samples:
            pt_n = l2_normalize(paired_feats_t).to(dtype=torch.float64, device="cpu")  # type: ignore[arg-type]
            pi_n = l2_normalize(paired_feats_i).to(dtype=torch.float64, device="cpu")  # type: ignore[arg-type]
            cross_maps["image_to_text"] = _identity_anchored_procrustes(pi_n, pt_n, self.anchor_weight)
            cross_maps["text_to_image"] = _identity_anchored_procrustes(pt_n, pi_n, self.anchor_weight)

        self.diagnostics = {
            "bridge_mode": self.bridge_mode,
            "partial_cov_mode": self.partial_cov_mode,
            "has_paired_transport": bool(has_moments or has_samples),
            "towers": {},
        }
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

            mean_norm, obs_set = max(st["mean_norm"], 1e-4), set(obs)
            is_partial = len(obs) < len(self.seen_classes)
            rot_p, p_bar, m_bar = _identity_anchored_procrustes(
                torch.stack([p_all[c] for c in obs], dim=0),
                torch.stack([st["centroids"][c] for c in obs], dim=0),
                anchor_weight=max(self.anchor_weight, 1.0),
            )
            t_key = f"{other_tw}_to_{tw}"
            conj = _estimate_conjugate_weights(obs, st["centroids"], p_all, other_st["centroids"], cross_maps.get(t_key), self.anchor_weight)
            prec = _safe_precision_from_cov(st["cov"], is_partial, self.partial_cov_mode, sigma2_trans=conj["sigma2_trans"])

            tau_m = float(st["tau"])
            if tau_m <= 0.0 and st.get("oof_u") is not None and st.get("oof_y") is not None:
                tau_m = _fit_direct_oof_tau(
                    st["oof_u"], st["oof_y"], obs, self.shrinkage, is_partial, self.partial_cov_mode, conj["sigma2_trans"], self.calib_folds
                )
            elif tau_m <= 0.0:
                tau_m = _fit_stat_oof_tau(
                    st["centroids"], st.get("counts", {}), st.get("raw_sample_cov"),
                    prec, is_partial, self.partial_cov_mode, calib_folds=self.calib_folds,
                )

            mu_by_class, transported_cls, obs_local_idx = [], [], []
            counts_m = st.get("counts", {})
            for idx, cls in enumerate(self.seen_classes):
                cand_p = _scale_to_norm(m_bar + rot_p @ (p_all[cls] - p_bar), mean_norm)
                if cls in obs_set:
                    obs_local_idx.append(idx)
                    mu_obs = st["centroids"][cls]
                    if self.bridge_mode == "conjugate_bayes":
                        n_c = float(counts_m.get(cls, 100))
                        prior_pull = (conj["r_p_eff"] ** 2) * (10.0 / (n_c + 10.0))
                        if prior_pull > 1e-4:
                            mu_obs = _scale_to_norm((1.0 - prior_pull) * _dir(mu_obs) + prior_pull * _dir(cand_p), float(mu_obs.norm().item()))
                    mu_by_class.append(mu_obs)
                    continue
                transported_cls.append(cls)
                if self.bridge_mode == "none_zero_shot":
                    mu_est = _scale_to_norm(p_all[cls], mean_norm)
                else:
                    cand_cm = None
                    if t_key in cross_maps and cls in other_st["centroids"]:
                        rot_cm, src_m, dst_m = cross_maps[t_key]
                        cand_cm = _scale_to_norm(dst_m + rot_cm @ (other_st["centroids"][cls] - src_m), mean_norm)
                    if self.bridge_mode == "prompt_bridge" or cand_cm is None:
                        mu_est = cand_p
                    elif self.bridge_mode == "cross_modal_transport":
                        mu_est = cand_cm
                    elif self.bridge_mode == "hybrid_bridge":
                        mu_est = _scale_to_norm(0.5 * _dir(cand_p) + 0.5 * _dir(cand_cm), mean_norm)
                    else:
                        mu_est = _scale_to_norm(conj["w_p"] * _dir(cand_p) + conj["w_cm"] * _dir(cand_cm), mean_norm)
                mu_by_class.append(mu_est)

            mu_stack = torch.stack(mu_by_class, dim=0)
            if is_partial and self.partial_cov_mode != "raw_oas":
                mu_stack = _equalize_mahalanobis_centroids(mu_stack, prec, obs_local_idx)
            w_mat = mu_stack @ prec
            self.weights[tw], self.biases[tw], self.taus[tw] = w_mat.float(), (-0.5 * (mu_stack * w_mat).sum(dim=-1)).float(), float(tau_m)
            self.diagnostics["towers"][tw] = {
                "observed": obs, "transported": transported_cls, "is_partial": is_partial,
                "partial_cov_mode": self.partial_cov_mode, "mean_centroid_norm": float(mean_norm),
                "rho": float(st["rho"]), "tau": float(tau_m), **conj,
            }
        return self

    def fit(
        self,
        train_feats_t: torch.Tensor | None,
        train_y_t: torch.Tensor | None,
        train_feats_i: torch.Tensor | None,
        train_y_i: torch.Tensor | None,
        paired_feats_t: torch.Tensor | None,
        paired_feats_i: torch.Tensor | None,
        protos_full: torch.Tensor,
        seen_classes: list[int],
        fallback_scale: float = 20.0,
        paired_moments: dict[str, Any] | None = None,
    ) -> "SelfNormalizedBayesTransportHead":
        """Fit class-conditional densities + modality-gap bridges using training features only."""
        n_seen = len(seen_classes)
        raw_stats = {"text": self._fit_single_tower_stats(train_feats_t, train_y_t, n_seen), "image": self._fit_single_tower_stats(train_feats_i, train_y_i, n_seen)}
        return self._assemble_towers(raw_stats, paired_feats_t, paired_feats_i, protos_full, seen_classes, fallback_scale, paired_moments=paired_moments)

    def fit_from_sufficient_stats(
        self,
        tower_stats: dict[str, dict[str, Any]],
        paired_feats_t: torch.Tensor | None,
        paired_feats_i: torch.Tensor | None,
        protos_full: torch.Tensor,
        seen_classes: list[int],
        fallback_scale: float = 20.0,
        paired_moments: dict[str, Any] | None = None,
    ) -> "SelfNormalizedBayesTransportHead":
        """Fit from exemplar-free transported centroids, within-class scatter, O(d^2) cross-moments, and OOF taus."""
        raw_stats: dict[str, dict[str, Any]] = {}
        d = int(protos_full.shape[-1])
        pm = paired_moments if paired_moments is not None else tower_stats.get("paired_moments")  # type: ignore[assignment]
        for tw in ("text", "image"):
            st = tower_stats.get(tw, {})
            centroids = st.get("centroids", {})
            obs = sorted(centroids)
            if not obs:
                raw_stats[tw] = {"observed_classes": [], "centroids": {}, "counts": {}, "mean_norm": 0.0}
                continue
            mean_norm = sum(float(mu.norm().item()) for mu in centroids.values()) / len(obs)
            scatter, dof = st.get("within_scatter"), int(st.get("scatter_dof", 0))
            n_tot = int(st.get("total_samples", dof + len(obs)))
            if scatter is not None and dof > 0:
                raw_sample_cov = scatter.to(dtype=torch.float64, device="cpu") / float(dof)
                cov, _, rho = _shrink_sample_cov(raw_sample_cov, n_samples=n_tot, dim=d, shrinkage=self.shrinkage)
            else:
                raw_sample_cov = None
                cov, rho = torch.eye(d, dtype=torch.float64), 1.0
            raw_stats[tw] = {
                "observed_classes": obs, "centroids": centroids, "counts": st.get("counts", {}),
                "mean_norm": float(mean_norm), "cov": cov, "raw_sample_cov": raw_sample_cov, "rho": float(rho),
                "tau": float(st.get("tau", 0.0)), "cov_trace": float(torch.trace(cov).item()),
            }
        return self._assemble_towers(raw_stats, paired_feats_t, paired_feats_i, protos_full, seen_classes, fallback_scale, paired_moments=pm)

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
