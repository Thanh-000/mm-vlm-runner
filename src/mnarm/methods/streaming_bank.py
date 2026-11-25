"""Exemplar-free streaming class sufficient-statistics bank in O(|Y|d + d^2) memory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from mnarm.methods.spherical_stats import _identity_anchored_procrustes
from mnarm.models.vlm import l2_normalize


class StreamingClassBank:
    """Stores zero-exemplar class centroids, within-class scatter, and O(d^2) cross-moments in theta_0."""

    def __init__(self) -> None:
        self.class_counts: dict[str, dict[int, int]] = {"text": {}, "image": {}}
        self.class_centroids: dict[str, dict[int, torch.Tensor]] = {"text": {}, "image": {}}
        self.within_scatter: dict[str, torch.Tensor | None] = {"text": None, "image": None}
        self.scatter_dof: dict[str, int] = {"text": 0, "image": 0}
        self.total_samples: dict[str, int] = {"text": 0, "image": 0}
        self.paired_count: int = 0
        self.paired_sum_t: torch.Tensor | None = None
        self.paired_sum_i: torch.Tensor | None = None
        self.paired_cross_ti: torch.Tensor | None = None

    def memory_floats(self) -> int:
        """Exact count of floating-point scalars stored in bank (strictly O(|Y|d + d^2))."""
        tot = 0
        for tw in ("text", "image"):
            for mu in self.class_centroids[tw].values():
                tot += int(mu.numel())
            sc = self.within_scatter[tw]
            if sc is not None:
                tot += int(sc.numel())
        for ten in (self.paired_sum_t, self.paired_sum_i, self.paired_cross_ti):
            if ten is not None:
                tot += int(ten.numel())
        return tot

    def update_stage(
        self,
        *,
        u_expert_t: torch.Tensor | None,
        u_base_t: torch.Tensor | None,
        y_t: torch.Tensor | None,
        u_expert_i: torch.Tensor | None,
        u_base_i: torch.Tensor | None,
        y_i: torch.Tensor | None,
        paired_base_t: torch.Tensor | None = None,
        paired_base_i: torch.Tensor | None = None,
    ) -> None:
        """Record stage centroids, rotated scatter, and O(d^2) paired cross-moments in theta_0."""
        for tw, ue, ub, y in (
            ("text", u_expert_t, u_base_t, y_t),
            ("image", u_expert_i, u_base_i, y_i),
        ):
            if ue is None or ub is None or y is None or ue.shape[0] == 0:
                continue
            ue64 = l2_normalize(ue).to(dtype=torch.float64, device="cpu")
            ub64 = l2_normalize(ub).to(dtype=torch.float64, device="cpu")
            rot, src_m, dst_m = _identity_anchored_procrustes(ue64, ub64, anchor_weight=1.0)
            aligned = (ue64 - src_m.unsqueeze(0)) @ rot.T + dst_m.unsqueeze(0)
            y_cpu = y.detach().cpu().long()

            by_cls: dict[int, list[torch.Tensor]] = defaultdict(list)
            for idx, lbl in enumerate(y_cpu.tolist()):
                by_cls[int(lbl)].append(aligned[idx])

            for cls, vecs in by_cls.items():
                mat = torch.stack(vecs, dim=0)
                n_new = int(mat.shape[0])
                mean_new = mat.mean(dim=0)
                if cls in self.class_centroids[tw]:
                    n_old = self.class_counts[tw][cls]
                    mean_old = self.class_centroids[tw][cls]
                    self.class_centroids[tw][cls] = (n_old * mean_old + n_new * mean_new) / float(n_old + n_new)
                    self.class_counts[tw][cls] = n_old + n_new
                else:
                    self.class_centroids[tw][cls] = mean_new
                    self.class_counts[tw][cls] = n_new

                if n_new >= 2:
                    res = mat - mean_new.unsqueeze(0)
                    sc = res.T @ res
                    prev_sc = self.within_scatter[tw]
                    self.within_scatter[tw] = sc if prev_sc is None else (prev_sc + sc)
                    self.scatter_dof[tw] += n_new - 1
                self.total_samples[tw] += n_new

        if paired_base_t is not None and paired_base_i is not None and paired_base_t.shape[0] > 0:
            pt64 = l2_normalize(paired_base_t).to(dtype=torch.float64, device="cpu")
            pi64 = l2_normalize(paired_base_i).to(dtype=torch.float64, device="cpu")
            st_sum = pt64.sum(dim=0)
            si_sum = pi64.sum(dim=0)
            c_ti = pt64.T @ pi64
            self.paired_count += int(pt64.shape[0])
            self.paired_sum_t = st_sum if self.paired_sum_t is None else (self.paired_sum_t + st_sum)
            self.paired_sum_i = si_sum if self.paired_sum_i is None else (self.paired_sum_i + si_sum)
            self.paired_cross_ti = c_ti if self.paired_cross_ti is None else (self.paired_cross_ti + c_ti)

    def transport_to_target_stats(
        self,
        *,
        bridge_u0_t: torch.Tensor | None,
        bridge_um_t: torch.Tensor | None,
        bridge_u0_i: torch.Tensor | None,
        bridge_um_i: torch.Tensor | None,
        seen_classes: list[int],
        probe_y_t: torch.Tensor | None = None,
        probe_y_i: torch.Tensor | None = None,
        protos_0: torch.Tensor | None = None,
        protos_m: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Transport canonical centroids, scatter, and O(d^2) cross-moments into target model space."""
        t0_bridge = bridge_u0_t if (bridge_u0_t is not None and bridge_u0_t.shape[0] >= 2) else protos_0
        tm_bridge = bridge_um_t if (bridge_um_t is not None and bridge_um_t.shape[0] >= 2) else protos_m

        maps: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = {"text": None, "image": None}
        for tw, u0, um in (("text", t0_bridge, tm_bridge), ("image", bridge_u0_i, bridge_um_i)):
            if u0 is not None and um is not None and u0.shape[0] >= 2:
                u0_64 = l2_normalize(u0).to(dtype=torch.float64, device="cpu")
                um_64 = l2_normalize(um).to(dtype=torch.float64, device="cpu")
                maps[tw] = _identity_anchored_procrustes(u0_64, um_64, anchor_weight=1.0)

        tower_stats: dict[str, dict[str, Any]] = {}
        probes = {"text": (bridge_um_t, probe_y_t), "image": (bridge_um_i, probe_y_i)}
        for tw in ("text", "image"):
            rot_info = maps[tw]
            centroids_out: dict[int, torch.Tensor] = {}
            for cls in seen_classes:
                if cls not in self.class_centroids[tw]:
                    continue
                mu0 = self.class_centroids[tw][cls]
                if rot_info is not None:
                    rot, src_m, dst_m = rot_info
                    centroids_out[int(cls)] = dst_m + rot @ (mu0 - src_m)
                else:
                    centroids_out[int(cls)] = mu0.clone()

            scatter = self.within_scatter[tw]
            if scatter is not None and rot_info is not None:
                rot = rot_info[0]
                scatter = rot @ scatter @ rot.T

            px, py = probes[tw]
            tower_stats[tw] = {
                "centroids": centroids_out,
                "counts": dict(self.class_counts[tw]),
                "within_scatter": scatter,
                "scatter_dof": self.scatter_dof[tw],
                "total_samples": self.total_samples[tw],
                "probe_x": px,
                "probe_y": py,
            }

        paired_moments: dict[str, Any] | None = None
        if self.paired_count >= 2 and self.paired_sum_t is not None and self.paired_sum_i is not None and self.paired_cross_ti is not None:
            n_f = float(self.paired_count)
            mu_t = self.paired_sum_t / n_f
            mu_i = self.paired_sum_i / n_f
            cov_ti = self.paired_cross_ti - n_f * torch.outer(mu_t, mu_i)
            rot_t = torch.eye(mu_t.numel(), dtype=torch.float64)
            rot_i = torch.eye(mu_i.numel(), dtype=torch.float64)
            if maps["text"] is not None:
                rot_t, st_m, dt_m = maps["text"]
                mu_t = dt_m + rot_t @ (mu_t - st_m)
            if maps["image"] is not None:
                rot_i, si_m, di_m = maps["image"]
                mu_i = di_m + rot_i @ (mu_i - si_m)
            cov_m = rot_t @ cov_ti @ rot_i.T
            paired_moments = {
                "n_pairs": int(self.paired_count),
                "sum_t": n_f * mu_t,
                "sum_i": n_f * mu_i,
                "cross_ti": cov_m + n_f * torch.outer(mu_t, mu_i),
            }

        return {"towers": {**tower_stats, "paired_moments": paired_moments}, "paired": (None, None), "paired_moments": paired_moments}
