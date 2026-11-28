"""Exemplar-free streaming class sufficient-statistics bank in O(|Y|d + d^2) memory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from mnarm.methods.reanchor import fit_canonical_reanchor
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

    def memory_bytes(self, bytes_per_float: int = 8) -> int:
        """Exact byte footprint of stored float64/float32 moments."""
        return self.memory_floats() * int(bytes_per_float)

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
        protos_0: torch.Tensor | None = None,
        protos_m: torch.Tensor | None = None,
    ) -> None:
        """Align stage expert features isometrically (O(d)) onto theta_0 and accumulate O(|Y|d + d^2) moments."""
        for tw, ue, ub, y in (
            ("text", u_expert_t, u_base_t, y_t),
            ("image", u_expert_i, u_base_i, y_i),
        ):
            if ue is None or ub is None or y is None or ue.shape[0] == 0:
                continue
            ue64 = l2_normalize(ue).to(dtype=torch.float64, device="cpu")
            ub64 = l2_normalize(ub).to(dtype=torch.float64, device="cpu")
            p0 = protos_0 if tw == "text" else None
            pm = protos_m if tw == "text" else None
            q_mat = fit_canonical_reanchor(ub64, ue64, protos_0=p0, protos_m=pm, anchor_weight=0.05)["q_mat"].to(dtype=torch.float64, device="cpu")
            aligned = ue64 @ q_mat.T
            y_cpu = y.detach().cpu().long()

            by_cls: dict[int, list[torch.Tensor]] = defaultdict(list)
            for idx, lbl in enumerate(y_cpu.tolist()):
                by_cls[int(lbl)].append(aligned[idx])

            for cls, vecs in by_cls.items():
                mat = torch.stack(vecs, dim=0)
                n_new = int(mat.shape[0])
                mean_new = mat.mean(dim=0)
                res = mat - mean_new.unsqueeze(0)
                sc_new = res.T @ res
                if cls in self.class_centroids[tw]:
                    n_old = self.class_counts[tw][cls]
                    mean_old = self.class_centroids[tw][cls]
                    n_tot = n_old + n_new
                    delta = mean_new - mean_old
                    sc_add = sc_new + (float(n_old * n_new) / float(n_tot)) * torch.outer(delta, delta)
                    dof_add = n_new
                    self.class_centroids[tw][cls] = (n_old * mean_old + n_new * mean_new) / float(n_tot)
                    self.class_counts[tw][cls] = n_tot
                else:
                    self.class_centroids[tw][cls] = mean_new
                    self.class_counts[tw][cls] = n_new
                    sc_add = sc_new
                    dof_add = n_new - 1

                if dof_add > 0 or self.within_scatter[tw] is not None:
                    prev_sc = self.within_scatter[tw]
                    self.within_scatter[tw] = sc_add if prev_sc is None else (prev_sc + sc_add)
                    self.scatter_dof[tw] += dof_add
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

    def canonical_stats(self, seen_classes: list[int]) -> dict[str, Any]:
        """Return canonical theta_0 class centroids, within-class scatter, and O(d^2) cross-moments directly."""
        tower_stats: dict[str, dict[str, Any]] = {}
        for tw in ("text", "image"):
            sc = self.within_scatter[tw]
            tower_stats[tw] = {
                "centroids": {int(c): self.class_centroids[tw][c].clone() for c in seen_classes if c in self.class_centroids[tw]},
                "counts": dict(self.class_counts[tw]),
                "within_scatter": sc.clone() if sc is not None else None,
                "scatter_dof": int(self.scatter_dof[tw]),
                "total_samples": int(self.total_samples[tw]),
            }
        paired_moments: dict[str, Any] | None = None
        if self.paired_count >= 2 and self.paired_sum_t is not None and self.paired_sum_i is not None and self.paired_cross_ti is not None:
            paired_moments = {
                "n_pairs": int(self.paired_count),
                "sum_t": self.paired_sum_t.clone(),
                "sum_i": self.paired_sum_i.clone(),
                "cross_ti": self.paired_cross_ti.clone(),
            }
        return {"towers": {**tower_stats, "paired_moments": paired_moments}, "paired": (None, None), "paired_moments": paired_moments}

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
        """Rotate canonical theta_0 moments by isometric O(d) map Q_{0->m} for un-reanchored target ablations."""
        del probe_y_t, probe_y_i
        t0_bridge = bridge_u0_t if (bridge_u0_t is not None and bridge_u0_t.shape[0] >= 2) else protos_0
        tm_bridge = bridge_um_t if (bridge_um_t is not None and bridge_um_t.shape[0] >= 2) else protos_m

        maps: dict[str, torch.Tensor | None] = {"text": None, "image": None}
        for tw, u0, um in (("text", t0_bridge, tm_bridge), ("image", bridge_u0_i, bridge_um_i)):
            if u0 is not None and um is not None and u0.shape[0] >= 2:
                u0_64 = l2_normalize(u0).to(dtype=torch.float64, device="cpu")
                um_64 = l2_normalize(um).to(dtype=torch.float64, device="cpu")
                maps[tw] = fit_canonical_reanchor(um_64, u0_64, anchor_weight=0.05)["q_mat"].to(dtype=torch.float64, device="cpu")

        tower_stats: dict[str, dict[str, Any]] = {}
        for tw in ("text", "image"):
            q0m = maps[tw]
            centroids_out: dict[int, torch.Tensor] = {}
            for cls in seen_classes:
                if cls not in self.class_centroids[tw]:
                    continue
                mu0 = self.class_centroids[tw][cls]
                centroids_out[int(cls)] = (q0m @ mu0) if q0m is not None else mu0.clone()

            scatter = self.within_scatter[tw]
            if scatter is not None:
                scatter = (q0m @ scatter @ q0m.T) if q0m is not None else scatter.clone()

            tower_stats[tw] = {
                "centroids": centroids_out,
                "counts": dict(self.class_counts[tw]),
                "within_scatter": scatter,
                "scatter_dof": int(self.scatter_dof[tw]),
                "total_samples": int(self.total_samples[tw]),
            }

        paired_moments: dict[str, Any] | None = None
        if self.paired_count >= 2 and self.paired_sum_t is not None and self.paired_sum_i is not None and self.paired_cross_ti is not None:
            q_t = maps["text"] if maps["text"] is not None else torch.eye(self.paired_sum_t.numel(), dtype=torch.float64)
            q_i = maps["image"] if maps["image"] is not None else torch.eye(self.paired_sum_i.numel(), dtype=torch.float64)
            paired_moments = {
                "n_pairs": int(self.paired_count),
                "sum_t": q_t @ self.paired_sum_t,
                "sum_i": q_i @ self.paired_sum_i,
                "cross_ti": q_t @ self.paired_cross_ti @ q_i.T,
            }

        return {"towers": {**tower_stats, "paired_moments": paired_moments}, "paired": (None, None), "paired_moments": paired_moments}
