"""Exemplar-free streaming class sufficient-statistics bank with canonical base-space transport."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from mnarm.methods.bayes_transport import _identity_anchored_procrustes
from mnarm.models.vlm import l2_normalize


class StreamingClassBank:
    """Stores zero-exemplar per-class centroids and pooled within-class scatter in canonical base space."""

    def __init__(self) -> None:
        self.class_counts: dict[str, dict[int, int]] = {"text": {}, "image": {}}
        self.class_centroids: dict[str, dict[int, torch.Tensor]] = {"text": {}, "image": {}}
        self.within_scatter: dict[str, torch.Tensor | None] = {"text": None, "image": None}
        self.scatter_dof: dict[str, int] = {"text": 0, "image": 0}
        self.total_samples: dict[str, int] = {"text": 0, "image": 0}
        self.paired_base_t: list[torch.Tensor] = []
        self.paired_base_i: list[torch.Tensor] = []

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
        """Record new-stage centroids & rotated within-class scatter in canonical theta_0 space."""
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
            self.paired_base_t.append(l2_normalize(paired_base_t).cpu())
            self.paired_base_i.append(l2_normalize(paired_base_i).cpu())

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
    ) -> dict[str, Any]:
        """Transport canonical class centroids & within-class scatter into merged model space."""
        maps: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = {"text": None, "image": None}
        for tw, u0, um in (("text", bridge_u0_t, bridge_um_t), ("image", bridge_u0_i, bridge_um_i)):
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
                "within_scatter": scatter,
                "scatter_dof": self.scatter_dof[tw],
                "total_samples": self.total_samples[tw],
                "probe_x": px,
                "probe_y": py,
            }

        paired_out: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        if self.paired_base_t and self.paired_base_i:
            p0_t = torch.cat(self.paired_base_t, dim=0).to(torch.float64)
            p0_i = torch.cat(self.paired_base_i, dim=0).to(torch.float64)
            if maps["text"] is not None:
                rt, st, dt = maps["text"]
                p0_t = (p0_t - st.unsqueeze(0)) @ rt.T + dt.unsqueeze(0)
            if maps["image"] is not None:
                ri, si, di = maps["image"]
                p0_i = (p0_i - si.unsqueeze(0)) @ ri.T + di.unsqueeze(0)
            paired_out = (p0_t.float(), p0_i.float())

        return {"towers": tower_stats, "paired": paired_out}
