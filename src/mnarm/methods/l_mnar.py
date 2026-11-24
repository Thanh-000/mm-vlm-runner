"""Co-observation graph analysis and Support-Aware Propensity Risk (L_MNAR)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F


def analyze_co_observation_graph(records: list[dict[str, Any]], seen_classes: list[int]) -> dict[str, Any]:
    """Build exact modality supports, propensities, and connected components of G_obs."""
    seen = sorted({int(c) for c in seen_classes})
    seen_set = set(seen)
    total_by_cls: dict[int, int] = defaultdict(int)
    mask_cls_counts: dict[tuple[str, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tower_cls_counts: dict[str, dict[int, int]] = {"text": defaultdict(int), "image": defaultdict(int)}

    for rec in records:
        cls = int(rec["label"])
        if cls not in seen_set:
            continue
        views = tuple(sorted(set(rec["mask"])))
        total_by_cls[cls] += 1
        mask_cls_counts[views][cls] += 1
        for tw in views:
            if tw in tower_cls_counts:
                tower_cls_counts[tw][cls] += 1

    # Connected components of G_obs under exact co-observed mask patterns M
    adj: dict[int, set[int]] = {c: set() for c in seen}
    mask_supports: dict[str, list[int]] = {}
    for views, cls_map in sorted(mask_cls_counts.items()):
        supp = sorted(c for c, cnt in cls_map.items() if cnt > 0)
        mask_supports["+".join(views)] = supp
        for c1 in supp:
            for c2 in supp:
                if c1 != c2:
                    adj[c1].add(c2)

    visited: set[int] = set()
    components: list[list[int]] = []
    for c in seen:
        if c in visited:
            continue
        comp: list[int] = []
        stack = [c]
        visited.add(c)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(adj[cur]):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))

    tower_supports = {
        tw: sorted(c for c in seen if tower_cls_counts[tw].get(c, 0) > 0)
        for tw in ("text", "image")
    }
    tower_propensities = {
        tw: {
            c: (
                float(tower_cls_counts[tw].get(c, 0)) / float(total_by_cls[c])
                if total_by_cls.get(c, 0) > 0
                else 0.0
            )
            for c in seen
        }
        for tw in ("text", "image")
    }
    has_zero_support = any(
        tower_cls_counts[tw].get(c, 0) == 0 for tw in ("text", "image") for c in seen
    )
    return {
        "seen_classes": seen,
        "mask_supports": mask_supports,
        "tower_supports": tower_supports,
        "tower_propensities": tower_propensities,
        "components": components,
        "num_components": len(components),
        "has_zero_support_mnar": has_zero_support,
    }


def tower_support_propensity_loss(
    logits: torch.Tensor,
    labels_global: torch.Tensor,
    seen_classes: list[int],
    supported_classes: list[int],
    propensities: dict[int, float],
) -> torch.Tensor:
    """Support-restricted propensity-adjusted CE giving exact 0 gradient on unsupported classes."""
    seen = [int(c) for c in seen_classes]
    supp = sorted({int(c) for c in supported_classes} & set(seen))
    if len(supp) < 2 or logits.numel() == 0:
        return logits.sum() * 0.0

    seen_pos = {c: i for i, c in enumerate(seen)}
    supp_pos = {c: i for i, c in enumerate(supp)}
    labels_list = [int(v) for v in labels_global.detach().tolist()]
    valid_rows = [idx for idx, yv in enumerate(labels_list) if yv in supp_pos]
    if not valid_rows:
        return logits.sum() * 0.0

    row_t = torch.tensor(valid_rows, dtype=torch.long, device=logits.device)
    col_t = torch.tensor([seen_pos[c] for c in supp], dtype=torch.long, device=logits.device)
    y_local = torch.tensor([supp_pos[labels_list[idx]] for idx in valid_rows], dtype=torch.long, device=logits.device)

    pi_vals = [max(float(propensities.get(c, 1.0)), 1e-8) for c in supp]
    log_pi = torch.log(torch.tensor(pi_vals, dtype=logits.dtype, device=logits.device))
    z_sub = logits.index_select(0, row_t).index_select(1, col_t) + log_pi.unsqueeze(0)
    return F.cross_entropy(z_sub, y_local)


def compute_multimodal_lmnar_loss(
    feats_by_mod: dict[str, torch.Tensor],
    protos_full: torch.Tensor,
    scale: float,
    labels_global: torch.Tensor,
    seen_classes: list[int],
    graph_info: dict[str, Any],
) -> torch.Tensor:
    """Compute per-tower support-aware L_MNAR risk summed/averaged over active modalities."""
    seen = [int(c) for c in seen_classes]
    p_seen = F.normalize(protos_full[seen].float(), dim=-1)
    losses: list[torch.Tensor] = []
    for tw in ("text", "image"):
        feat = feats_by_mod.get(tw)
        if feat is None or feat.shape[0] == 0:
            continue
        u = F.normalize(feat.float(), dim=-1)
        z_tw = float(scale) * (u @ p_seen.to(u.device).T)
        loss_tw = tower_support_propensity_loss(
            z_tw,
            labels_global=labels_global,
            seen_classes=seen,
            supported_classes=graph_info["tower_supports"][tw],
            propensities=graph_info["tower_propensities"][tw],
        )
        losses.append(loss_tw)
    if not losses:
        raise ValueError("No active modalities passed to compute_multimodal_lmnar_loss")
    return torch.stack(losses).mean()


def compute_standard_ce_loss(
    feats_by_mod: dict[str, torch.Tensor],
    protos_full: torch.Tensor,
    scale: float,
    labels_global: torch.Tensor,
    seen_classes: list[int],
) -> torch.Tensor:
    """Legacy full-class ERM cross-entropy baseline on averaged modality features."""
    seen = [int(c) for c in seen_classes]
    cls_map = {c: i for i, c in enumerate(seen)}
    normed = [F.normalize(feats_by_mod[m].float(), dim=-1) for m in ("text", "image") if m in feats_by_mod]
    mean_u = torch.stack(normed).mean(dim=0)
    p_seen = F.normalize(protos_full[seen].float(), dim=-1).to(mean_u.device)
    logits = float(scale) * (mean_u @ p_seen.T)
    y_local = torch.tensor([cls_map[int(v)] for v in labels_global.detach().tolist()], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, y_local)
