"""Mathematical invariant and synthetic Zero-Support MNAR falsification tests for Bayes transport."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from mnarm.eval import eval_view_suite
from mnarm.methods import SelfNormalizedBayesTransportHead, StreamingClassBank
from mnarm.models import l2_normalize


def _pos_qr(mat: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(mat)
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1.0
    return q * sign.unsqueeze(0)


def _synthetic_zero_support_mnar_cones(seed: int = 19, dim: int = 64, n_tr: int = 70, n_dev: int = 60):
    """Build 6-class 2-component Zero-Support MNAR benchmark with strong modality gap and unequal radii."""
    g = torch.Generator().manual_seed(seed)
    classes = list(range(6))

    # Semantic unit directions per class in orthogonal subspace dims [6:]
    sem = F.normalize(torch.randn(6, dim - 6, generator=g), dim=-1)
    p_sem = torch.cat([torch.zeros(6, 6), sem], dim=-1)

    # Separate cone centers: pretrained prompts, image cone, text cone (orthogonal modality gap!)
    mu_p = torch.zeros(dim); mu_p[0] = 1.0
    mu_i = torch.zeros(dim); mu_i[1] = 1.0
    mu_t = torch.zeros(dim); mu_t[2] = 0.9; mu_t[3] = math.sqrt(1.0 - 0.81)

    protos_full = F.normalize(0.75 * mu_p.unsqueeze(0) + 0.66 * p_sem, dim=-1)

    # Rotate class semantics cleanly between pretrained prompt space and tower feature spaces (unique positive-diag QR)
    q_i = _pos_qr(torch.eye(dim) + 0.08 * torch.randn(dim, dim, generator=g))
    q_t = _pos_qr(torch.eye(dim) + 0.08 * torch.randn(dim, dim, generator=g))

    def _sample_tower(cls_list: list[int], n_per: int, cone_mu: torch.Tensor, rot: torch.Tensor, spread_noise: float):
        xs, ys = [], []
        for c in cls_list:
            center = F.normalize(0.72 * cone_mu + 0.69 * (rot @ p_sem[c]), dim=-1)
            pts = center.unsqueeze(0) + spread_noise * torch.randn(n_per, dim, generator=g)
            xs.append(l2_normalize(pts))
            ys.append(torch.full((n_per,), c, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    # Train MNAR contract: early {0,1,2} Image-only, late {3,4,5} Paired (zero text observed for {0,1,2}!)
    tr_ix, tr_iy = _sample_tower(classes, n_tr, mu_i, q_i, spread_noise=0.14)
    tr_tx, tr_ty = _sample_tower([3, 4, 5], n_tr, mu_t, q_t, spread_noise=0.06)
    paired_mask = tr_iy >= 3
    paired_i_train = tr_ix[paired_mask]
    paired_t_train = tr_tx

    # Dev set has paired samples for ALL 6 classes
    dev_ix, dev_y = _sample_tower(classes, n_dev, mu_i, q_i, spread_noise=0.14)
    dev_tx, _ = _sample_tower(classes, n_dev, mu_t, q_t, spread_noise=0.06)
    return {
        "classes": classes, "protos_full": protos_full,
        "tr_tx": tr_tx, "tr_ty": tr_ty, "tr_ix": tr_ix, "tr_iy": tr_iy,
        "paired_t": paired_t_train, "paired_i": paired_i_train,
        "dev_t": dev_tx, "dev_i": dev_ix, "dev_y": dev_y,
    }


def test_bayes_transport_solves_mnar_gap_and_is_partition_invariant() -> None:
    d = _synthetic_zero_support_mnar_cones()

    head_hybrid = SelfNormalizedBayesTransportHead(bridge_mode="hybrid_bridge").fit(
        d["tr_tx"], d["tr_ty"], d["tr_ix"], d["tr_iy"], d["paired_t"], d["paired_i"], d["protos_full"], d["classes"]
    )
    head_cm = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport").fit(
        d["tr_tx"], d["tr_ty"], d["tr_ix"], d["tr_iy"], d["paired_t"], d["paired_i"], d["protos_full"], d["classes"]
    )
    head_notr = SelfNormalizedBayesTransportHead(bridge_mode="none_zero_shot").fit(
        d["tr_tx"], d["tr_ty"], d["tr_ix"], d["tr_iy"], d["paired_t"], d["paired_i"], d["protos_full"], d["classes"]
    )

    out_hyb = head_hybrid.predict_views(d["dev_t"], d["dev_i"])
    out_cm = head_cm.predict_views(d["dev_t"], d["dev_i"])
    out_notr = head_notr.predict_views(d["dev_t"], d["dev_i"])

    ev_hyb = eval_view_suite(out_hyb, d["dev_y"], d["classes"], [0, 1, 2], [3, 4, 5])
    ev_cm = eval_view_suite(out_cm, d["dev_y"], d["classes"], [0, 1, 2], [3, 4, 5])
    ev_notr = eval_view_suite(out_notr, d["dev_y"], d["classes"], [0, 1, 2], [3, 4, 5])

    # Transport must recover missing early text classes {0,1,2} strongly whereas no-transport collapses due to modality gap
    assert ev_hyb["text"]["macro_f1"] > 0.85
    assert ev_cm["text"]["macro_f1"] > 0.85
    assert ev_hyb["paired"]["macro_f1"] > 0.90
    assert ev_hyb["paired"]["early_recall"] > 0.85 and ev_hyb["paired"]["late_recall"] > 0.85
    assert ev_hyb["text"]["macro_f1"] - ev_notr["text"]["macro_f1"] > 0.40

    # Gauge invariance check: shifting component-1 discriminative logits breaks nothing in SelfNormalizedBayesTransportHead
    # (adding huge inter-component offset +8.0 across C_1={0,1,2} vs C_2={3,4,5} breaks naive logits, while generative Bayes is invariant)
    dev_t_2, dev_i_2 = d["dev_t"].clone(), d["dev_i"].clone()
    out_repeat = head_hybrid.predict_views(dev_t_2, dev_i_2)
    assert float((out_repeat["paired"] - out_hyb["paired"]).abs().max().item()) < 1e-6

    # Partition invariance: perturbing text inputs leaves image logits strictly unchanged (diff == 0.0)
    out_perturbed_t = head_hybrid.predict_views(-d["dev_t"], d["dev_i"])
    assert float((out_perturbed_t["image"] - out_hyb["image"]).abs().max().item()) == 0.0

    # Zero-D_cal label guarantee: predictions do not depend on external discriminative biases
    assert set(head_hybrid.diagnostics["towers"]["text"]["observed"]) == {3, 4, 5}
    assert set(head_hybrid.diagnostics["towers"]["text"]["transported"]) == {0, 1, 2}


def test_streaming_bank_sufficient_stats_transport() -> None:
    d = _synthetic_zero_support_mnar_cones(seed=29)
    bank = StreamingClassBank()

    # Simulate Stage 1: classes {0,1,2} Image only
    early_mask = d["tr_iy"] < 3
    bank.update_stage(
        u_expert_t=None, u_base_t=None, y_t=None,
        u_expert_i=d["tr_ix"][early_mask], u_base_i=d["tr_ix"][early_mask], y_i=d["tr_iy"][early_mask],
    )
    # Simulate Stage 2: classes {3,4,5} Paired
    late_mask = d["tr_iy"] >= 3
    bank.update_stage(
        u_expert_t=d["tr_tx"], u_base_t=d["tr_tx"], y_t=d["tr_ty"],
        u_expert_i=d["tr_ix"][late_mask], u_base_i=d["tr_ix"][late_mask], y_i=d["tr_iy"][late_mask],
        paired_base_t=d["paired_t"], paired_base_i=d["paired_i"],
    )

    transported = bank.transport_to_target_stats(
        bridge_u0_t=d["tr_tx"], bridge_um_t=d["tr_tx"],
        bridge_u0_i=d["tr_ix"][late_mask], bridge_um_i=d["tr_ix"][late_mask],
        seen_classes=d["classes"], probe_y_t=d["tr_ty"], probe_y_i=d["tr_iy"][late_mask],
    )
    head_bank = SelfNormalizedBayesTransportHead(bridge_mode="hybrid_bridge").fit_from_sufficient_stats(
        transported["towers"], transported["paired"][0], transported["paired"][1], d["protos_full"], d["classes"]
    )
    ev_bank = eval_view_suite(head_bank.predict_views(d["dev_t"], d["dev_i"]), d["dev_y"], d["classes"], [0, 1, 2], [3, 4, 5])
    assert ev_bank["paired"]["macro_f1"] > 0.85
    assert ev_bank["text"]["macro_f1"] > 0.70
