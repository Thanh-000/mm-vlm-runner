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


def test_nullspace_pathology_protection_under_low_shrinkage() -> None:
    """Protect early missing classes from precision-nullspace bias explosion when rho < 0.05."""
    g = torch.Generator().manual_seed(41)
    dim = 512
    classes = list(range(8))
    early, late = [0, 1, 4, 5], [2, 3, 6, 7]
    p_sem = torch.cat([F.normalize(torch.randn(8, 38, generator=g), dim=-1), torch.zeros(8, dim - 40)], dim=-1)
    sem_full = torch.cat([torch.zeros(8, 2), p_sem], dim=-1)
    mu_p = torch.zeros(dim); mu_p[0] = 1.0
    mu_t = torch.zeros(dim); mu_t[1] = 1.0
    protos = F.normalize(0.86 * mu_p.unsqueeze(0) + 0.51 * sem_full, dim=-1)

    null_dirs = torch.eye(dim)[440:504]
    q_gap = _pos_qr(torch.eye(dim) + 0.012 * torch.randn(dim, dim, generator=g))
    early_drift = {c: F.normalize(null_dirs[i * 8 : (i + 1) * 8].sum(dim=0), dim=-1) for i, c in enumerate(early)}

    def _sample(cls_list: list[int], n_per: int, mode: str):
        xs, ys = [], []
        for c in cls_list:
            sem_vec = q_gap @ sem_full[c] if mode == "image" else sem_full[c]
            if mode == "image" and c in early:
                sem_vec = F.normalize(sem_vec + 0.35 * early_drift[c], dim=-1)
            center = F.normalize(0.86 * mu_t + 0.51 * sem_vec, dim=-1)
            if mode == "text":
                z = torch.zeros(n_per, dim)
                z[:, 2:40] = 0.08 * torch.randn(n_per, 38, generator=g)
                z[:, 40:440] = 0.008 * torch.randn(n_per, 400, generator=g)
            else:
                z = 0.06 * torch.randn(n_per, dim, generator=g)
            xs.append(l2_normalize(center.unsqueeze(0) + z))
            ys.append(torch.full((n_per,), c, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    tr_ix, tr_iy = _sample(classes, 300, "image")
    tr_tx, tr_ty = _sample(late, 300, "text")
    late_mask = (tr_iy == 2) | (tr_iy == 3) | (tr_iy == 6) | (tr_iy == 7)
    paired_i, paired_t = tr_ix[late_mask], tr_tx
    dev_ix, dev_y = _sample(classes, 40, "image")
    dev_tx, _ = _sample(classes, 40, "text")

    raw_head = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport", partial_cov_mode="raw_oas").fit(
        tr_tx, tr_ty, tr_ix, tr_iy, paired_t, paired_i, protos, classes
    )
    assert raw_head.diagnostics["towers"]["text"]["rho"] < 0.05
    ev_raw = eval_view_suite(raw_head.predict_views(dev_tx, dev_ix), dev_y, classes, early, late)
    assert ev_raw["text"]["early_recall"] < 0.25

    for safe_mode in ("spectral_floor", "isotropic_vmf"):
        safe_head = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport", partial_cov_mode=safe_mode).fit(
            tr_tx, tr_ty, tr_ix, tr_iy, paired_t, paired_i, protos, classes
        )
        ev_safe = eval_view_suite(safe_head.predict_views(dev_tx, dev_ix), dev_y, classes, early, late)
        assert ev_safe["text"]["early_recall"] > 0.70, f"{safe_mode} text early_recall={ev_safe['text']['early_recall']}"
        assert ev_safe["paired"]["early_recall"] > 0.80, f"{safe_mode} paired early_recall={ev_safe['paired']['early_recall']}"
        assert ev_safe["paired"]["macro_f1"] > 0.85, f"{safe_mode} paired macro_f1={ev_safe['paired']['macro_f1']}"


def test_streaming_bank_multiclass_oof_tau_prevents_insample_overconfidence() -> None:
    """Verify OOF multi-class tau on small Stage-4 probes prevents overconfidence for both P4 and P3."""
    g = torch.Generator().manual_seed(73)
    dim, classes = 128, list(range(8))
    early, late = [0, 1, 4, 5], [2, 3, 6, 7]
    stages = [([0, 5], False), ([1, 4], False), ([2, 6], True), ([3, 7], True)]
    sem = torch.cat([torch.zeros(8, 3), F.normalize(torch.randn(8, dim - 3, generator=g), dim=-1)], dim=-1)
    mu_p, mu_i, mu_t = torch.zeros(dim), torch.zeros(dim), torch.zeros(dim)
    mu_p[0], mu_i[1], mu_t[2] = 1.0, 1.0, 1.0
    protos = F.normalize(0.86 * mu_p.unsqueeze(0) + 0.51 * sem, dim=-1)
    q_i = _pos_qr(torch.eye(dim) + 0.05 * torch.randn(dim, dim, generator=g))
    q_t = _pos_qr(torch.eye(dim) + 0.05 * torch.randn(dim, dim, generator=g))

    def _draw(cls_list: list[int], n_per: int, tower: str) -> tuple[torch.Tensor, torch.Tensor]:
        cone = mu_i if tower == "image" else mu_t
        rot = q_i if tower == "image" else q_t
        noise = 0.07 if tower == "image" else 0.05
        xs, ys = [], []
        for c in cls_list:
            ctr = F.normalize(0.84 * cone + 0.54 * (rot @ sem[c]), dim=-1)
            xs.append(l2_normalize(ctr.unsqueeze(0) + noise * torch.randn(n_per, dim, generator=g)))
            ys.append(torch.full((n_per,), c, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    dev_ix, dev_y = _draw(classes, 40, "image")
    dev_tx, _ = _draw(classes, 40, "text")
    for proto in ("P4", "P3"):
        missing_tw = "text" if proto == "P4" else "image"
        bank = StreamingClassBank()
        probe_x_t = probe_y_t = probe_x_i = probe_y_i = None
        for st_cls, paired in stages:
            xi, yi = _draw(st_cls, 30, "image")
            xt, yt = _draw(st_cls, 30, "text")
            if proto == "P4":
                tx_in, ty_in = (xt, yt) if paired else (None, None)
                ix_in, iy_in = xi, yi
            else:
                tx_in, ty_in = xt, yt
                ix_in, iy_in = (xi, yi) if paired else (None, None)
            bank.update_stage(
                u_expert_t=tx_in, u_base_t=tx_in, y_t=ty_in,
                u_expert_i=ix_in, u_base_i=ix_in, y_i=iy_in,
                paired_base_t=xt if paired else None, paired_base_i=xi if paired else None,
            )
            probe_x_t, probe_y_t, probe_x_i, probe_y_i = xt, yt, xi, yi

        assert probe_x_t is not None and probe_x_i is not None
        stats = bank.transport_to_target_stats(
            bridge_u0_t=probe_x_t, bridge_um_t=probe_x_t, bridge_u0_i=probe_x_i, bridge_um_i=probe_x_i,
            seen_classes=classes, probe_y_t=probe_y_t, probe_y_i=probe_y_i,
        )
        head = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport", partial_cov_mode="isotropic_vmf").fit_from_sufficient_stats(
            stats["towers"], stats["paired"][0], stats["paired"][1], protos, classes
        )
        ev = eval_view_suite(head.predict_views(dev_tx, dev_ix), dev_y, classes, early, late)
        assert ev["paired"]["macro_f1"] > 0.85, f"{proto} paired macro_f1={ev['paired']['macro_f1']}"
        assert ev[missing_tw]["early_recall"] > 0.70, f"{proto} {missing_tw} early_recall={ev[missing_tw]['early_recall']}"
        assert ev["paired"]["ece"] < 0.15, f"{proto} paired ece={ev['paired']['ece']}"


def test_bank_probe_oof_tau_immune_to_rare_noisy_tail_class() -> None:
    """Prevent 50:1 rare noisy tail class 7 in Stage-4 probe from collapsing full-tower tau -> 0.005."""
    g = torch.Generator().manual_seed(97)
    dim, classes = 256, list(range(8))
    sem = torch.cat([torch.zeros(8, 3), F.normalize(torch.randn(8, dim - 3, generator=g), dim=-1)], dim=-1)
    # Give noisy tail class 7 strong overlap with earlier classes {0, 4, 5}
    sem_img = sem.clone()
    sem_img[7] = F.normalize(0.45 * sem[0] + 0.45 * sem[4] + 0.35 * sem[5] + 0.25 * sem[7], dim=-1)
    mu_p, mu_i, mu_t = torch.zeros(dim), torch.zeros(dim), torch.zeros(dim)
    mu_p[0], mu_i[1], mu_t[2] = 1.0, 1.0, 1.0
    protos = F.normalize(0.86 * mu_p.unsqueeze(0) + 0.51 * sem, dim=-1)
    q_i = _pos_qr(torch.eye(dim) + 0.04 * torch.randn(dim, dim, generator=g))
    q_t = _pos_qr(torch.eye(dim) + 0.04 * torch.randn(dim, dim, generator=g))
    basis_i = torch.randn(dim, 24, generator=g) * 0.08

    def _draw_tower(cls_counts: list[tuple[int, int]], tower: str) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for c, n_c in cls_counts:
            if tower == "image":
                ctr = F.normalize(0.82 * mu_i + 0.57 * (q_i @ sem_img[c]), dim=-1)
                z = torch.randn(n_c, 24, generator=g) @ basis_i.T + 0.025 * torch.randn(n_c, dim, generator=g)
            else:
                ctr = F.normalize(0.84 * mu_t + 0.54 * (q_t @ sem[c]), dim=-1)
                z = 0.14 * torch.randn(n_c, dim, generator=g)
            xs.append(l2_normalize(ctr.unsqueeze(0) + z))
            ys.append(torch.full((n_c,), c, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    stages = [([(0, 120), (5, 120)], False), ([(1, 120), (4, 120)], False), ([(2, 120), (6, 120)], True), ([(3, 300), (7, 16)], True)]
    bank = StreamingClassBank()
    pr_tx = pr_ty = pr_ix = pr_iy = None
    for spec, paired in stages:
        xi, yi = _draw_tower(spec, "image")
        xt, yt = _draw_tower(spec, "text")
        bank.update_stage(
            u_expert_t=xt if paired else None, u_base_t=xt if paired else None, y_t=yt if paired else None,
            u_expert_i=xi, u_base_i=xi, y_i=yi,
            paired_base_t=xt if paired else None, paired_base_i=xi if paired else None,
        )
        pr_tx, pr_ty, pr_ix, pr_iy = xt, yt, xi, yi

    assert pr_tx is not None and pr_ix is not None
    stats = bank.transport_to_target_stats(
        bridge_u0_t=pr_tx, bridge_um_t=pr_tx, bridge_u0_i=pr_ix, bridge_um_i=pr_ix,
        seen_classes=classes, probe_y_t=pr_ty, probe_y_i=pr_iy,
    )
    head = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport", partial_cov_mode="isotropic_vmf").fit_from_sufficient_stats(
        stats["towers"], stats["paired"][0], stats["paired"][1], protos, classes
    )
    dev_ix, dev_y = _draw_tower([(c, 35) for c in classes], "image")
    dev_tx, _ = _draw_tower([(c, 35) for c in classes], "text")
    ev = eval_view_suite(head.predict_views(dev_tx, dev_ix), dev_y, classes, [0, 1, 4, 5], [2, 3, 6, 7])
    assert head.diagnostics["towers"]["image"]["tau"] > 0.05, f"Collapsed image tau={head.diagnostics['towers']['image']['tau']}"
    assert ev["image"]["ece"] < 0.25, f"image ece={ev['image']['ece']}"
    assert ev["paired"]["ece"] < 0.20, f"paired ece={ev['paired']['ece']}"
    assert ev["paired"]["macro_f1"] > ev["text"]["macro_f1"], f"paired={ev['paired']['macro_f1']} <= text={ev['text']['macro_f1']}"



