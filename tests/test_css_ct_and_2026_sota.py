"""Tests for CSS-CT (O(d^2) cross-moments, O(d) weight fold, conjugate Bayes) and 2025-2026 SOTA."""

from __future__ import annotations

import math

import torch
from torch import nn

from mnarm.eval import eval_retrieval_r1, eval_view_suite
from mnarm.methods import (
    PALRLSClassifier,
    SelfNormalizedBayesTransportHead,
    StreamingClassBank,
    _identity_anchored_procrustes,
    _procrustes_from_cross_moments,
    apply_reanchor,
    fit_canonical_reanchor,
    fit_gr_clip_gap,
    gr_clip_logits,
    merge_task_deltas,
    mg_clip_eval_logits,
    verify_projection_weight_fold,
)
from mnarm.models.vlm import l2_normalize


def _make_ortho(dim: int, eps: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    sk = torch.randn(dim, dim, dtype=torch.float64, generator=g) / math.sqrt(dim)
    return torch.linalg.matrix_exp(0.5 * eps * (sk - sk.T))


def test_streaming_bank_zero_exemplar_cross_moment_exactness() -> None:
    torch.manual_seed(42)
    dim = 64
    q_rot = _make_ortho(dim, 0.3, seed=7)
    bank_small = StreamingClassBank()
    assert not hasattr(bank_small, "paired_base_t") and not hasattr(bank_small, "paired_base_i")

    pt_all, pi_all = [], []
    for n_per in (20, 400):
        bnk = StreamingClassBank()
        for cls_off in (0, 2):
            y = torch.tensor([cls_off, cls_off + 1]).repeat_interleave(n_per)
            pi = l2_normalize(torch.randn(len(y), dim))
            pt = l2_normalize(pi.double() @ q_rot.T + 0.1 * torch.randn(len(y), dim, dtype=torch.float64)).float()
            bnk.update_stage(u_expert_t=pt, u_base_t=pt, y_t=y, u_expert_i=pi, u_base_i=pi, y_i=y, paired_base_t=pt, paired_base_i=pi)
            if n_per == 20:
                bank_small = bnk
                pt_all.append(pt); pi_all.append(pi)
        if n_per == 20:
            mem_small = bnk.memory_floats()
        else:
            mem_large = bnk.memory_floats()
    assert mem_small == mem_large
    for tw in ("text", "image"):
        for mu in bank_small.class_centroids[tw].values():
            assert float(mu.norm().item()) <= 1.0 + 1e-12

    pt_cat = l2_normalize(torch.cat(pt_all, dim=0)).double()
    pi_cat = l2_normalize(torch.cat(pi_all, dim=0)).double()
    r_samp, sm_s, tm_s = _identity_anchored_procrustes(pi_cat, pt_cat, anchor_weight=1.0)
    r_mom, sm_m, tm_m = _procrustes_from_cross_moments(
        bank_small.paired_count, bank_small.paired_sum_i, bank_small.paired_sum_t, bank_small.paired_cross_ti, anchor_weight=1.0
    )
    assert float((r_samp - r_mom).abs().max().item()) < 1e-11
    assert float((sm_s - sm_m).abs().max().item()) < 1e-11
    assert float((tm_s - tm_m).abs().max().item()) < 1e-11


def test_isometric_reanchor_and_weight_fold_equivalence() -> None:
    torch.manual_seed(13)
    d_in, d_out = 24, 32
    q_drift = _make_ortho(d_out, 14.0, seed=19).float()

    class _MockCLIP:
        def __init__(self) -> None:
            self.clip = nn.Module()
            self.clip.text_projection = nn.Linear(d_in, d_out, bias=False)
            self.clip.visual_projection = nn.Linear(d_in, d_out, bias=False)
            with torch.no_grad():
                self.clip.text_projection.weight.copy_(q_drift @ self.clip.text_projection.weight)
                self.clip.visual_projection.weight.copy_(q_drift @ self.clip.visual_projection.weight)
            self.clip.get_text_features = lambda input_ids: self.clip.text_projection(input_ids.float())
            self.clip.get_image_features = lambda pixel_values: self.clip.visual_projection(pixel_values.float())
            self.device, self.is_siglip = "cpu", False

    class _MockSigLIP:
        def __init__(self) -> None:
            self.clip = nn.Module()
            self.clip.text_model = nn.Module()
            self.clip.text_model.head = nn.Linear(d_in, d_out, bias=True)
            with torch.no_grad():
                self.clip.text_model.head.weight.copy_(q_drift @ self.clip.text_model.head.weight)
                self.clip.text_model.head.bias.copy_(q_drift @ self.clip.text_model.head.bias)
            self.clip.get_text_features = lambda input_ids: self.clip.text_model.head(input_ids.float())
            self.clip.get_image_features = lambda pixel_values: pixel_values.float() @ self.clip.text_model.head.weight.T
            self.device, self.is_siglip = "cpu", True

    v_clip = _MockCLIP()
    x_t0 = l2_normalize(torch.randn(40, d_in) @ v_clip.clip.text_projection.weight.T)
    x_tm = l2_normalize(x_t0 @ q_drift.T)
    op = fit_canonical_reanchor(x_t0, x_tm, anchor_weight=0.01)
    assert op["active"] and op["cos_after"] > 0.99

    r1_before = eval_retrieval_r1(x_tm, x_t0)["i2t_r1"]
    r1_after = eval_retrieval_r1(apply_reanchor(x_tm, op), x_t0)["i2t_r1"]
    assert r1_after == 1.0 and r1_after > r1_before + 0.5

    inp = {"input_ids": torch.randn(10, d_in), "pixel_values": torch.randn(10, d_in)}
    w_before = v_clip.clip.text_projection.weight.detach().clone()
    chk_clip = verify_projection_weight_fold(v_clip, "text", op, inp)
    assert chk_clip["max_abs_diff"] < 1e-5 and chk_clip["restored"]
    assert torch.equal(v_clip.clip.text_projection.weight, w_before)

    v_sig = _MockSigLIP()
    b_before = v_sig.clip.text_model.head.bias.detach().clone()
    chk_sig_t = verify_projection_weight_fold(v_sig, "text", op, inp)
    chk_sig_i = verify_projection_weight_fold(v_sig, "image", op, inp)
    assert chk_sig_t["fold_mode"] == "exact_weight_and_bias_fold" and chk_sig_t["max_abs_diff"] < 1e-5
    assert chk_sig_i["fold_mode"] == "post_pool_orthogonal" and chk_sig_i["norm_preservation_diff"] < 1e-5
    assert torch.equal(v_sig.clip.text_model.head.bias, b_before)


def test_css_ct_adapts_across_noisy_prompt_and_101cls_fine_grained_regimes() -> None:
    torch.manual_seed(11)
    d = 96

    def _run_regime(n_cls: int, p_noise: float, cm_noise: float) -> tuple[float, float, float, float, float]:
        def _n64(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(x, dim=-1)

        cone = 0.65 * _n64(torch.randn(1, d, dtype=torch.float64))
        z_cls = _n64(cone + 0.75 * _n64(torch.randn(n_cls, d, dtype=torch.float64)))
        q_mod = _make_ortho(d, 0.25, seed=n_cls)
        gap = 0.45 * _n64(torch.randn(1, d, dtype=torch.float64))
        true_t = _n64(z_cls)
        true_i = _n64(0.80 * (z_cls @ q_mod.T) + gap + (cm_noise / math.sqrt(d)) * torch.randn(n_cls, d, dtype=torch.float64))
        protos = _n64(true_t + (p_noise / math.sqrt(d)) * torch.randn(n_cls, d, dtype=torch.float64)).float()
        late = list(range(n_cls // 2, n_cls))
        classes = list(range(n_cls))
        early = list(range(n_cls // 2))

        ix_l, iy_l, tx_l, ty_l, pt_l, pi_l = [], [], [], [], [], []
        for c in range(n_cls):
            xi = _n64(true_i[c] + (0.85 / math.sqrt(d)) * torch.randn(60, d, dtype=torch.float64)).float()
            ix_l.append(xi); iy_l.append(torch.full((60,), c, dtype=torch.long))
            if c in late:
                xt = _n64(true_t[c] + (0.85 / math.sqrt(d)) * torch.randn(60, d, dtype=torch.float64)).float()
                tx_l.append(xt); ty_l.append(torch.full((60,), c, dtype=torch.long))
                pt_l.append(xt); pi_l.append(xi)

        tr_tx, tr_ty = torch.cat(tx_l, 0), torch.cat(ty_l, 0)
        tr_ix, tr_iy = torch.cat(ix_l, 0), torch.cat(iy_l, 0)
        pt, pi = torch.cat(pt_l, 0), torch.cat(pi_l, 0)

        te_tx = torch.cat([_n64(true_t[c] + (0.85 / math.sqrt(d)) * torch.randn(25, d, dtype=torch.float64)).float() for c in range(n_cls)], 0)
        te_ix = torch.cat([_n64(true_i[c] + (0.85 / math.sqrt(d)) * torch.randn(25, d, dtype=torch.float64)).float() for c in range(n_cls)], 0)
        te_y = torch.arange(n_cls).repeat_interleave(25)

        h_css = SelfNormalizedBayesTransportHead(bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic").fit(
            tr_tx, tr_ty, tr_ix, tr_iy, pt, pi, protos, classes
        )
        h_pr = SelfNormalizedBayesTransportHead(bridge_mode="prompt_bridge", partial_cov_mode="isotropic_vmf").fit(
            tr_tx, tr_ty, tr_ix, tr_iy, pt, pi, protos, classes
        )
        h_cm = SelfNormalizedBayesTransportHead(bridge_mode="cross_modal_transport", partial_cov_mode="isotropic_vmf").fit(
            tr_tx, tr_ty, tr_ix, tr_iy, pt, pi, protos, classes
        )
        diag_t = h_css.diagnostics["towers"]["text"]
        f_css = eval_view_suite(h_css.predict_views(te_tx, te_ix), te_y, classes, early, late)["text"]["macro_f1"]
        f_pr = eval_view_suite(h_pr.predict_views(te_tx, te_ix), te_y, classes, early, late)["text"]["macro_f1"]
        f_cm = eval_view_suite(h_cm.predict_views(te_tx, te_ix), te_y, classes, early, late)["text"]["macro_f1"]
        return diag_t["w_p"], diag_t["w_cm"], f_css, f_pr, f_cm

    wp_c, wcm_c, f_css_c, f_pr_c, _ = _run_regime(8, p_noise=1.9, cm_noise=0.2)
    assert wcm_c > 5.0 * wp_c and f_css_c > f_pr_c + 0.20

    wp_f, wcm_f, f_css_f, _, f_cm_f = _run_regime(101, p_noise=0.2, cm_noise=1.5)
    assert wp_f > 5.0 * wcm_f and f_css_f > f_cm_f + 0.25


def test_iso_c_dop_gr_clip_and_pal_rls_behavior() -> None:
    torch.manual_seed(5)
    deltas = [
        {"text_model.q": torch.zeros(16, 16), "vision_model.q": torch.randn(16, 16) * 0.05},
        {"text_model.q": torch.zeros(16, 16), "vision_model.q": torch.randn(16, 16) * 0.05},
        {"text_model.q": torch.randn(16, 16) * 0.05, "vision_model.q": torch.randn(16, 16) * 0.05},
        {"text_model.q": torch.randn(16, 16) * 0.05, "vision_model.q": torch.randn(16, 16) * 0.05},
    ]
    for m in ("iso_c", "dop"):
        out = merge_task_deltas(deltas, m)
        assert all(torch.isfinite(v).all() for v in out.values())
    s_iso = torch.linalg.svdvals(merge_task_deltas(deltas, "iso_c")["vision_model.q"])
    assert float((s_iso.max() - s_iso.min()).item()) < 1e-4

    gap_op = fit_gr_clip_gap(torch.ones(10, 8), -torch.ones(10, 8))
    assert gap_op["active"] and gap_op["labels_used"] is False
    gl = gr_clip_logits(torch.randn(6, 8), torch.randn(6, 8), torch.randn(4, 8), [0, 1, 2, 3], 10.0, gap_op)
    assert set(gl.keys()) == {"text", "image", "paired"}

    pal = PALRLSClassifier(embed_dim=8, num_classes=4, lam=1.0)
    pal.update(torch.randn(12, 8), None, torch.tensor([0, 1]).repeat(6), ("text",))
    pal.update(torch.randn(12, 8), torch.randn(12, 8), torch.tensor([2, 3]).repeat(6), ("text", "image"))
    pv = pal.predict_views(torch.randn(5, 8), torch.randn(5, 8), [0, 1, 2, 3])
    assert pv["paired"].shape == (5, 4)

    class _Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.eye(8)[:4])
            self.initialized = torch.tensor([False, False, True, True])
        def forward(self, u: torch.Tensor, classes: list[int]) -> torch.Tensor:
            return torch.nn.functional.linear(torch.nn.functional.normalize(u.float(), dim=-1), torch.nn.functional.normalize(self.weight[classes], dim=-1))

    heads = {"text": _Dummy(), "image": _Dummy()}
    cross = {k: torch.zeros(3, 4) for k in ("text", "image", "paired")}
    mg = mg_clip_eval_logits(torch.randn(3, 8), torch.randn(3, 8), cross, heads, [0, 1, 2, 3], 10.0)
    assert mg["paired"].shape == (3, 4) and torch.isfinite(mg["paired"]).all()


def test_lora_nullspace_zero_leakage_and_verbatim_author_parity() -> None:
    from mnarm.eval import eval_classification_metrics

    torch.manual_seed(7)
    d, r_lora, n_tasks = 64, 4, 4
    deltas = [{"v": (torch.randn(d, r_lora) @ torch.randn(r_lora, d)) * 0.02} for _ in range(n_tasks)]

    # 1. Exact active-rank SVD truncation (zero nullspace noise leakage beyond rank 16)
    for m in ("iso_c", "iso_cts", "tsvm", "knots", "dc_lora"):
        merged = merge_task_deltas(deltas, m, rank_per_task=r_lora)["v"]
        svals = torch.linalg.svdvals(merged)
        assert float(svals[n_tasks * r_lora :].max().item()) < 1e-5, f"{m} leaked into nullspace"
        if m in ("iso_c", "iso_cts"):
            act_s = svals[: n_tasks * r_lora]
            assert float((act_s.max() - act_s.min()).item()) < 1e-4, f"{m} spectrum not isotropic"

    # 2. WUDI strictly reduces author input-span interference objective vs TA initialization
    vecs = torch.stack([d_t["v"] for d_t in deltas], dim=0)
    l2_norms = vecs.reshape(n_tasks, -1).square().sum(dim=-1)
    def _wudi_obj(w: torch.Tensor) -> float:
        inner = torch.matmul(w.unsqueeze(0) - vecs, vecs.transpose(1, 2))
        return float((inner.square() / l2_norms.unsqueeze(-1).unsqueeze(-1)).sum().item())
    ta_init = vecs.sum(dim=0)
    wudi_merged = merge_task_deltas(deltas, "wudi", rank_per_task=r_lora)["v"]
    assert _wudi_obj(wudi_merged) < _wudi_obj(ta_init)

    # 3. Verbatim GR-CLIP 3-mean centering parity (qry_mean, doc_text_mean, doc_image_mean)
    cal_t, cal_i = torch.randn(15, 8) + 1.0, torch.randn(15, 8) - 1.0
    gap_op = fit_gr_clip_gap(cal_t, cal_i)
    dt, di = torch.randn(5, 8), torch.randn(5, 8)
    protos = torch.randn(4, 8)
    seen = [0, 1, 2, 3]
    out_gr = gr_clip_logits(dt, di, protos, seen, 10.0, gap_op, alpha=0.5)
    q_ref = l2_normalize(l2_normalize(protos) - l2_normalize(protos).mean(dim=0, keepdim=True))
    dt_c = l2_normalize(dt) - l2_normalize(cal_t).mean(dim=0, keepdim=True)
    di_c = l2_normalize(di) - l2_normalize(cal_i).mean(dim=0, keepdim=True)
    assert float((out_gr["text"] - 10.0 * (l2_normalize(dt_c) @ q_ref.T)).abs().max().item()) < 1e-5
    assert float((out_gr["paired"] - 10.0 * (l2_normalize(0.5 * dt_c + 0.5 * di_c) @ q_ref.T)).abs().max().item()) < 1e-5

    # 4. Verbatim MG-CLIP inference formula: (1 * softmax(softmax(cross)) + 4 * softmax(unscaled_intra)) / 5
    class _CosHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.initialized = torch.tensor([True, True, True, True])
        def forward(self, u: torch.Tensor, classes: list[int]) -> torch.Tensor:
            return torch.nn.functional.linear(l2_normalize(u), l2_normalize(self.weight[classes]))

    heads = {"text": _CosHead(), "image": _CosHead()}
    ut, ui = torch.randn(4, 8), torch.randn(4, 8)
    cross = {"text": torch.randn(4, 4), "image": torch.randn(4, 4), "paired": torch.randn(4, 4)}
    mg_out = mg_clip_eval_logits(ut, ui, cross, heads, seen, logit_scale=50.0)
    expected_txt = torch.log((torch.softmax(torch.softmax(cross["text"], -1), -1) + 4.0 * torch.softmax(heads["text"](ut, seen), -1)) / 5.0)
    assert float((mg_out["text"] - expected_txt).abs().max().item()) < 1e-5

    # 5. Verbatim PAL-RLS ReLU expansion + closed-form ridge parity
    pal = PALRLSClassifier(embed_dim=8, num_classes=4, lam=1.0, expand_dim=32, seed=19)
    xt, xi, y = torch.randn(12, 8), torch.randn(12, 8), torch.arange(4).repeat(3)
    pal.update(xt, xi, y, ("text", "image"))
    xq = torch.randn(6, 8)
    pred = pal.predict_batch(xq, xq, ("text", "image"), seen)
    h_tr = torch.relu(torch.cat([l2_normalize(xt), l2_normalize(xi)], -1).double() @ pal.w_up)
    w_ref = torch.linalg.solve(h_tr.T @ h_tr + torch.eye(32, dtype=torch.float64), h_tr.T @ torch.nn.functional.one_hot(y, 4).double()).float()
    h_q = torch.relu(torch.cat([l2_normalize(xq), l2_normalize(xq)], -1).double() @ pal.w_up).float()
    assert float((pred - h_q @ w_ref).abs().max().item()) < 1e-5

    # 6. Sato et al. (arXiv:2609.01103) Predicted-Class Gini & Excess Gini
    y_bal = torch.arange(4).repeat(10)
    z_unif = torch.eye(4).repeat(10, 1) * 5.0
    m_unif = eval_classification_metrics(z_unif, y_bal, seen)
    assert abs(m_unif["pred_gini"]) < 1e-6 and abs(m_unif["excess_gini"]) < 1e-6
    z_hub = torch.tensor([[5.0, 0.0, 0.0, 0.0]]).repeat(40, 1)
    m_hub = eval_classification_metrics(z_hub, y_bal, seen)
    assert m_hub["pred_gini"] > 0.7 and m_hub["excess_gini"] > 0.7
