"""Split-independent CSS-CT / NCM / OAS-LDA fitting and fast split scoring helpers."""

from __future__ import annotations

from typing import Any

import torch

from mnarm.data import PreprocessedTensorStore
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, measure_cone_drift
from mnarm.methods import (
    SelfNormalizedBayesTransportHead,
    StreamingClassBank,
    apply_bias,
    apply_reanchor,
    fit_cal_bias,
    fit_canonical_reanchor,
    fit_gr_clip_gap,
    fit_unlabeled_center,
    gr_clip_logits,
    mg_clip_eval_logits,
)
from mnarm.models import MultimodalVLM, proto_logits
from mnarm.runners.stage_ops import (
    collect_masked_cal_bundle,
    collect_mg_clip_cal_logits,
    collect_train_features_by_modality,
    encode_prompts_under_payload,
    extract_paired_features_cached,
)

ABLATION_SPECS = (
    ("none_vmf", "none_zero_shot", "isotropic_vmf", False),
    ("prompt_vmf", "prompt_bridge", "isotropic_vmf", False),
    ("cm_raw_oas", "cross_modal_transport", "raw_oas", False),
    ("cm_sfloor", "cross_modal_transport", "spectral_floor", False),
    ("cm_vmf", "cross_modal_transport", "isotropic_vmf", False),
    ("hybrid_vmf", "hybrid_bridge", "isotropic_vmf", False),
    ("css_ct", "conjugate_bayes", "marginal_epistemic", True),
)


def _fit_head_prepared(
    *,
    bridge_mode: str,
    partial_cov_mode: str,
    use_reanchor: bool,
    protos_full: torch.Tensor,
    protos_m: torch.Tensor,
    seen_classes: list[int],
    fallback_scale: float,
    shrinkage: float | None,
    re_t: dict[str, Any],
    re_i: dict[str, Any],
    canon_bstats: dict[str, Any] | None = None,
    target_bstats: dict[str, Any] | None = None,
    tf_raw: dict[str, torch.Tensor | None] | None = None,
) -> SelfNormalizedBayesTransportHead:
    head = SelfNormalizedBayesTransportHead(bridge_mode=bridge_mode, partial_cov_mode=partial_cov_mode, shrinkage=shrinkage)
    if canon_bstats is None or target_bstats is None:
        assert tf_raw is not None
        tx = apply_reanchor(tf_raw["text_x"], re_t) if (use_reanchor and tf_raw["text_x"] is not None) else tf_raw["text_x"]
        ix = apply_reanchor(tf_raw["image_x"], re_i) if (use_reanchor and tf_raw["image_x"] is not None) else tf_raw["image_x"]
        pt = apply_reanchor(tf_raw["paired_t"], re_t) if (use_reanchor and tf_raw["paired_t"] is not None) else tf_raw["paired_t"]
        pi = apply_reanchor(tf_raw["paired_i"], re_i) if (use_reanchor and tf_raw["paired_i"] is not None) else tf_raw["paired_i"]
        head.fit(tx, tf_raw["text_y"], ix, tf_raw["image_y"], pt, pi, protos_full, seen_classes, fallback_scale=fallback_scale)
        return head
    bstats = canon_bstats if use_reanchor else target_bstats
    protos_fit = protos_full if use_reanchor else protos_m
    head.fit_from_sufficient_stats(
        bstats["towers"], None, None, protos_fit, seen_classes,
        fallback_scale=fallback_scale, paired_moments=bstats["paired_moments"],
    )
    return head


def fit_css_ct_with_reanchor(
    *,
    bridge_mode: str,
    partial_cov_mode: str,
    use_reanchor: bool,
    vlm: MultimodalVLM,
    kind: str,
    payload: dict[str, torch.Tensor],
    train_recs: list[dict[str, Any]],
    streaming_bank: StreamingClassBank | None,
    store: PreprocessedTensorStore,
    orig_prompts: list[str],
    protos_full: torch.Tensor,
    seen_classes: list[int],
    eval_batch_size: int,
    use_amp: bool,
    shrinkage: float | None,
    cal_recs: list[dict[str, Any]] | None = None,
    cal0_fx: dict[str, torch.Tensor] | None = None,
    cal_fx: dict[str, torch.Tensor] | None = None,
) -> tuple[SelfNormalizedBayesTransportHead, dict[str, Any], dict[str, Any]]:
    """Fit Pillar 2 O(d) canonical re-anchoring (from P U U_cal, 0 labels) + Pillar 3 CSS-CT in theta_0."""
    if cal0_fx is None and cal_recs:
        _, _, cal0_fx = collect_masked_cal_bundle(
            vlm=vlm, kind="lora", payload={}, cal_recs=cal_recs, store=store,
            protos_full=protos_full, seen_classes=seen_classes, batch_size=eval_batch_size, use_amp=use_amp,
        )
    if cal_fx is None and cal_recs:
        _, _, cal_fx = collect_masked_cal_bundle(
            vlm=vlm, kind=kind, payload=payload, cal_recs=cal_recs, store=store,
            protos_full=protos_full, seen_classes=seen_classes, batch_size=eval_batch_size, use_amp=use_amp,
        )
    c0, cm = cal0_fx or {}, cal_fx or {}
    u0_t, um_t, u0_i, um_i = c0.get("text"), cm.get("text"), c0.get("image"), cm.get("image")
    protos_m = encode_prompts_under_payload(vlm=vlm, kind=kind, payload=payload, prompts=orig_prompts)
    re_t, re_i = fit_canonical_reanchor(u0_t, um_t, protos_full, protos_m), fit_canonical_reanchor(u0_i, um_i)
    canon_b = target_b = tf_raw = None
    if streaming_bank is None:
        tf_raw = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=train_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
    elif use_reanchor:
        canon_b = target_b = streaming_bank.canonical_stats(seen_classes)
    else:
        canon_b = streaming_bank.transport_to_target_stats(
            bridge_u0_t=u0_t, bridge_um_t=um_t, bridge_u0_i=u0_i, bridge_um_i=um_i,
            seen_classes=seen_classes, protos_0=protos_full, protos_m=protos_m,
        )
        target_b = canon_b
    head = _fit_head_prepared(
        bridge_mode=bridge_mode, partial_cov_mode=partial_cov_mode, use_reanchor=use_reanchor,
        protos_full=protos_full, protos_m=protos_m, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
        shrinkage=shrinkage, re_t=re_t, re_i=re_i, canon_bstats=canon_b, target_bstats=target_b, tf_raw=tf_raw,
    )
    return head, re_t, re_i


def fit_arm_bundle(
    *,
    arm_name: str,
    kind: str,
    payload: dict[str, torch.Tensor],
    vlm: MultimodalVLM,
    orig_prompts: list[str],
    protos_full: torch.Tensor,
    seen_classes: list[int],
    cal_recs: list[dict[str, Any]],
    train_recs: list[dict[str, Any]],
    streaming_bank: StreamingClassBank | None,
    cal0_fx: dict[str, torch.Tensor] | None,
    store: PreprocessedTensorStore,
    eval_batch_size: int,
    use_amp: bool,
    shrinkage: float | None,
    profiler: Profiler,
    mg_heads: dict[str, Any] | None = None,
    cal_cache: dict[tuple[str, ...], torch.Tensor] | None = None,
    include_ablations: bool = False,
) -> dict[str, Any]:
    """Fit all split-independent calibration biases, re-anchoring operators, and heads ONCE per arm."""
    protos_m = encode_prompts_under_payload(vlm=vlm, kind=kind, payload=payload, prompts=orig_prompts)
    protos_raw = protos_m if mg_heads is not None else protos_full
    cal_sc, cal_lb, cal_fx = collect_masked_cal_bundle(
        vlm=vlm, kind=kind, payload=payload, cal_recs=cal_recs, store=store,
        protos_full=protos_raw, seen_classes=seen_classes, batch_size=eval_batch_size, use_amp=use_amp, cal_cache=cal_cache,
    )
    if mg_heads is not None and cal_recs:
        cal_sc = collect_mg_clip_cal_logits(
            vlm=vlm, kind=kind, payload=payload, cal_recs=cal_recs, store=store,
            orig_prompts=orig_prompts, seen_classes=seen_classes, mg_heads=mg_heads,
            batch_size=eval_batch_size, use_amp=use_amp, cal_cache=cal_cache, protos_cur=protos_m,
        )
    gap_op = fit_gr_clip_gap(cal_fx.get("text"), cal_fx.get("image"))
    unlab_b = {vw: fit_unlabeled_center(cal_sc[vw])["bias"] for vw in cal_sc} if cal_sc else {}
    cal_b = {vw: fit_cal_bias(cal_sc[vw], cal_lb[vw], seen_classes)["bias"] for vw in cal_sc} if cal_sc else {}

    c0 = cal0_fx or {}
    u0_t, um_t, u0_i, um_i = c0.get("text"), cal_fx.get("text"), c0.get("image"), cal_fx.get("image")
    re_t, re_i = fit_canonical_reanchor(u0_t, um_t, protos_full, protos_m), fit_canonical_reanchor(u0_i, um_i)
    canon_b = target_b = tf_raw = None
    if streaming_bank is None:
        tf_raw = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=train_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
    else:
        canon_b = streaming_bank.canonical_stats(seen_classes)
        target_b = streaming_bank.transport_to_target_stats(
            bridge_u0_t=u0_t, bridge_um_t=um_t, bridge_u0_i=u0_i, bridge_um_i=um_i,
            seen_classes=seen_classes, protos_0=protos_full, protos_m=protos_m,
        )

    kw_fit = dict(
        protos_full=protos_full, protos_m=protos_m, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
        shrinkage=shrinkage, re_t=re_t, re_i=re_i, canon_bstats=canon_b, target_bstats=target_b, tf_raw=tf_raw,
    )
    ncm_head = _fit_head_prepared(bridge_mode="none_zero_shot", partial_cov_mode="isotropic_vmf", use_reanchor=False, **kw_fit)
    oas_head = _fit_head_prepared(bridge_mode="none_zero_shot", partial_cov_mode="raw_oas", use_reanchor=False, **kw_fit)
    with profiler.track(f"fit_zero_mnar:{arm_name}"):
        zn_head = _fit_head_prepared(bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic", use_reanchor=True, **kw_fit)

    ab_heads = {
        ab_name: (zn_head if use_re and bmode == "conjugate_bayes" else _fit_head_prepared(bridge_mode=bmode, partial_cov_mode=pcmode, use_reanchor=use_re, **kw_fit), use_re)
        for ab_name, bmode, pcmode, use_re in ABLATION_SPECS
    } if include_ablations else {}

    return {
        "protos_full": protos_full, "protos_raw": protos_raw, "gap_op": gap_op,
        "unlab_biases": unlab_b, "cal_biases": cal_b, "re_t": re_t, "re_i": re_i,
        "ncm_head": ncm_head, "oas_head": oas_head, "zn_head": zn_head,
        "bank_floats": streaming_bank.memory_floats() if streaming_bank else 0,
        "ablation_heads": ab_heads,
    }


def score_arm_on_split(
    fitted: dict[str, Any], *, arm_name: str, kind: str, payload: dict[str, torch.Tensor], vlm: MultimodalVLM,
    unseen_protos_full: torch.Tensor, seen_classes: list[int], early_classes: list[int], late_classes: list[int],
    eval_rows: list[dict[str, Any]], eval_cache: dict[tuple[str, ...], torch.Tensor], eval0_t: torch.Tensor, eval0_i: torch.Tensor,
    store: PreprocessedTensorStore, eval_batch_size: int, use_amp: bool, profiler: Profiler, mg_heads: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Score one fitted arm bundle against one evaluation split (dev or test)."""
    with profiler.track(f"eval_dev:{arm_name}", n_samples=len(eval_rows)):
        dev_t, dev_i, dev_y = extract_paired_features_cached(
            vlm=vlm, kind=kind, payload=payload, records=eval_rows, tensor_store=store,
            batch_size=eval_batch_size, cache=eval_cache, use_amp=use_amp,
        )
    protos_full, protos_raw = fitted["protos_full"], fitted["protos_raw"]
    zt_raw, zi_raw = proto_logits(dev_t, protos_raw, seen_classes, vlm.logit_scale), proto_logits(dev_i, protos_raw, seen_classes, vlm.logit_scale)
    cross_raw = {"text": zt_raw, "image": zi_raw, "paired": 0.5 * (zt_raw + zi_raw)}
    raw_logits = mg_clip_eval_logits(dev_t, dev_i, cross_raw, mg_heads, seen_classes, vlm.logit_scale) if mg_heads is not None else cross_raw
    regimes: dict[str, Any] = {"raw": eval_view_suite(raw_logits, dev_y, seen_classes, early_classes, late_classes)}
    boot_p: dict[str, torch.Tensor] = {"raw": raw_logits["paired"]}

    gr_l = gr_clip_logits(dev_t, dev_i, protos_full, seen_classes, vlm.logit_scale, fitted["gap_op"])
    regimes["gr_clip"] = eval_view_suite(gr_l, dev_y, seen_classes, early_classes, late_classes)
    boot_p["gr_clip"] = gr_l["paired"]

    for f_name, f_h in (("ncm", fitted["ncm_head"]), ("oas_lda", fitted["oas_head"])):
        f_l = f_h.predict_views(dev_t, dev_i)
        regimes[f_name], boot_p[f_name] = eval_view_suite(f_l, dev_y, seen_classes, early_classes, late_classes), f_l["paired"]

    if fitted["unlab_biases"] and fitted["cal_biases"]:
        unlab_b, cal_b = fitted["unlab_biases"], fitted["cal_biases"]
        unlab_l = {vw: (apply_bias(raw_logits[vw], unlab_b[vw]) if vw in unlab_b else raw_logits[vw]) for vw in ("text", "image", "paired")}
        cal_l = {vw: (apply_bias(raw_logits[vw], cal_b[vw]) if vw in cal_b else raw_logits[vw]) for vw in ("text", "image", "paired")}
        regimes["unlab_center"] = eval_view_suite(unlab_l, dev_y, seen_classes, early_classes, late_classes)
        regimes["cal_bias"] = eval_view_suite(cal_l, dev_y, seen_classes, early_classes, late_classes)
        boot_p["unlab_center"], boot_p["cal_bias"] = unlab_l["paired"], cal_l["paired"]

    re_t, re_i = fitted["re_t"], fitted["re_i"]
    dev_t_re, dev_i_re = apply_reanchor(dev_t, re_t), apply_reanchor(dev_i, re_i)
    zn_logits = fitted["zn_head"].predict_views(dev_t_re, dev_i_re)
    regimes["zero_mnar"] = {
        **eval_view_suite(zn_logits, dev_y, seen_classes, early_classes, late_classes),
        "diagnostics": fitted["zn_head"].diagnostics,
        "reanchor": {"text_cos_after": re_t["cos_after"], "image_cos_after": re_i["cos_after"], "bank_floats": fitted["bank_floats"]},
    }
    boot_p["zero_mnar"] = zn_logits["paired"]

    zt_u, zi_u = proto_logits(dev_t, unseen_protos_full, seen_classes, vlm.logit_scale), proto_logits(dev_i, unseen_protos_full, seen_classes, vlm.logit_scale)
    zt_ur, zi_ur = proto_logits(dev_t_re, unseen_protos_full, seen_classes, vlm.logit_scale), proto_logits(dev_i_re, unseen_protos_full, seen_classes, vlm.logit_scale)
    enc_pres = {
        "unseen_prompt_paired_f1": float(eval_view_suite({"text": zt_u, "image": zi_u, "paired": 0.5 * (zt_u + zi_u)}, dev_y, seen_classes, early_classes, late_classes)["paired"]["macro_f1"]),
        "unseen_prompt_reanchored_f1": float(eval_view_suite({"text": zt_ur, "image": zi_ur, "paired": 0.5 * (zt_ur + zi_ur)}, dev_y, seen_classes, early_classes, late_classes)["paired"]["macro_f1"]),
        "retrieval": eval_retrieval_r1(dev_t, dev_i),
        "retrieval_reanchored": eval_retrieval_r1(dev_t_re, dev_i_re),
    }
    out_arm: dict[str, Any] = {"cone_drift": {"text": measure_cone_drift(dev_t, eval0_t), "image": measure_cone_drift(dev_i, eval0_i)}, "regimes": regimes, "encoder_preservation": enc_pres}
    if mg_heads is not None and getattr(mg_heads.get("text"), "mgp_trace", None):
        out_arm["mgp_trace"] = dict(mg_heads["text"].mgp_trace)
    return out_arm, boot_p
