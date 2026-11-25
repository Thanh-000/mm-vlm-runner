"""Arm evaluation, CSS-CT fitting with O(d) re-anchoring, and PAL-RLS evaluation helpers."""

from __future__ import annotations

from typing import Any

import torch

from mnarm.data import PreprocessedTensorStore
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, measure_cone_drift
from mnarm.methods import (
    PALRLSClassifier,
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
    collect_train_features_by_modality,
    encode_prompts_under_payload,
    extract_paired_features_cached,
)


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
    fit4_recs: list[dict[str, Any]],
    store: PreprocessedTensorStore,
    orig_prompts: list[str],
    protos_full: torch.Tensor,
    seen_classes: list[int],
    eval_batch_size: int,
    use_amp: bool,
    shrinkage: float | None,
) -> tuple[SelfNormalizedBayesTransportHead, dict[str, Any], dict[str, Any]]:
    """Fit Pillar 2 O(d) canonical re-anchoring + Pillar 3 conjugate Bayes head."""
    f4_b = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=fit4_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp)
    f4_a = collect_train_features_by_modality(vlm=vlm, kind=kind, payload=payload, records=fit4_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp)
    protos_m = encode_prompts_under_payload(vlm=vlm, kind=kind, payload=payload, prompts=orig_prompts)
    re_t = fit_canonical_reanchor(f4_b["text_x"], f4_a["text_x"], protos_full, protos_m)
    re_i = fit_canonical_reanchor(f4_b["image_x"], f4_a["image_x"])

    head = SelfNormalizedBayesTransportHead(bridge_mode=bridge_mode, partial_cov_mode=partial_cov_mode, shrinkage=shrinkage)
    if streaming_bank is None:
        tf = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=train_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
        tx = apply_reanchor(tf["text_x"], re_t) if (use_reanchor and tf["text_x"] is not None) else tf["text_x"]
        ix = apply_reanchor(tf["image_x"], re_i) if (use_reanchor and tf["image_x"] is not None) else tf["image_x"]
        pt = apply_reanchor(tf["paired_t"], re_t) if (use_reanchor and tf["paired_t"] is not None) else tf["paired_t"]
        pi = apply_reanchor(tf["paired_i"], re_i) if (use_reanchor and tf["paired_i"] is not None) else tf["paired_i"]
        head.fit(tx, tf["text_y"], ix, tf["image_y"], pt, pi, protos_full, seen_classes, fallback_scale=vlm.logit_scale)
        return head, re_t, re_i

    um_t = apply_reanchor(f4_a["text_x"], re_t) if (use_reanchor and f4_a["text_x"] is not None) else f4_a["text_x"]
    um_i = apply_reanchor(f4_a["image_x"], re_i) if (use_reanchor and f4_a["image_x"] is not None) else f4_a["image_x"]
    pm_use = apply_reanchor(protos_m, re_t) if use_reanchor else protos_m
    bstats = streaming_bank.transport_to_target_stats(
        bridge_u0_t=f4_b["text_x"], bridge_um_t=um_t,
        bridge_u0_i=f4_b["image_x"], bridge_um_i=um_i,
        seen_classes=seen_classes, probe_y_t=f4_a["text_y"], probe_y_i=f4_a["image_y"],
        protos_0=protos_full, protos_m=pm_use,
    )
    head.fit_from_sufficient_stats(
        bstats["towers"], None, None, protos_full, seen_classes,
        fallback_scale=vlm.logit_scale, paired_moments=bstats["paired_moments"],
    )
    return head, re_t, re_i


def evaluate_arm_all_regimes(
    *,
    arm_name: str,
    kind: str,
    payload: dict[str, torch.Tensor],
    vlm: MultimodalVLM,
    orig_prompts: list[str],
    protos_full: torch.Tensor,
    unseen_protos_full: torch.Tensor,
    seen_classes: list[int],
    early_classes: list[int],
    late_classes: list[int],
    dev_rows: list[dict[str, Any]],
    cal_recs: list[dict[str, Any]],
    dev_cache: dict[tuple[str, ...], torch.Tensor],
    dev0_t: torch.Tensor,
    dev0_i: torch.Tensor,
    train_recs: list[dict[str, Any]],
    streaming_bank: StreamingClassBank | None,
    fit4_recs: list[dict[str, Any]],
    store: PreprocessedTensorStore,
    eval_batch_size: int,
    use_amp: bool,
    shrinkage: float | None,
    profiler: Profiler,
    mg_heads: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate one merged/continual arm across raw, gr_clip, unlab_center, cal_bias, and CSS-CT."""
    with profiler.track(f"eval_dev:{arm_name}", n_samples=len(dev_rows)):
        dev_t, dev_i, dev_y = extract_paired_features_cached(
            vlm=vlm, kind=kind, payload=payload, records=dev_rows,
            tensor_store=store, batch_size=eval_batch_size, cache=dev_cache, use_amp=use_amp,
        )

    cone_t, cone_i = measure_cone_drift(dev_t, dev0_t), measure_cone_drift(dev_i, dev0_i)
    zt_raw = proto_logits(dev_t, protos_full, seen_classes, vlm.logit_scale)
    zi_raw = proto_logits(dev_i, protos_full, seen_classes, vlm.logit_scale)
    cross_raw = {"text": zt_raw, "image": zi_raw, "paired": 0.5 * (zt_raw + zi_raw)}
    raw_logits = mg_clip_eval_logits(dev_t, dev_i, cross_raw, mg_heads, seen_classes, vlm.logit_scale) if mg_heads is not None else cross_raw
    regimes: dict[str, Any] = {"raw": eval_view_suite(raw_logits, dev_y, seen_classes, early_classes, late_classes)}
    boot_p: dict[str, torch.Tensor] = {"raw": raw_logits["paired"]}

    cal_sc, cal_lb, cal_fx = collect_masked_cal_bundle(
        vlm=vlm, kind=kind, payload=payload, cal_recs=cal_recs, store=store,
        protos_full=protos_full, seen_classes=seen_classes, batch_size=eval_batch_size, use_amp=use_amp,
    )
    if mg_heads is not None and cal_fx:
        native_cal = mg_clip_eval_logits(cal_fx.get("text"), cal_fx.get("image"), cal_sc, mg_heads, seen_classes, vlm.logit_scale)
        cal_sc = {vw: native_cal.get(vw, cal_sc[vw]) for vw in cal_sc}

    gap_op = fit_gr_clip_gap(cal_fx.get("text"), cal_fx.get("image"))
    gr_l = gr_clip_logits(dev_t, dev_i, protos_full, seen_classes, vlm.logit_scale, gap_op)
    regimes["gr_clip"] = eval_view_suite(gr_l, dev_y, seen_classes, early_classes, late_classes)
    boot_p["gr_clip"] = gr_l["paired"]

    if cal_sc:
        unlab_l = {vw: (apply_bias(raw_logits[vw], fit_unlabeled_center(cal_sc[vw])["bias"]) if vw in cal_sc else raw_logits[vw]) for vw in ("text", "image", "paired")}
        cal_l = {vw: (apply_bias(raw_logits[vw], fit_cal_bias(cal_sc[vw], cal_lb[vw], seen_classes)["bias"]) if vw in cal_sc else raw_logits[vw]) for vw in ("text", "image", "paired")}
        regimes["unlab_center"] = eval_view_suite(unlab_l, dev_y, seen_classes, early_classes, late_classes)
        regimes["cal_bias"] = eval_view_suite(cal_l, dev_y, seen_classes, early_classes, late_classes)
        boot_p["unlab_center"], boot_p["cal_bias"] = unlab_l["paired"], cal_l["paired"]

    with profiler.track(f"fit_zero_mnar:{arm_name}"):
        zn_head, re_t, re_i = fit_css_ct_with_reanchor(
            bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic", use_reanchor=True,
            vlm=vlm, kind=kind, payload=payload, train_recs=train_recs,
            streaming_bank=streaming_bank, fit4_recs=fit4_recs, store=store,
            orig_prompts=orig_prompts, protos_full=protos_full, seen_classes=seen_classes,
            eval_batch_size=eval_batch_size, use_amp=use_amp, shrinkage=shrinkage,
        )
    dev_t_re, dev_i_re = apply_reanchor(dev_t, re_t), apply_reanchor(dev_i, re_i)
    zn_logits = zn_head.predict_views(dev_t_re, dev_i_re)
    regimes["zero_mnar"] = {
        **eval_view_suite(zn_logits, dev_y, seen_classes, early_classes, late_classes),
        "diagnostics": zn_head.diagnostics,
        "reanchor": {"text_cos_after": re_t["cos_after"], "image_cos_after": re_i["cos_after"], "bank_floats": streaming_bank.memory_floats() if streaming_bank else 0},
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
    return {"cone_drift": {"text": cone_t, "image": cone_i}, "regimes": regimes, "encoder_preservation": enc_pres}, boot_p


def eval_pal_rls_arm(
    pal: PALRLSClassifier, dev0_t: torch.Tensor, dev0_i: torch.Tensor, dev_y: torch.Tensor,
    seen: list[int], early: list[int], late: list[int], unseen_protos: torch.Tensor, scale: float,
    cal_recs: list[dict[str, Any]], vlm: MultimodalVLM, store: PreprocessedTensorStore, eval_bs: int, use_amp: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate frozen-feature PAL-RLS analytic baseline across raw and masked cal regimes."""
    pal_raw = pal.predict_views(dev0_t, dev0_i, seen)
    regimes: dict[str, Any] = {"raw": eval_view_suite(pal_raw, dev_y, seen, early, late)}
    boot_p: dict[str, torch.Tensor] = {"raw": pal_raw["paired"]}
    vlm.restore_base_weights()
    vlm.zero_lora()
    vlm.clip.eval()
    amp_on = bool(use_amp and vlm.device == "cuda")
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    cal_sc_p: dict[str, list[torch.Tensor]] = {"text": [], "image": [], "paired": []}
    cal_lb_p: dict[str, list[torch.Tensor]] = {"text": [], "image": [], "paired": []}
    with torch.inference_mode():
        for views, batch in store.iter_batches(cal_recs, batch_size=eval_bs, shuffle=False):
            dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in batch.items() if k not in ("label", "sample_id")}
            lbl = batch["label"].clone()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                ut = vlm.encode_text(dev_inp).float().cpu() if "text" in views else None
                ui = vlm.encode_image(dev_inp).float().cpu() if "image" in views else None
            if ut is not None:
                cal_sc_p["text"].append(pal.predict_batch(ut, None, ("text",), seen)); cal_lb_p["text"].append(lbl)
            if ui is not None:
                cal_sc_p["image"].append(pal.predict_batch(None, ui, ("image",), seen)); cal_lb_p["image"].append(lbl)
            if ut is not None or ui is not None:
                cal_sc_p["paired"].append(pal.predict_batch(ut, ui, views, seen)); cal_lb_p["paired"].append(lbl)
    if cal_sc_p["paired"]:
        cal_sc = {vw: torch.cat(ps, dim=0) for vw, ps in cal_sc_p.items() if ps}
        cal_lb = {vw: torch.cat(ls, dim=0) for vw, ls in cal_lb_p.items() if ls}
        unlab_p = {vw: (apply_bias(pal_raw[vw], fit_unlabeled_center(cal_sc[vw])["bias"]) if vw in cal_sc else pal_raw[vw]) for vw in ("text", "image", "paired")}
        cal_p = {vw: (apply_bias(pal_raw[vw], fit_cal_bias(cal_sc[vw], cal_lb[vw], seen)["bias"]) if vw in cal_sc else pal_raw[vw]) for vw in ("text", "image", "paired")}
        regimes["unlab_center"] = eval_view_suite(unlab_p, dev_y, seen, early, late)
        regimes["cal_bias"] = eval_view_suite(cal_p, dev_y, seen, early, late)
        boot_p["unlab_center"], boot_p["cal_bias"] = unlab_p["paired"], cal_p["paired"]
    zt_u, zi_u = proto_logits(dev0_t, unseen_protos, seen, scale), proto_logits(dev0_i, unseen_protos, seen, scale)
    r1 = eval_retrieval_r1(dev0_t, dev0_i)
    ood_f1 = float(eval_view_suite({"text": zt_u, "image": zi_u, "paired": 0.5 * (zt_u + zi_u)}, dev_y, seen, early, late)["paired"]["macro_f1"])
    enc = {"unseen_prompt_paired_f1": ood_f1, "unseen_prompt_reanchored_f1": ood_f1, "retrieval": r1, "retrieval_reanchored": r1}
    return {"cone_drift": {"text": {"theta_deg": 0.0}, "image": {"theta_deg": 0.0}}, "regimes": regimes, "encoder_preservation": enc}, boot_p
