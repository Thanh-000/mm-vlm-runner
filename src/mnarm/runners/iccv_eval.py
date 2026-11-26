"""Arm evaluation, CSS-CT fitting with O(d) re-anchoring, and PAL-RLS evaluation helpers."""

from __future__ import annotations

from typing import Any

import torch

from mnarm.data import PreprocessedTensorStore
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, measure_cone_drift, paired_bootstrap_diff
from mnarm.methods import (
    ALL_MERGE_METHODS,
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
    merge_task_deltas,
    mg_clip_eval_logits,
)
from mnarm.models import MultimodalVLM, proto_logits
from mnarm.runners.stage_ops import (
    collect_masked_cal_bundle,
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
    u0_t, um_t = c0.get("text"), cm.get("text")
    u0_i, um_i = c0.get("image"), cm.get("image")

    protos_m = encode_prompts_under_payload(vlm=vlm, kind=kind, payload=payload, prompts=orig_prompts)
    re_t = fit_canonical_reanchor(u0_t, um_t, protos_full, protos_m)
    re_i = fit_canonical_reanchor(u0_i, um_i)

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

    if use_reanchor:
        bstats = streaming_bank.canonical_stats(seen_classes)
        protos_fit = protos_full
    else:
        bstats = streaming_bank.transport_to_target_stats(
            bridge_u0_t=u0_t, bridge_um_t=um_t, bridge_u0_i=u0_i, bridge_um_i=um_i,
            seen_classes=seen_classes, protos_0=protos_full, protos_m=protos_m,
        )
        protos_fit = protos_m
    head.fit_from_sufficient_stats(
        bstats["towers"], None, None, protos_fit, seen_classes,
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
    cal0_fx: dict[str, torch.Tensor] | None,
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
            streaming_bank=streaming_bank, store=store,
            orig_prompts=orig_prompts, protos_full=protos_full, seen_classes=seen_classes,
            eval_batch_size=eval_batch_size, use_amp=use_amp, shrinkage=shrinkage,
            cal_recs=cal_recs, cal0_fx=cal0_fx, cal_fx=cal_fx,
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
    out_arm: dict[str, Any] = {"cone_drift": {"text": cone_t, "image": cone_i}, "regimes": regimes, "encoder_preservation": enc_pres}
    if mg_heads is not None and getattr(mg_heads.get("text"), "mgp_trace", None):
        out_arm["mgp_trace"] = dict(mg_heads["text"].mgp_trace)
    return out_arm, boot_p


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


def evaluate_protocol_split(
    *,
    config: dict[str, Any], ds: dict[str, Any], eval_rows: list[dict[str, Any]], split_name: str,
    seen: list[int], exp_deltas: list[dict[str, torch.Tensor]], bank: StreamingClassBank,
    seq_ft_state: dict[str, torch.Tensor] | None, seq_bank: StreamingClassBank,
    seq_lmnar_state: dict[str, torch.Tensor] | None, seq_lmnar_bank: StreamingClassBank,
    seq_rep_state: dict[str, torch.Tensor] | None, seq_rep_bank: StreamingClassBank,
    mg_state: dict[str, torch.Tensor] | None, mg_bank: StreamingClassBank, mg_heads: dict[str, Any],
    joint_state: dict[str, torch.Tensor], joint_recs: list[dict[str, Any]], pal: PALRLSClassifier,
    cal0_fx: dict[str, torch.Tensor], vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, feat_cache: dict[tuple[str, ...], torch.Tensor], eff_eval_bs: int, use_amp: bool,
) -> dict[str, Any]:
    """Evaluate all 14 mergers + continual/SOTA tracks + ablations + bootstrap on one split (dev or test)."""
    profiler, seed = Profiler(), int(config["seed"])
    st_last = ds["stages"][-1]
    cal_recs = st_last["calibration_latest_views"]
    lora_rank_eff = int(getattr(vlm, "rank", config.get("lora_rank", 8)))
    eval0_t, eval0_i, eval_y = extract_paired_features_cached(
        vlm=vlm, kind="lora", payload={}, records=eval_rows, tensor_store=store,
        batch_size=eff_eval_bs, cache=feat_cache, use_amp=use_amp,
    )
    arms_out: dict[str, Any] = {}
    boot_map: dict[str, dict[str, torch.Tensor]] = {}
    tsvm_delta: dict[str, torch.Tensor] | None = None

    def _eval_one(aname: str, akind: str, apayload: dict[str, torch.Tensor], tr_recs: list[dict[str, Any]], bnk: StreamingClassBank | None, mgh: dict[str, Any] | None) -> None:
        arec, blogits = evaluate_arm_all_regimes(
            arm_name=aname, kind=akind, payload=apayload, vlm=vlm, orig_prompts=ds["orig_prompts"], protos_full=protos_full,
            unseen_protos_full=unseen_protos_full, seen_classes=seen, early_classes=ds["early_classes"], late_classes=ds["late_classes"],
            dev_rows=eval_rows, cal_recs=cal_recs, dev_cache=feat_cache, dev0_t=eval0_t, dev0_i=eval0_i, train_recs=tr_recs, streaming_bank=bnk,
            cal0_fx=cal0_fx, store=store, eval_batch_size=eff_eval_bs, use_amp=use_amp, shrinkage=config.get("shrinkage"), profiler=profiler, mg_heads=mgh,
        )
        arms_out[aname], boot_map[aname] = arec, blogits

    for m in ALL_MERGE_METHODS:
        with profiler.track(f"merge:{m}"):
            delta_m = merge_task_deltas(exp_deltas, m, rank_per_task=lora_rank_eff)
        _eval_one(m, "delta", delta_m, [], bank, None)
        if m == "tsvm":
            tsvm_delta = delta_m
        else:
            del delta_m

    traj_specs: list[tuple[str, str, dict[str, torch.Tensor], list[dict[str, Any]], StreamingClassBank | None, dict[str, Any] | None]] = [
        ("seq_ft", "lora", seq_ft_state or {}, [], seq_bank, None),
        ("seq_lmnar", "lora", seq_lmnar_state or {}, [], seq_lmnar_bank, None),
        ("seq_replay256", "lora", seq_rep_state or {}, [], seq_rep_bank, None),
        *([("mg_clip", "lora", mg_state, [], mg_bank, mg_heads)] if bool(config.get("include_mg_clip", True)) and mg_state is not None else []),
        ("joint_oracle", "lora", joint_state, joint_recs, None, None),
    ]
    for spec in traj_specs:
        _eval_one(*spec)

    pal_rec, pal_boot = eval_pal_rls_arm(pal, eval0_t, eval0_i, eval_y, seen, ds["early_classes"], ds["late_classes"], unseen_protos_full, vlm.logit_scale, cal_recs, vlm, store, eff_eval_bs, use_amp)
    arms_out["pal_rls"], boot_map["pal_rls"] = pal_rec, pal_boot
    tsvm_delta = tsvm_delta if tsvm_delta is not None else merge_task_deltas(exp_deltas, "tsvm", rank_per_task=lora_rank_eff)
    tsvm_t, tsvm_i, _ = extract_paired_features_cached(vlm=vlm, kind="delta", payload=tsvm_delta, records=eval_rows, tensor_store=store, batch_size=eff_eval_bs, cache=feat_cache, use_amp=use_amp)
    _, _, cal_tsvm_fx = collect_masked_cal_bundle(vlm=vlm, kind="delta", payload=tsvm_delta, cal_recs=cal_recs, store=store, protos_full=protos_full, seen_classes=seen, batch_size=eff_eval_bs, use_amp=use_amp)
    ablation_out: dict[str, Any] = {}
    for ab_name, bmode, pcmode, use_re in ABLATION_SPECS:
        ab_h, rt_ab, ri_ab = fit_css_ct_with_reanchor(
            bridge_mode=bmode, partial_cov_mode=pcmode, use_reanchor=use_re, vlm=vlm, kind="delta", payload=tsvm_delta, train_recs=[],
            streaming_bank=bank, store=store, orig_prompts=ds["orig_prompts"], protos_full=protos_full, seen_classes=seen,
            eval_batch_size=eff_eval_bs, use_amp=use_amp, shrinkage=config.get("shrinkage"), cal_recs=cal_recs, cal0_fx=cal0_fx, cal_fx=cal_tsvm_fx,
        )
        tx_ab, ix_ab = (apply_reanchor(tsvm_t, rt_ab), apply_reanchor(tsvm_i, ri_ab)) if use_re else (tsvm_t, tsvm_i)
        ablation_out[ab_name] = eval_view_suite(ab_h.predict_views(tx_ab, ix_ab), eval_y, seen, ds["early_classes"], ds["late_classes"])

    anchor = boot_map["tsvm"]["zero_mnar"]
    boot_refs = [
        ("tsvm:zero_mnar_vs_tsvm:raw", boot_map["tsvm"]["raw"]), ("tsvm:zero_mnar_vs_tsvm:gr_clip", boot_map["tsvm"].get("gr_clip")),
        ("tsvm:zero_mnar_vs_tsvm:cal_bias", boot_map["tsvm"].get("cal_bias")), ("tsvm:zero_mnar_vs_knots:cal_bias", boot_map["knots"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_iso_c:cal_bias", boot_map["iso_c"].get("cal_bias")), ("tsvm:zero_mnar_vs_iso_cts:cal_bias", boot_map["iso_cts"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_wudi:cal_bias", boot_map["wudi"].get("cal_bias")), ("tsvm:zero_mnar_vs_dop:cal_bias", boot_map["dop"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_dc_lora:cal_bias", boot_map["dc_lora"].get("cal_bias")), ("tsvm:zero_mnar_vs_seq_lmnar:raw", boot_map["seq_lmnar"].get("raw")),
        ("tsvm:zero_mnar_vs_mg_clip:raw", boot_map.get("mg_clip", {}).get("raw")), ("tsvm:zero_mnar_vs_pal_rls:raw", boot_map["pal_rls"].get("raw")),
    ]
    bootstrap = {lbl: paired_bootstrap_diff(anchor, ref, eval_y, seen, seed=seed) for lbl, ref in boot_refs if ref is not None}
    return {
        "schema": "mnarm-iccv-protocol-v2", "split": split_name, "config": config, "arms": arms_out,
        "ablation_tsvm": ablation_out, "bootstrap": bootstrap, "theta0_preservation": dict(pal_rec["encoder_preservation"]), "profiling": profiler.records,
    }
