"""Arm evaluation orchestration across dev/test splits with shared lossless calibration & fits."""

from __future__ import annotations

from typing import Any

import torch

from mnarm.data import PreprocessedTensorStore
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, paired_bootstrap_diff
from mnarm.methods import (
    ALL_MERGE_METHODS,
    PALRLSClassifier,
    StreamingClassBank,
    apply_bias,
    apply_reanchor,
    fit_cal_bias,
    fit_unlabeled_center,
    merge_task_deltas,
)
from mnarm.models import MultimodalVLM, proto_logits
from mnarm.runners.iccv_arm_fit import (
    ABLATION_SPECS,
    fit_arm_bundle,
    fit_css_ct_with_reanchor,
    score_arm_on_split,
)
from mnarm.runners.stage_ops import extract_paired_features_cached


def evaluate_arm_all_regimes(
    *,
    arm_name: str, kind: str, payload: dict[str, torch.Tensor], vlm: MultimodalVLM,
    orig_prompts: list[str], protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    seen_classes: list[int], early_classes: list[int], late_classes: list[int],
    dev_rows: list[dict[str, Any]], cal_recs: list[dict[str, Any]], dev_cache: dict[tuple[str, ...], torch.Tensor],
    dev0_t: torch.Tensor, dev0_i: torch.Tensor, train_recs: list[dict[str, Any]],
    streaming_bank: StreamingClassBank | None, cal0_fx: dict[str, torch.Tensor] | None,
    store: PreprocessedTensorStore, eval_batch_size: int, use_amp: bool,
    shrinkage: float | None, profiler: Profiler, mg_heads: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate one merged/continual arm across all regimes on a single split."""
    fitted = fit_arm_bundle(
        arm_name=arm_name, kind=kind, payload=payload, vlm=vlm, orig_prompts=orig_prompts,
        protos_full=protos_full, seen_classes=seen_classes, cal_recs=cal_recs, train_recs=train_recs,
        streaming_bank=streaming_bank, cal0_fx=cal0_fx, store=store, eval_batch_size=eval_batch_size,
        use_amp=use_amp, shrinkage=shrinkage, profiler=profiler, mg_heads=mg_heads,
    )
    return score_arm_on_split(
        fitted, arm_name=arm_name, kind=kind, payload=payload, vlm=vlm, unseen_protos_full=unseen_protos_full,
        seen_classes=seen_classes, early_classes=early_classes, late_classes=late_classes,
        eval_rows=dev_rows, eval_cache=dev_cache, eval0_t=dev0_t, eval0_i=dev0_i, store=store,
        eval_batch_size=eval_batch_size, use_amp=use_amp, profiler=profiler, mg_heads=mg_heads,
    )


def _fit_pal_cal_biases(
    pal: PALRLSClassifier, seen: list[int], cal_recs: list[dict[str, Any]],
    vlm: MultimodalVLM, store: PreprocessedTensorStore, eval_bs: int, use_amp: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not cal_recs:
        return {}, {}
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
    if not cal_sc_p["paired"]:
        return {}, {}
    cal_sc = {vw: torch.cat(ps, dim=0) for vw, ps in cal_sc_p.items() if ps}
    cal_lb = {vw: torch.cat(ls, dim=0) for vw, ls in cal_lb_p.items() if ls}
    return (
        {vw: fit_unlabeled_center(cal_sc[vw])["bias"] for vw in cal_sc},
        {vw: fit_cal_bias(cal_sc[vw], cal_lb[vw], seen)["bias"] for vw in cal_sc},
    )


def _score_pal_rls_on_split(
    pal: PALRLSClassifier, dev0_t: torch.Tensor, dev0_i: torch.Tensor, dev_y: torch.Tensor,
    seen: list[int], early: list[int], late: list[int], unseen_protos: torch.Tensor, scale: float,
    unlab_b: dict[str, torch.Tensor], cal_b: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    pal_raw = pal.predict_views(dev0_t, dev0_i, seen)
    regimes: dict[str, Any] = {"raw": eval_view_suite(pal_raw, dev_y, seen, early, late)}
    boot_p: dict[str, torch.Tensor] = {"raw": pal_raw["paired"]}
    if unlab_b and cal_b:
        unlab_p = {vw: (apply_bias(pal_raw[vw], unlab_b[vw]) if vw in unlab_b else pal_raw[vw]) for vw in ("text", "image", "paired")}
        cal_p = {vw: (apply_bias(pal_raw[vw], cal_b[vw]) if vw in cal_b else pal_raw[vw]) for vw in ("text", "image", "paired")}
        regimes["unlab_center"] = eval_view_suite(unlab_p, dev_y, seen, early, late)
        regimes["cal_bias"] = eval_view_suite(cal_p, dev_y, seen, early, late)
        boot_p["unlab_center"], boot_p["cal_bias"] = unlab_p["paired"], cal_p["paired"]
    zt_u, zi_u = proto_logits(dev0_t, unseen_protos, seen, scale), proto_logits(dev0_i, unseen_protos, seen, scale)
    r1 = eval_retrieval_r1(dev0_t, dev0_i)
    ood_f1 = float(eval_view_suite({"text": zt_u, "image": zi_u, "paired": 0.5 * (zt_u + zi_u)}, dev_y, seen, early, late)["paired"]["macro_f1"])
    enc = {"unseen_prompt_paired_f1": ood_f1, "unseen_prompt_reanchored_f1": ood_f1, "retrieval": r1, "retrieval_reanchored": r1}
    return {"cone_drift": {"text": {"theta_deg": 0.0}, "image": {"theta_deg": 0.0}}, "regimes": regimes, "encoder_preservation": enc}, boot_p


def eval_pal_rls_arm(
    pal: PALRLSClassifier, dev0_t: torch.Tensor, dev0_i: torch.Tensor, dev_y: torch.Tensor,
    seen: list[int], early: list[int], late: list[int], unseen_protos: torch.Tensor, scale: float,
    cal_recs: list[dict[str, Any]], vlm: MultimodalVLM, store: PreprocessedTensorStore, eval_bs: int, use_amp: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Evaluate frozen-feature PAL-RLS analytic baseline across raw and masked cal regimes."""
    unlab_b, cal_b = _fit_pal_cal_biases(pal, seen, cal_recs, vlm, store, eval_bs, use_amp)
    return _score_pal_rls_on_split(pal, dev0_t, dev0_i, dev_y, seen, early, late, unseen_protos, scale, unlab_b, cal_b)


def evaluate_protocol_splits(
    *,
    splits: list[tuple[str, list[dict[str, Any]], dict[str, Any], dict[tuple[str, ...], torch.Tensor]]],
    ds: dict[str, Any], seen: list[int], exp_deltas: list[dict[str, torch.Tensor]], bank: StreamingClassBank,
    seq_ft_state: dict[str, torch.Tensor] | None, seq_bank: StreamingClassBank,
    seq_mm_deltas: list[dict[str, torch.Tensor]], seq_mm_bank: StreamingClassBank,
    seq_lmnar_state: dict[str, torch.Tensor] | None, seq_lmnar_bank: StreamingClassBank,
    seq_rep_state: dict[str, torch.Tensor] | None, seq_rep_bank: StreamingClassBank,
    mg_state: dict[str, torch.Tensor] | None, mg_bank: StreamingClassBank, mg_heads: dict[str, Any],
    joint_state: dict[str, torch.Tensor], joint_recs: list[dict[str, Any]], joint_bank: StreamingClassBank, pal: PALRLSClassifier,
    cal0_fx: dict[str, torch.Tensor], vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, eff_eval_bs: int, use_amp: bool,
) -> dict[str, dict[str, Any]]:
    """Merge & fit once per arm, then score identically across all requested splits (dev and/or test)."""
    del joint_recs
    cal_recs = ds["stages"][-1]["calibration_latest_views"]
    base_cfg = splits[0][2]
    lora_rank_eff = int(getattr(vlm, "rank", base_cfg.get("lora_rank", 8)))
    cal_cache: dict[tuple[str, ...], torch.Tensor] = {}

    sp_state: dict[str, dict[str, Any]] = {}
    for sp_name, rows, cfg, cache in splits:
        prof = Profiler()
        e0_t, e0_i, e_y = extract_paired_features_cached(
            vlm=vlm, kind="lora", payload={}, records=rows, tensor_store=store, batch_size=eff_eval_bs, cache=cache, use_amp=use_amp,
        )
        sp_state[sp_name] = {
            "rows": rows, "cfg": cfg, "cache": cache, "prof": prof, "e0_t": e0_t, "e0_i": e0_i, "y": e_y,
            "arms": {}, "boot": {}, "ablation": {},
        }

    fit_prof = sp_state[splits[0][0]]["prof"]

    def _run_arm(aname: str, akind: str, apayload: dict[str, torch.Tensor], bnk: StreamingClassBank | None, mgh: dict[str, Any] | None) -> None:
        fitted = fit_arm_bundle(
            arm_name=aname, kind=akind, payload=apayload, vlm=vlm, orig_prompts=ds["orig_prompts"],
            protos_full=protos_full, seen_classes=seen, cal_recs=cal_recs, train_recs=[],
            streaming_bank=bnk, cal0_fx=cal0_fx, store=store, eval_batch_size=eff_eval_bs, use_amp=use_amp,
            shrinkage=base_cfg.get("shrinkage"), profiler=fit_prof, mg_heads=mgh, cal_cache=cal_cache,
            include_ablations=(aname == "tsvm"),
        )
        fit_rec = fit_prof.records.get(f"fit_zero_mnar:{aname}")
        for sp_name, st in sp_state.items():
            if fit_rec is not None and f"fit_zero_mnar:{aname}" not in st["prof"].records:
                st["prof"].records[f"fit_zero_mnar:{aname}"] = dict(fit_rec)
            arec, blogits = score_arm_on_split(
                fitted, arm_name=aname, kind=akind, payload=apayload, vlm=vlm, unseen_protos_full=unseen_protos_full,
                seen_classes=seen, early_classes=ds["early_classes"], late_classes=ds["late_classes"],
                eval_rows=st["rows"], eval_cache=st["cache"], eval0_t=st["e0_t"], eval0_i=st["e0_i"],
                store=store, eval_batch_size=eff_eval_bs, use_amp=use_amp, profiler=st["prof"], mg_heads=mgh,
            )
            st["arms"][aname], st["boot"][aname] = arec, blogits
            if aname == "tsvm":
                tsvm_t, tsvm_i, _ = extract_paired_features_cached(
                    vlm=vlm, kind="delta", payload=apayload, records=st["rows"], tensor_store=store,
                    batch_size=eff_eval_bs, cache=st["cache"], use_amp=use_amp,
                )
                for ab_name, (ab_h, use_re) in fitted["ablation_heads"].items():
                    tx_ab, ix_ab = (apply_reanchor(tsvm_t, fitted["re_t"]), apply_reanchor(tsvm_i, fitted["re_i"])) if use_re else (tsvm_t, tsvm_i)
                    st["ablation"][ab_name] = eval_view_suite(ab_h.predict_views(tx_ab, ix_ab), st["y"], seen, ds["early_classes"], ds["late_classes"])

    for m in ALL_MERGE_METHODS:
        with fit_prof.track(f"merge:{m}"):
            delta_m = merge_task_deltas(exp_deltas, m, rank_per_task=lora_rank_eff)
        m_rec = fit_prof.records[f"merge:{m}"]
        for st in sp_state.values():
            st["prof"].records[f"merge:{m}"] = dict(m_rec)
        _run_arm(m, "delta", delta_m, bank, None)
        del delta_m

    seq_mm_merged = merge_task_deltas(seq_mm_deltas, "magmax", rank_per_task=lora_rank_eff)
    traj_specs: list[tuple[str, str, dict[str, torch.Tensor], StreamingClassBank | None, dict[str, Any] | None]] = [
        ("seq_ft", "lora", seq_ft_state or {}, seq_bank, None),
        ("seq_magmax", "delta", seq_mm_merged, seq_mm_bank, None),
        ("seq_lmnar", "lora", seq_lmnar_state or {}, seq_lmnar_bank, None),
        ("seq_replay256", "lora", seq_rep_state or {}, seq_rep_bank, None),
        *([("mg_clip", "lora", mg_state, mg_bank, mg_heads)] if bool(base_cfg.get("include_mg_clip", True)) and mg_state is not None else []),
        ("joint_oracle", "lora", joint_state, joint_bank, None),
    ]
    for aname, akind, apayload, bnk, mgh in traj_specs:
        _run_arm(aname, akind, apayload, bnk, mgh)

    unlab_pal, cal_pal = _fit_pal_cal_biases(pal, seen, cal_recs, vlm, store, eff_eval_bs, use_amp)
    out_by_split: dict[str, dict[str, Any]] = {}
    for sp_name, _, cfg, _ in splits:
        st = sp_state[sp_name]
        pal_rec, pal_boot = _score_pal_rls_on_split(
            pal, st["e0_t"], st["e0_i"], st["y"], seen, ds["early_classes"], ds["late_classes"],
            unseen_protos_full, vlm.logit_scale, unlab_pal, cal_pal,
        )
        st["arms"]["pal_rls"], st["boot"]["pal_rls"] = pal_rec, pal_boot
        anchor, boot_map = st["boot"]["tsvm"]["zero_mnar"], st["boot"]
        boot_refs = [
            ("tsvm:zero_mnar_vs_tsvm:raw", boot_map["tsvm"]["raw"]), ("tsvm:zero_mnar_vs_tsvm:gr_clip", boot_map["tsvm"].get("gr_clip")),
            ("tsvm:zero_mnar_vs_tsvm:cal_bias", boot_map["tsvm"].get("cal_bias")), ("tsvm:zero_mnar_vs_knots:cal_bias", boot_map["knots"].get("cal_bias")),
            ("tsvm:zero_mnar_vs_iso_c:cal_bias", boot_map["iso_c"].get("cal_bias")), ("tsvm:zero_mnar_vs_iso_cts:cal_bias", boot_map["iso_cts"].get("cal_bias")),
            ("tsvm:zero_mnar_vs_wudi:cal_bias", boot_map["wudi"].get("cal_bias")), ("tsvm:zero_mnar_vs_dop:cal_bias", boot_map["dop"].get("cal_bias")),
            ("tsvm:zero_mnar_vs_dc_lora:cal_bias", boot_map["dc_lora"].get("cal_bias")), ("tsvm:zero_mnar_vs_tsvm:ncm", boot_map["tsvm"].get("ncm")),
            ("tsvm:zero_mnar_vs_seq_magmax:ncm", boot_map.get("seq_magmax", {}).get("ncm")), ("tsvm:zero_mnar_vs_seq_lmnar:raw", boot_map["seq_lmnar"].get("raw")),
            ("tsvm:zero_mnar_vs_mg_clip:raw", boot_map.get("mg_clip", {}).get("raw")), ("tsvm:zero_mnar_vs_pal_rls:raw", boot_map["pal_rls"].get("raw")),
        ]
        bootstrap = {lbl: paired_bootstrap_diff(anchor, ref, st["y"], seen, seed=int(cfg["seed"])) for lbl, ref in boot_refs if ref is not None}
        out_by_split[sp_name] = {
            "schema": "mnarm-iccv-protocol-v2", "split": sp_name, "config": cfg, "arms": st["arms"],
            "ablation_tsvm": st["ablation"], "bootstrap": bootstrap, "theta0_preservation": dict(pal_rec["encoder_preservation"]), "profiling": st["prof"].records,
        }
    return out_by_split


def evaluate_protocol_split(
    *,
    config: dict[str, Any], ds: dict[str, Any], eval_rows: list[dict[str, Any]], split_name: str,
    seen: list[int], exp_deltas: list[dict[str, torch.Tensor]], bank: StreamingClassBank,
    seq_ft_state: dict[str, torch.Tensor] | None, seq_bank: StreamingClassBank,
    seq_mm_deltas: list[dict[str, torch.Tensor]], seq_mm_bank: StreamingClassBank,
    seq_lmnar_state: dict[str, torch.Tensor] | None, seq_lmnar_bank: StreamingClassBank,
    seq_rep_state: dict[str, torch.Tensor] | None, seq_rep_bank: StreamingClassBank,
    mg_state: dict[str, torch.Tensor] | None, mg_bank: StreamingClassBank, mg_heads: dict[str, Any],
    joint_state: dict[str, torch.Tensor], joint_recs: list[dict[str, Any]], joint_bank: StreamingClassBank, pal: PALRLSClassifier,
    cal0_fx: dict[str, torch.Tensor], vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, feat_cache: dict[tuple[str, ...], torch.Tensor], eff_eval_bs: int, use_amp: bool,
) -> dict[str, Any]:
    """Evaluate all 14 mergers + continual/SOTA tracks + ablations + bootstrap on one split (dev or test)."""
    return evaluate_protocol_splits(
        splits=[(split_name, eval_rows, config, feat_cache)], ds=ds, seen=seen, exp_deltas=exp_deltas, bank=bank,
        seq_ft_state=seq_ft_state, seq_bank=seq_bank, seq_mm_deltas=seq_mm_deltas, seq_mm_bank=seq_mm_bank,
        seq_lmnar_state=seq_lmnar_state, seq_lmnar_bank=seq_lmnar_bank, seq_rep_state=seq_rep_state, seq_rep_bank=seq_rep_bank,
        mg_state=mg_state, mg_bank=mg_bank, mg_heads=mg_heads, joint_state=joint_state, joint_recs=joint_recs, joint_bank=joint_bank,
        pal=pal, cal0_fx=cal0_fx, vlm=vlm, protos_full=protos_full, unseen_protos_full=unseen_protos_full, store=store,
        eff_eval_bs=eff_eval_bs, use_amp=use_amp,
    )[split_name]
