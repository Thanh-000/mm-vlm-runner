"""Unified ICCV Same-Bank benchmark runner: 14 verbatim mergers + VLM-CL/Analytic SOTA vs CSS-CT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mnarm.core import build_run_lock, load_yaml, seed_everything, write_json_atomic
from mnarm.data import PreprocessedTensorStore, load_benchmark_contracts
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, paired_bootstrap_diff
from mnarm.methods import (
    ALL_MERGE_METHODS,
    PALRLSClassifier,
    StreamingClassBank,
    apply_reanchor,
    merge_task_deltas,
    train_mg_clip_stage,
)
from mnarm.models import MultimodalVLM
from mnarm.runners.iccv_eval import eval_pal_rls_arm, evaluate_arm_all_regimes, fit_css_ct_with_reanchor
from mnarm.runners.iccv_report import render_protocol_report, write_same_bank_suite_outputs
from mnarm.runners.stage_ops import collect_masked_cal_bundle, collect_train_features_by_modality, extract_paired_features_cached, train_lora_stage

ABLATION_SPECS = (
    ("none_vmf", "none_zero_shot", "isotropic_vmf", False),
    ("prompt_vmf", "prompt_bridge", "isotropic_vmf", False),
    ("cm_raw_oas", "cross_modal_transport", "raw_oas", False),
    ("cm_sfloor", "cross_modal_transport", "spectral_floor", False),
    ("cm_vmf", "cross_modal_transport", "isotropic_vmf", False),
    ("hybrid_vmf", "hybrid_bridge", "isotropic_vmf", False),
    ("css_ct", "conjugate_bayes", "marginal_epistemic", True),
)


def _cached_base0_features(
    vlm: MultimodalVLM, records: list[dict[str, Any]], store: PreprocessedTensorStore, eval_bs: int, use_amp: bool,
    cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | None]:
    """Encode frozen theta_0 features per uid once across protocols for StreamingClassBank."""
    uncached = [r for r in records if ("text" in r["mask"] and r["uid"] not in cache["text"]) or ("image" in r["mask"] and r["uid"] not in cache["image"])]
    if uncached:
        tf_new = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=uncached, tensor_store=store, batch_size=eval_bs, use_amp=use_amp)
        t_pos = i_pos = 0
        for r in uncached:
            if "text" in r["mask"] and tf_new["text_x"] is not None:
                cache["text"].setdefault(r["uid"], tf_new["text_x"][t_pos]); t_pos += 1
            if "image" in r["mask"] and tf_new["image_x"] is not None:
                cache["image"].setdefault(r["uid"], tf_new["image_x"][i_pos]); i_pos += 1
    tx, ty, ix, iy, pt, pi = [], [], [], [], [], []
    for r in records:
        uid, lbl = r["uid"], int(r["label"])
        has_t, has_i = ("text" in r["mask"] and uid in cache["text"]), ("image" in r["mask"] and uid in cache["image"])
        if has_t:
            tx.append(cache["text"][uid]); ty.append(lbl)
        if has_i:
            ix.append(cache["image"][uid]); iy.append(lbl)
        if has_t and has_i:
            pt.append(cache["text"][uid]); pi.append(cache["image"][uid])
    return {
        "text_x": torch.stack(tx, 0) if tx else None, "text_y": torch.tensor(ty, dtype=torch.long) if ty else None,
        "image_x": torch.stack(ix, 0) if ix else None, "image_y": torch.tensor(iy, dtype=torch.long) if iy else None,
        "paired_t": torch.stack(pt, 0) if pt else None, "paired_i": torch.stack(pi, 0) if pi else None,
    }


def run_single_protocol_same_bank(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path,
    vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, dev_cache: dict[tuple[str, ...], torch.Tensor],
    base0_cache: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Run all 14 author mergers + seq_ft(CSS-CT) + seq_replay256 + MG-CLIP + PAL-RLS + joint_oracle."""
    if bool(config.get("allow_test", False)):
        raise ValueError("ICCV Same-Bank benchmark forbids allow_test=True prior to final freeze.")
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    use_amp = bool(config.get("use_amp", True)) and vlm.device == "cuda"
    eff_eval_bs = max(int(config["eval_batch_size"]), 512) if (vlm.device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 36 * (1024 ** 3)) else int(config["eval_batch_size"])
    b0_cache = base0_cache if base0_cache is not None else {"text": {}, "image": {}}
    ds = load_benchmark_contracts(str(config.get("dataset", "crisismmd")), seq_inputs, images_dir, config["protocol"], allow_test=False, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "RUN_LOCK.json", build_run_lock(config, repo_root=Path(__file__).resolve().parents[3], dataset_hashes=ds["hashes"]))
    profiler = Profiler()

    st4 = ds["stages"][-1]
    seen = st4["seen_classes"]
    dev_rows = [r for r in ds["dev"] if int(r["label"]) in set(seen)]
    dev0_t, dev0_i, dev_y = extract_paired_features_cached(
        vlm=vlm, kind="lora", payload={}, records=dev_rows, tensor_store=store,
        batch_size=eff_eval_bs, cache=dev_cache, use_amp=use_amp,
    )

    exp_deltas: list[dict[str, torch.Tensor]] = []
    bank, seq_bank, seq_lmnar_bank, seq_rep_bank, mg_bank = (
        StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(),
    )
    pal = PALRLSClassifier(vlm.embed_dim, len(ds["orig_prompts"]), lam=1.0, seed=seed)
    seq_ft_state = seq_lmnar_state = seq_rep_state = mg_state = None
    mg_heads: dict[str, Any] = {}
    seen_so_far: list[int] = []
    t0 = time.perf_counter()
    for st in ds["stages"]:
        t = int(st["stage"])
        new_c = [int(c) for c in st["new_classes"]]
        old_c = [c for c in seen_so_far if c not in set(new_c)]
        seen_so_far = [int(c) for c in st["seen_classes"]]

        exp_s, _ = train_lora_stage(
            vlm=vlm, init_lora=None, records=st["fit"], protos_full=protos_full, seen_classes=new_c,
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed, use_amp=use_amp, tensor_store=store,
        )
        exp_deltas.append(vlm.extract_lora_delta(exp_s))
        tf_st = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=exp_s, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        tf_b0 = _cached_base0_features(vlm, st["fit"], store, eff_eval_bs, use_amp, b0_cache)
        bank.update_stage(u_expert_t=tf_st["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_st["text_y"], u_expert_i=tf_st["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_st["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])

        if tf_b0["paired_t"] is not None and tf_b0["paired_i"] is not None:
            pal.update(tf_b0["paired_t"], tf_b0["paired_i"], tf_b0["text_y"], ("text", "image"))
        else:
            if tf_b0["text_x"] is not None:
                pal.update(tf_b0["text_x"], None, tf_b0["text_y"], ("text",))
            if tf_b0["image_x"] is not None:
                pal.update(None, tf_b0["image_x"], tf_b0["image_y"], ("image",))

        seq_ft_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_ft_state, records=st["fit"], protos_full=protos_full, seen_classes=seen_so_far,
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 1000, use_amp=use_amp, tensor_store=store,
        )
        tf_sq = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=seq_ft_state, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        seq_bank.update_stage(u_expert_t=tf_sq["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_sq["text_y"], u_expert_i=tf_sq["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_sq["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])

        seq_lmnar_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_lmnar_state, records=st["fit"], protos_full=protos_full, seen_classes=seen_so_far,
            objective="l_mnar", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 1500, use_amp=use_amp, tensor_store=store,
        )
        tf_slm = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=seq_lmnar_state, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        seq_lmnar_bank.update_stage(u_expert_t=tf_slm["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_slm["text_y"], u_expert_i=tf_slm["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_slm["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])

        rep_recs = st["fit"] + st["buffer_before"]
        seq_rep_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_rep_state, records=rep_recs, protos_full=protos_full, seen_classes=seen_so_far,
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 2000, use_amp=use_amp, tensor_store=store,
        )
        tf_sr = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=seq_rep_state, records=rep_recs, tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        tf_b0_rep = _cached_base0_features(vlm, rep_recs, store, eff_eval_bs, use_amp, b0_cache)
        seq_rep_bank.update_stage(u_expert_t=tf_sr["text_x"], u_base_t=tf_b0_rep["text_x"], y_t=tf_sr["text_y"], u_expert_i=tf_sr["image_x"], u_base_i=tf_b0_rep["image_x"], y_i=tf_sr["image_y"], paired_base_t=tf_b0_rep["paired_t"], paired_base_i=tf_b0_rep["paired_i"])

        if bool(config.get("include_mg_clip", True)):
            mg_state, mg_heads = train_mg_clip_stage(
                vlm=vlm, prev_state=mg_state, mg_heads=mg_heads, records=st["fit"], protos_full=protos_full,
                old_classes=old_c, new_classes=new_c, stage_idx=t, epochs=int(config["epochs"]),
                effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
                lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 3000, use_amp=use_amp, tensor_store=store,
            )
            tf_mg = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=mg_state, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
            mg_bank.update_stage(u_expert_t=tf_mg["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_mg["text_y"], u_expert_i=tf_mg["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_mg["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])
        print(f"[{config['backbone']} {config['protocol']}] trained stage {t}/{len(ds['stages'])} ({time.perf_counter() - t0:.1f}s)", flush=True)

    joint_recs = st4["fit"] + st4["joint_before"]
    joint_state, _ = train_lora_stage(
        vlm=vlm, init_lora=None, records=joint_recs, protos_full=protos_full, seen_classes=seen,
        objective="std_ce", stage_idx=len(ds["stages"]), epochs=int(config["epochs"]),
        effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
        lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 4000, use_amp=use_amp, tensor_store=store,
    )

    _, _, cal0_fx = collect_masked_cal_bundle(
        vlm=vlm, kind="lora", payload={}, cal_recs=st4["calibration_latest_views"], store=store,
        protos_full=protos_full, seen_classes=seen, batch_size=eff_eval_bs, use_amp=use_amp,
    )

    lora_rank_eff = int(getattr(vlm, "rank", config.get("lora_rank", 8)))
    arms_out: dict[str, Any] = {}
    boot_map: dict[str, dict[str, torch.Tensor]] = {}
    tsvm_delta: dict[str, torch.Tensor] | None = None

    def _eval_one(
        aname: str, akind: str, apayload: dict[str, torch.Tensor], tr_recs: list[dict[str, Any]],
        bnk: StreamingClassBank | None, mgh: dict[str, Any] | None,
    ) -> None:
        arec, blogits = evaluate_arm_all_regimes(
            arm_name=aname, kind=akind, payload=apayload, vlm=vlm, orig_prompts=ds["orig_prompts"], protos_full=protos_full,
            unseen_protos_full=unseen_protos_full, seen_classes=seen, early_classes=ds["early_classes"],
            late_classes=ds["late_classes"], dev_rows=dev_rows, cal_recs=st4["calibration_latest_views"],
            dev_cache=dev_cache, dev0_t=dev0_t, dev0_i=dev0_i, train_recs=tr_recs, streaming_bank=bnk,
            cal0_fx=cal0_fx, store=store, eval_batch_size=eff_eval_bs,
            use_amp=use_amp, shrinkage=config.get("shrinkage"), profiler=profiler, mg_heads=mgh,
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
    ]
    if bool(config.get("include_mg_clip", True)) and mg_state is not None:
        traj_specs.append(("mg_clip", "lora", mg_state, [], mg_bank, mg_heads))
    traj_specs.append(("joint_oracle", "lora", joint_state, joint_recs, None, None))
    for spec in traj_specs:
        _eval_one(*spec)

    pal_rec, pal_boot = eval_pal_rls_arm(
        pal, dev0_t, dev0_i, dev_y, seen, ds["early_classes"], ds["late_classes"], unseen_protos_full,
        vlm.logit_scale, st4["calibration_latest_views"], vlm, store, eff_eval_bs, use_amp,
    )
    arms_out["pal_rls"], boot_map["pal_rls"] = pal_rec, pal_boot
    t0_pres = dict(pal_rec["encoder_preservation"])

    if tsvm_delta is None:
        tsvm_delta = merge_task_deltas(exp_deltas, "tsvm", rank_per_task=lora_rank_eff)
    tsvm_t, tsvm_i, _ = extract_paired_features_cached(vlm=vlm, kind="delta", payload=tsvm_delta, records=dev_rows, tensor_store=store, batch_size=eff_eval_bs, cache=dev_cache, use_amp=use_amp)
    _, _, cal_tsvm_fx = collect_masked_cal_bundle(
        vlm=vlm, kind="delta", payload=tsvm_delta, cal_recs=st4["calibration_latest_views"], store=store,
        protos_full=protos_full, seen_classes=seen, batch_size=eff_eval_bs, use_amp=use_amp,
    )
    ablation_out: dict[str, Any] = {}
    for ab_name, bmode, pcmode, use_re in ABLATION_SPECS:
        ab_h, rt_ab, ri_ab = fit_css_ct_with_reanchor(
            bridge_mode=bmode, partial_cov_mode=pcmode, use_reanchor=use_re, vlm=vlm, kind="delta", payload=tsvm_delta,
            train_recs=[], streaming_bank=bank, store=store, orig_prompts=ds["orig_prompts"],
            protos_full=protos_full, seen_classes=seen, eval_batch_size=eff_eval_bs, use_amp=use_amp, shrinkage=config.get("shrinkage"),
            cal_recs=st4["calibration_latest_views"], cal0_fx=cal0_fx, cal_fx=cal_tsvm_fx,
        )
        tx_ab, ix_ab = (apply_reanchor(tsvm_t, rt_ab), apply_reanchor(tsvm_i, ri_ab)) if use_re else (tsvm_t, tsvm_i)
        ablation_out[ab_name] = eval_view_suite(ab_h.predict_views(tx_ab, ix_ab), dev_y, seen, ds["early_classes"], ds["late_classes"])

    bootstrap: dict[str, Any] = {}
    anchor = boot_map["tsvm"]["zero_mnar"]
    for label, ref in [
        ("tsvm:zero_mnar_vs_tsvm:raw", boot_map["tsvm"]["raw"]),
        ("tsvm:zero_mnar_vs_tsvm:gr_clip", boot_map["tsvm"].get("gr_clip")),
        ("tsvm:zero_mnar_vs_tsvm:cal_bias", boot_map["tsvm"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_knots:cal_bias", boot_map["knots"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_iso_c:cal_bias", boot_map["iso_c"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_iso_cts:cal_bias", boot_map["iso_cts"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_wudi:cal_bias", boot_map["wudi"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_dop:cal_bias", boot_map["dop"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_dc_lora:cal_bias", boot_map["dc_lora"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_seq_lmnar:raw", boot_map["seq_lmnar"].get("raw")),
        ("tsvm:zero_mnar_vs_mg_clip:raw", boot_map.get("mg_clip", {}).get("raw")),
        ("tsvm:zero_mnar_vs_pal_rls:raw", boot_map["pal_rls"].get("raw")),
    ]:
        if ref is not None:
            bootstrap[label] = paired_bootstrap_diff(anchor, ref, dev_y, seen, seed=seed)

    payload_out = {
        "schema": "mnarm-iccv-protocol-v2", "config": config, "arms": arms_out,
        "ablation_tsvm": ablation_out, "bootstrap": bootstrap, "theta0_preservation": t0_pres, "profiling": profiler.records,
    }
    write_json_atomic(out_dir / "ICCV_PROTOCOL_RESULTS.json", payload_out)
    render_protocol_report(payload_out, out_dir / "ICCV_PROTOCOL_REPORT.md")
    return payload_out


def run_iccv_same_bank(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path,
    protocols: list[str] | tuple[str, ...] | None = None, resume: bool = False,
    model_override: str | None = None, vlm: MultimodalVLM | None = None, store: PreprocessedTensorStore | None = None,
) -> dict[str, Any]:
    """Run unified Same-Bank ICCV evaluation across requested protocols sharing 1 VLM and cache."""
    if bool(config.get("allow_test", False)):
        raise ValueError("ICCV Same-Bank benchmark forbids allow_test=True prior to final freeze.")
    protos = [str(p).upper() for p in (protocols or ("P4", "P3", "P2", "P1"))]
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    backbone = str(config["backbone"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ds = load_benchmark_contracts(str(config.get("dataset", "crisismmd")), seq_inputs, images_dir, "P4", allow_test=False, seed=int(config.get("seed", 11)))
    vlm = vlm or MultimodalVLM(backbone, model_override or config["model_id"], device=device, rank=int(config["lora_rank"]), alpha=int(config["lora_alpha"]), revision=None if model_override else config.get("revision"), local_files_only=bool(model_override))
    protos_full, unseen_protos = vlm.encode_prompts(base_ds["orig_prompts"]), vlm.encode_prompts(base_ds["unseen_prompts"])
    store = store or PreprocessedTensorStore(base_ds["lookup"], vlm, max_cached_images=int(config.get("max_cached_images", 25000)), workers=int(config.get("workers", 0)))
    dev_cache: dict[tuple[str, ...], torch.Tensor] = {}
    base0_cache: dict[str, dict[str, torch.Tensor]] = {"text": {}, "image": {}}

    per_proto: dict[str, dict[str, Any]] = {}
    for p in protos:
        p_out = out_dir / f"{backbone}_{p.lower()}"
        p_json = p_out / "ICCV_PROTOCOL_RESULTS.json"
        cached_p = json.loads(p_json.read_text(encoding="utf8")) if (resume and p_json.is_file()) else None
        is_b1_valid = bool(
            cached_p
            and cached_p.get("schema") == "mnarm-iccv-protocol-v2"
            and "balanced_ece" in cached_p.get("arms", {}).get("tsvm", {}).get("regimes", {}).get("raw", {}).get("paired", {})
        )
        if is_b1_valid:
            assert cached_p is not None
            per_proto[p] = cached_p
        else:
            pcfg = {**config, "protocol": p, "experiment": f"iccv_same_bank_{config.get('dataset', 'crisismmd')}_{backbone}_{p.lower()}"}
            per_proto[p] = run_single_protocol_same_bank(pcfg, seq_inputs, images_dir, p_out, vlm, protos_full, unseen_protos, store, dev_cache, base0_cache=base0_cache)
        write_same_bank_suite_outputs(config, per_proto, out_dir)
    return write_same_bank_suite_outputs(config, per_proto, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seq-inputs", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["P4", "P3", "P2", "P1"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()
    summary = run_iccv_same_bank(load_yaml(args.config), args.seq_inputs, args.images, args.out, protocols=args.protocols, resume=args.resume, model_override=args.model_path)
    print(f"[ICCV SAME-BANK DONE] {summary['backbone']} -> {list(summary['protocols'].keys())}")


if __name__ == "__main__":
    main()
