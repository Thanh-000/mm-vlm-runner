"""Unified ICCV Same-Bank benchmark runner comparing 9 verbatim author mergers + 3 seq arms across 4 regimes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mnarm.core import build_run_lock, load_yaml, seed_everything, write_json_atomic
from mnarm.data import PreprocessedTensorStore, load_crisismmd_contracts
from mnarm.eval import Profiler, eval_retrieval_r1, eval_view_suite, measure_cone_drift, paired_bootstrap_diff
from mnarm.methods import (
    ALL_MERGE_METHODS,
    SelfNormalizedBayesTransportHead,
    StreamingClassBank,
    apply_bias,
    fit_cal_bias,
    fit_unlabeled_center,
    merge_task_deltas,
)
from mnarm.models import MultimodalVLM, proto_logits
from mnarm.runners.iccv_report import render_protocol_report, write_same_bank_suite_outputs
from mnarm.runners.stage_ops import (
    collect_masked_cal_logits,
    collect_train_features_by_modality,
    extract_paired_features_cached,
    train_lora_stage,
)

ABLATION_SPECS = (
    ("none_vmf", "none_zero_shot", "isotropic_vmf"),
    ("prompt_vmf", "prompt_bridge", "isotropic_vmf"),
    ("cm_raw_oas", "cross_modal_transport", "raw_oas"),
    ("cm_sfloor", "cross_modal_transport", "spectral_floor"),
    ("cm_vmf", "cross_modal_transport", "isotropic_vmf"),
    ("hybrid_vmf", "hybrid_bridge", "isotropic_vmf"),
)


def _fit_frozen_zero_mnar(
    *,
    bridge_mode: str,
    partial_cov_mode: str,
    vlm: MultimodalVLM,
    kind: str,
    payload: dict[str, torch.Tensor],
    train_recs: list[dict[str, Any]],
    streaming_bank: StreamingClassBank | None,
    fit4_recs: list[dict[str, Any]],
    store: PreprocessedTensorStore,
    protos_full: torch.Tensor,
    seen_classes: list[int],
    eval_batch_size: int,
    use_amp: bool,
    shrinkage: float | None,
) -> SelfNormalizedBayesTransportHead:
    head = SelfNormalizedBayesTransportHead(bridge_mode=bridge_mode, partial_cov_mode=partial_cov_mode, shrinkage=shrinkage)
    if streaming_bank is None:
        tf = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=train_recs,
            tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
        return head.fit(
            train_feats_t=tf["text_x"], train_y_t=tf["text_y"],
            train_feats_i=tf["image_x"], train_y_i=tf["image_y"],
            paired_feats_t=tf["paired_t"], paired_feats_i=tf["paired_i"],
            protos_full=protos_full, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
        )
    f4_b = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=fit4_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp)
    f4_a = collect_train_features_by_modality(vlm=vlm, kind=kind, payload=payload, records=fit4_recs, tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp)
    bstats = streaming_bank.transport_to_target_stats(
        bridge_u0_t=f4_b["text_x"], bridge_um_t=f4_a["text_x"],
        bridge_u0_i=f4_b["image_x"], bridge_um_i=f4_a["image_x"],
        seen_classes=seen_classes, probe_y_t=f4_a["text_y"], probe_y_i=f4_a["image_y"],
    )
    return head.fit_from_sufficient_stats(
        tower_stats=bstats["towers"], paired_feats_t=bstats["paired"][0], paired_feats_i=bstats["paired"][1],
        protos_full=protos_full, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
    )


def _evaluate_arm_all_regimes(
    *,
    arm_name: str,
    kind: str,
    payload: dict[str, torch.Tensor],
    vlm: MultimodalVLM,
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
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with profiler.track(f"eval_dev:{arm_name}", n_samples=len(dev_rows)):
        dev_t, dev_i, dev_y = extract_paired_features_cached(
            vlm=vlm, kind=kind, payload=payload, records=dev_rows,
            tensor_store=store, batch_size=eval_batch_size, cache=dev_cache, use_amp=use_amp,
        )

    cone_t, cone_i = measure_cone_drift(dev_t, dev0_t), measure_cone_drift(dev_i, dev0_i)
    zt_raw = proto_logits(dev_t, protos_full, seen_classes, vlm.logit_scale)
    zi_raw = proto_logits(dev_i, protos_full, seen_classes, vlm.logit_scale)
    raw_logits = {"text": zt_raw, "image": zi_raw, "paired": 0.5 * (zt_raw + zi_raw)}
    regimes: dict[str, Any] = {"raw": eval_view_suite(raw_logits, dev_y, seen_classes, early_classes, late_classes)}
    boot_p: dict[str, torch.Tensor] = {"raw": raw_logits["paired"]}

    cal_sc, cal_lb = collect_masked_cal_logits(
        vlm=vlm, kind=kind, payload=payload, cal_recs=cal_recs, store=store,
        protos_full=protos_full, seen_classes=seen_classes, batch_size=eval_batch_size, use_amp=use_amp,
    )
    if cal_sc:
        unlab_l = {vw: (apply_bias(raw_logits[vw], fit_unlabeled_center(cal_sc[vw])["bias"]) if vw in cal_sc else raw_logits[vw]) for vw in ("text", "image", "paired")}
        cal_l = {vw: (apply_bias(raw_logits[vw], fit_cal_bias(cal_sc[vw], cal_lb[vw], seen_classes)["bias"]) if vw in cal_sc else raw_logits[vw]) for vw in ("text", "image", "paired")}
        regimes["unlab_center"] = eval_view_suite(unlab_l, dev_y, seen_classes, early_classes, late_classes)
        regimes["cal_bias"] = eval_view_suite(cal_l, dev_y, seen_classes, early_classes, late_classes)
        boot_p["unlab_center"], boot_p["cal_bias"] = unlab_l["paired"], cal_l["paired"]

    with profiler.track(f"fit_zero_mnar:{arm_name}"):
        zn_head = _fit_frozen_zero_mnar(
            bridge_mode="cross_modal_transport", partial_cov_mode="isotropic_vmf",
            vlm=vlm, kind=kind, payload=payload, train_recs=train_recs,
            streaming_bank=streaming_bank, fit4_recs=fit4_recs, store=store,
            protos_full=protos_full, seen_classes=seen_classes,
            eval_batch_size=eval_batch_size, use_amp=use_amp, shrinkage=shrinkage,
        )
    zn_logits = zn_head.predict_views(dev_t, dev_i)
    regimes["zero_mnar"] = {**eval_view_suite(zn_logits, dev_y, seen_classes, early_classes, late_classes), "diagnostics": zn_head.diagnostics}
    boot_p["zero_mnar"] = zn_logits["paired"]

    zt_u = proto_logits(dev_t, unseen_protos_full, seen_classes, vlm.logit_scale)
    zi_u = proto_logits(dev_i, unseen_protos_full, seen_classes, vlm.logit_scale)
    unseen_ev = eval_view_suite({"text": zt_u, "image": zi_u, "paired": 0.5 * (zt_u + zi_u)}, dev_y, seen_classes, early_classes, late_classes)
    enc_pres = {"unseen_prompt_paired_f1": float(unseen_ev["paired"]["macro_f1"]), "retrieval": eval_retrieval_r1(dev_t, dev_i)}
    return {"cone_drift": {"text": cone_t, "image": cone_i}, "regimes": regimes, "encoder_preservation": enc_pres}, boot_p


def run_single_protocol_same_bank(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path,
    vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, dev_cache: dict[tuple[str, ...], torch.Tensor],
) -> dict[str, Any]:
    """Run all 9 verbatim author mergers + 3 seq/oracle arms on one shared expert bank."""
    if bool(config.get("allow_test", False)):
        raise ValueError("ICCV Same-Bank benchmark forbids allow_test=True prior to final freeze.")
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    use_amp = bool(config.get("use_amp", True)) and vlm.device == "cuda"
    ds = load_crisismmd_contracts(seq_inputs, images_dir, config["protocol"], allow_test=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "RUN_LOCK.json", build_run_lock(config, repo_root=Path(__file__).resolve().parents[3], dataset_hashes=ds["hashes"]))
    profiler = Profiler()

    st4 = ds["stages"][-1]
    seen = st4["seen_classes"]
    dev_rows = [r for r in ds["dev"] if int(r["label"]) in set(seen)]
    dev0_t, dev0_i, dev_y = extract_paired_features_cached(
        vlm=vlm, kind="lora", payload={}, records=dev_rows, tensor_store=store,
        batch_size=int(config["eval_batch_size"]), cache=dev_cache, use_amp=use_amp,
    )

    exp_deltas: list[dict[str, torch.Tensor]] = []
    bank = StreamingClassBank()
    seq_ft_state = seq_rep_state = None
    t0 = time.perf_counter()
    for st in ds["stages"]:
        t = int(st["stage"])
        exp_s, _ = train_lora_stage(
            vlm=vlm, init_lora=None, records=st["fit"], protos_full=protos_full, seen_classes=st["new_classes"],
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed, use_amp=use_amp, tensor_store=store,
        )
        exp_deltas.append(vlm.extract_lora_delta(exp_s))
        tf_st = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=exp_s, records=st["fit"], tensor_store=store, batch_size=int(config["eval_batch_size"]), use_amp=use_amp)
        tf_b0 = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=st["fit"], tensor_store=store, batch_size=int(config["eval_batch_size"]), use_amp=use_amp)
        bank.update_stage(
            u_expert_t=tf_st["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_st["text_y"],
            u_expert_i=tf_st["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_st["image_y"],
            paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"],
        )
        seq_ft_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_ft_state, records=st["fit"], protos_full=protos_full, seen_classes=st["seen_classes"],
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 1000, use_amp=use_amp, tensor_store=store,
        )
        seq_rep_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_rep_state, records=st["fit"] + st["buffer_before"], protos_full=protos_full, seen_classes=st["seen_classes"],
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 2000, use_amp=use_amp, tensor_store=store,
        )
        print(f"[{config['backbone']} {config['protocol']}] trained stage {t}/4 ({time.perf_counter() - t0:.1f}s)", flush=True)

    joint_recs = st4["fit"] + st4["joint_before"]
    joint_state, _ = train_lora_stage(
        vlm=vlm, init_lora=None, records=joint_recs, protos_full=protos_full, seen_classes=seen,
        objective="std_ce", stage_idx=4, epochs=int(config["epochs"]),
        effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
        lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 4000, use_amp=use_amp, tensor_store=store,
    )
    print(f"[{config['backbone']} {config['protocol']}] joint_oracle finished ({time.perf_counter() - t0:.1f}s)", flush=True)

    arm_specs: list[tuple[str, str, dict[str, torch.Tensor], list[dict[str, Any]], StreamingClassBank | None]] = [
        (m, "delta", merge_task_deltas(exp_deltas, m), [], bank) for m in ALL_MERGE_METHODS
    ] + [
        ("seq_ft", "lora", seq_ft_state or {}, st4["fit"], None),
        ("seq_replay256", "lora", seq_rep_state or {}, st4["fit"] + st4["buffer_before"], None),
        ("joint_oracle", "lora", joint_state, joint_recs, None),
    ]

    arms_out: dict[str, Any] = {}
    boot_map: dict[str, dict[str, torch.Tensor]] = {}
    for aname, akind, apayload, tr_recs, bnk in arm_specs:
        arec, blogits = _evaluate_arm_all_regimes(
            arm_name=aname, kind=akind, payload=apayload, vlm=vlm, protos_full=protos_full,
            unseen_protos_full=unseen_protos_full, seen_classes=seen, early_classes=ds["early_classes"],
            late_classes=ds["late_classes"], dev_rows=dev_rows, cal_recs=st4["calibration_latest_views"],
            dev_cache=dev_cache, dev0_t=dev0_t, dev0_i=dev0_i, train_recs=tr_recs, streaming_bank=bnk,
            fit4_recs=st4["fit"], store=store, eval_batch_size=int(config["eval_batch_size"]),
            use_amp=use_amp, shrinkage=config.get("shrinkage"), profiler=profiler,
        )
        arms_out[aname], boot_map[aname] = arec, blogits

    tsvm_delta = merge_task_deltas(exp_deltas, "tsvm")
    tsvm_t, tsvm_i, _ = extract_paired_features_cached(vlm=vlm, kind="delta", payload=tsvm_delta, records=dev_rows, tensor_store=store, batch_size=int(config["eval_batch_size"]), cache=dev_cache, use_amp=use_amp)
    ablation_out: dict[str, Any] = {}
    for ab_name, bmode, pcmode in ABLATION_SPECS:
        ab_h = _fit_frozen_zero_mnar(
            bridge_mode=bmode, partial_cov_mode=pcmode, vlm=vlm, kind="delta", payload=tsvm_delta,
            train_recs=[], streaming_bank=bank, fit4_recs=st4["fit"], store=store,
            protos_full=protos_full, seen_classes=seen, eval_batch_size=int(config["eval_batch_size"]),
            use_amp=use_amp, shrinkage=config.get("shrinkage"),
        )
        ablation_out[ab_name] = eval_view_suite(ab_h.predict_views(tsvm_t, tsvm_i), dev_y, seen, ds["early_classes"], ds["late_classes"])

    bootstrap: dict[str, Any] = {}
    anchor = boot_map["tsvm"]["zero_mnar"]
    for label, ref in [
        ("tsvm:zero_mnar_vs_tsvm:raw", boot_map["tsvm"]["raw"]),
        ("tsvm:zero_mnar_vs_tsvm:unlab_center", boot_map["tsvm"].get("unlab_center")),
        ("tsvm:zero_mnar_vs_tsvm:cal_bias", boot_map["tsvm"].get("cal_bias")),
        ("tsvm:zero_mnar_vs_dc_lora:cal_bias", boot_map["dc_lora"].get("cal_bias")),
    ]:
        if ref is not None:
            bootstrap[label] = paired_bootstrap_diff(anchor, ref, dev_y, seen, seed=seed)

    payload_out = {"schema": "mnarm-iccv-protocol-v1", "config": config, "arms": arms_out, "ablation_tsvm": ablation_out, "bootstrap": bootstrap, "profiling": profiler.records}
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
    base_ds = load_crisismmd_contracts(seq_inputs, images_dir, "P4", allow_test=False)
    vlm = vlm or MultimodalVLM(backbone, model_override or config["model_id"], device=device, rank=int(config["lora_rank"]), alpha=int(config["lora_alpha"]), revision=None if model_override else config.get("revision"), local_files_only=bool(model_override))
    protos_full, unseen_protos = vlm.encode_prompts(base_ds["orig_prompts"]), vlm.encode_prompts(base_ds["unseen_prompts"])
    store = store or PreprocessedTensorStore(base_ds["lookup"], vlm, max_cached_images=int(config.get("max_cached_images", 25000)), workers=int(config.get("workers", 4)))
    dev_cache: dict[tuple[str, ...], torch.Tensor] = {}

    per_proto: dict[str, dict[str, Any]] = {}
    for p in protos:
        p_out = out_dir / f"{backbone}_{p.lower()}"
        p_json = p_out / "ICCV_PROTOCOL_RESULTS.json"
        if resume and p_json.is_file():
            per_proto[p] = json.loads(p_json.read_text(encoding="utf8"))
        else:
            pcfg = {**config, "protocol": p, "experiment": f"iccv_same_bank_{backbone}_{p.lower()}"}
            per_proto[p] = run_single_protocol_same_bank(pcfg, seq_inputs, images_dir, p_out, vlm, protos_full, unseen_protos, store, dev_cache)
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
