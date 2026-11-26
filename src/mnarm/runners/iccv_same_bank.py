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
from mnarm.methods import PALRLSClassifier, StreamingClassBank, train_mg_clip_stage
from mnarm.models import MultimodalVLM
from mnarm.runners.iccv_eval import evaluate_protocol_splits
from mnarm.runners.iccv_report import render_protocol_report, write_ecosystem_master_summary, write_same_bank_suite_outputs
from mnarm.runners.stage_ops import collect_masked_cal_bundle, collect_train_features_by_modality, train_lora_stage


def _cached_base0_features(
    vlm: MultimodalVLM, records: list[dict[str, Any]], store: PreprocessedTensorStore, eval_bs: int, use_amp: bool,
    cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Encode frozen theta_0 features per uid aligned with make_batch_specs mask groups."""
    uncached = [r for r in records if ("text" in r["mask"] and r["uid"] not in cache["text"]) or ("image" in r["mask"] and r["uid"] not in cache["image"])]
    if uncached:
        ord_un = [r for _, grp in store.make_batch_specs(uncached, batch_size=max(1, len(uncached)), shuffle=False) for r in grp]
        tf_new = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=uncached, tensor_store=store, batch_size=eval_bs, use_amp=use_amp)
        t_pos = i_pos = 0
        for r in ord_un:
            if "text" in r["mask"] and tf_new["text_x"] is not None:
                cache["text"].setdefault(r["uid"], tf_new["text_x"][t_pos]); t_pos += 1
            if "image" in r["mask"] and tf_new["image_x"] is not None:
                cache["image"].setdefault(r["uid"], tf_new["image_x"][i_pos]); i_pos += 1
    tx, ty, ix, iy, pt, pi = [], [], [], [], [], []
    by_mask: dict[tuple[str, ...], tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]] = {}
    for views, grp in store.make_batch_specs(records, batch_size=max(1, len(records)), shuffle=False):
        gt = torch.stack([cache["text"][r["uid"]] for r in grp], 0) if "text" in views else None
        gi = torch.stack([cache["image"][r["uid"]] for r in grp], 0) if "image" in views else None
        gy = torch.tensor([int(r["label"]) for r in grp], dtype=torch.long)
        by_mask[views] = (gt, gi, gy)
        if gt is not None:
            tx.append(gt); ty.append(gy)
        if gi is not None:
            ix.append(gi); iy.append(gy)
        if gt is not None and gi is not None:
            pt.append(gt); pi.append(gi)
    return {
        "text_x": torch.cat(tx, 0) if tx else None, "text_y": torch.cat(ty, 0) if ty else None,
        "image_x": torch.cat(ix, 0) if ix else None, "image_y": torch.cat(iy, 0) if iy else None,
        "paired_t": torch.cat(pt, 0) if pt else None, "paired_i": torch.cat(pi, 0) if pi else None,
        "by_mask": by_mask,
    }


def run_single_protocol_same_bank(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path,
    vlm: MultimodalVLM, protos_full: torch.Tensor, unseen_protos_full: torch.Tensor,
    store: PreprocessedTensorStore, dev_cache: dict[tuple[str, ...], torch.Tensor],
    base0_cache: dict[str, dict[str, torch.Tensor]] | None = None,
    confirm_freeze: bool = False,
) -> dict[str, Any]:
    """Run all 14 author mergers + seq_ft(CSS-CT) + seq_replay256 + MG-CLIP + PAL-RLS + joint_oracle."""
    allow_test = bool(confirm_freeze or config.get("confirm_freeze", False))
    if bool(config.get("allow_test", False)) and not allow_test:
        raise ValueError("ICCV Same-Bank benchmark forbids allow_test=True prior to final freeze.")
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    use_amp = bool(config.get("use_amp", True)) and vlm.device == "cuda"
    vram_bytes = torch.cuda.get_device_properties(0).total_memory if (vlm.device == "cuda" and torch.cuda.is_available()) else 0
    eff_eval_bs = max(int(config["eval_batch_size"]), 1024 if vram_bytes >= 40 * (1024 ** 3) else (512 if vram_bytes >= 22 * (1024 ** 3) else int(config["eval_batch_size"])))
    b0_cache = base0_cache if base0_cache is not None else {"text": {}, "image": {}}
    ds = load_benchmark_contracts(str(config.get("dataset", "crisismmd")), seq_inputs, images_dir, config["protocol"], allow_test=allow_test, seed=seed)
    lock_cfg = {**config, "allow_test": allow_test, **({"method_freeze": "CSS-CT-v2.2-fair"} if allow_test else {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "RUN_LOCK.json", build_run_lock(lock_cfg, repo_root=Path(__file__).resolve().parents[3], dataset_hashes=ds["hashes"]))

    st4 = ds["stages"][-1]
    seen = st4["seen_classes"]
    dev_rows = [r for r in ds["dev"] if int(r["label"]) in set(seen)]

    exp_deltas: list[dict[str, torch.Tensor]] = []
    seq_mm_deltas: list[dict[str, torch.Tensor]] = []
    bank, seq_bank, seq_mm_bank, seq_lmnar_bank, seq_rep_bank, mg_bank, joint_bank = (
        StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(), StreamingClassBank(),
    )
    pal = PALRLSClassifier(vlm.embed_dim, len(ds["orig_prompts"]), lam=1.0, seed=seed, auto_lam=True)
    seq_ft_state = seq_mm_state = seq_lmnar_state = seq_rep_state = mg_state = None
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

        for m_views, (m_t, m_i, m_y) in tf_b0["by_mask"].items():
            pal.update(m_t, m_i, m_y, m_views)

        seq_ft_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_ft_state, records=st["fit"], protos_full=protos_full, seen_classes=seen_so_far,
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 1000, use_amp=use_amp, tensor_store=store,
        )
        tf_sq = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=seq_ft_state, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        seq_bank.update_stage(u_expert_t=tf_sq["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_sq["text_y"], u_expert_i=tf_sq["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_sq["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])

        seq_mm_state, _ = train_lora_stage(
            vlm=vlm, init_lora=seq_mm_state, records=st["fit"], protos_full=protos_full, seen_classes=new_c,
            objective="std_ce", stage_idx=t, epochs=int(config["epochs"]),
            effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
            lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), seed=seed + 1250, use_amp=use_amp, tensor_store=store,
        )
        seq_mm_deltas.append(vlm.extract_lora_delta(seq_mm_state))
        tf_smm = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=seq_mm_state, records=st["fit"], tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
        seq_mm_bank.update_stage(u_expert_t=tf_smm["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_smm["text_y"], u_expert_i=tf_smm["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_smm["image_y"], paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"])

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
            mg_lr = float(config.get("mg_clip_lr", max(float(config["lr"]), 5e-4)))
            mg_state, mg_heads = train_mg_clip_stage(
                vlm=vlm, prev_state=mg_state, mg_heads=mg_heads, records=st["fit"], protos_full=protos_full,
                old_classes=old_c, new_classes=new_c, stage_idx=t, epochs=int(config["epochs"]),
                effective_batch_size=int(config["effective_batch_size"]), micro_batch_size=int(config["micro_batch_size"]),
                lr=mg_lr, weight_decay=float(config["weight_decay"]), seed=seed + 3000, use_amp=use_amp, tensor_store=store,
                eval_batch_size=eff_eval_bs, orig_prompts=ds["orig_prompts"],
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
    tf_jn = collect_train_features_by_modality(vlm=vlm, kind="lora", payload=joint_state, records=joint_recs, tensor_store=store, batch_size=eff_eval_bs, use_amp=use_amp)
    tf_b0_jn = _cached_base0_features(vlm, joint_recs, store, eff_eval_bs, use_amp, b0_cache)
    joint_bank.update_stage(u_expert_t=tf_jn["text_x"], u_base_t=tf_b0_jn["text_x"], y_t=tf_jn["text_y"], u_expert_i=tf_jn["image_x"], u_base_i=tf_b0_jn["image_x"], y_i=tf_jn["image_y"], paired_base_t=tf_b0_jn["paired_t"], paired_base_i=tf_b0_jn["paired_i"])

    _, _, cal0_fx = collect_masked_cal_bundle(
        vlm=vlm, kind="lora", payload={}, cal_recs=st4["calibration_latest_views"], store=store,
        protos_full=protos_full, seen_classes=seen, batch_size=eff_eval_bs, use_amp=use_amp,
    )

    common_eval = dict(
        ds=ds, seen=seen, exp_deltas=exp_deltas, bank=bank, seq_ft_state=seq_ft_state, seq_bank=seq_bank,
        seq_mm_deltas=seq_mm_deltas, seq_mm_bank=seq_mm_bank, seq_lmnar_state=seq_lmnar_state, seq_lmnar_bank=seq_lmnar_bank,
        seq_rep_state=seq_rep_state, seq_rep_bank=seq_rep_bank, mg_state=mg_state, mg_bank=mg_bank, mg_heads=mg_heads,
        joint_state=joint_state, joint_recs=joint_recs, joint_bank=joint_bank, pal=pal, cal0_fx=cal0_fx,
        vlm=vlm, protos_full=protos_full, unseen_protos_full=unseen_protos_full, store=store, eff_eval_bs=eff_eval_bs, use_amp=use_amp,
    )
    splits_spec: list[tuple[str, list[dict[str, Any]], dict[str, Any], dict[tuple[str, ...], torch.Tensor]]] = [
        ("dev", dev_rows, {**config, "allow_test": False}, dev_cache),
    ]
    if allow_test and ds.get("test"):
        splits_spec.append(("test", [r for r in ds["test"] if int(r["label"]) in set(seen)], lock_cfg, {}))
    split_outs = evaluate_protocol_splits(splits=splits_spec, **common_eval)
    payload_out = split_outs["dev"]
    write_json_atomic(out_dir / "ICCV_PROTOCOL_RESULTS.json", payload_out)
    render_protocol_report(payload_out, out_dir / "ICCV_PROTOCOL_REPORT.md")
    if "test" in split_outs:
        write_json_atomic(out_dir / "ICCV_PROTOCOL_TEST_RESULTS.json", split_outs["test"])
        render_protocol_report(split_outs["test"], out_dir / "ICCV_PROTOCOL_TEST_REPORT.md")
    return payload_out


def _is_v21_valid(p_json: Path) -> dict[str, Any] | None:
    if not p_json.is_file():
        return None
    cached_p = json.loads(p_json.read_text(encoding="utf8"))
    arms = cached_p.get("arms", {})
    tsvm_rg = arms.get("tsvm", {}).get("regimes", {})
    ok = (
        cached_p.get("schema") == "mnarm-iccv-protocol-v2"
        and "seq_magmax" in arms
        and "ncm" in tsvm_rg
        and "oas_lda" in tsvm_rg
        and "balanced_ece" in tsvm_rg.get("raw", {}).get("paired", {})
        and "cone_scale_p" in tsvm_rg.get("zero_mnar", {}).get("diagnostics", {}).get("towers", {}).get("text", {})
    )
    return cached_p if ok else None


def run_iccv_same_bank(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path,
    protocols: list[str] | tuple[str, ...] | None = None, resume: bool = False,
    model_override: str | None = None, vlm: MultimodalVLM | None = None, store: PreprocessedTensorStore | None = None,
    write_suite: bool = True, confirm_freeze: bool = False,
) -> dict[str, Any]:
    """Run unified Same-Bank ICCV evaluation across requested protocols sharing 1 VLM and cache."""
    allow_test = bool(confirm_freeze or config.get("confirm_freeze", False))
    if bool(config.get("allow_test", False)) and not allow_test:
        raise ValueError("ICCV Same-Bank benchmark forbids allow_test=True prior to final freeze.")
    protos = [str(p).upper() for p in (protocols or ("P4", "P3", "P2", "P1"))]
    device, backbone = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")), str(config["backbone"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ds = load_benchmark_contracts(str(config.get("dataset", "crisismmd")), seq_inputs, images_dir, "P4", allow_test=allow_test, seed=int(config.get("seed", 11)))
    vlm = vlm or MultimodalVLM(backbone, model_override or config["model_id"], device=device, rank=int(config["lora_rank"]), alpha=int(config["lora_alpha"]), revision=None if model_override else config.get("revision"), local_files_only=bool(model_override))
    protos_full, unseen_protos = vlm.encode_prompts(base_ds["orig_prompts"]), vlm.encode_prompts(base_ds["unseen_prompts"])
    store = store or PreprocessedTensorStore(base_ds["lookup"], vlm, max_cached_images=int(config.get("max_cached_images", 25000)), workers=int(config.get("workers", 0)))
    dev_cache: dict[tuple[str, ...], torch.Tensor] = {}
    base0_cache: dict[str, dict[str, torch.Tensor]] = {"text": {}, "image": {}}

    per_proto: dict[str, dict[str, Any]] = {}
    per_proto_test: dict[str, dict[str, Any]] = {}
    for p in protos:
        p_out = out_dir / f"{backbone}_{p.lower()}"
        c_dev = _is_v21_valid(p_out / "ICCV_PROTOCOL_RESULTS.json") if resume else None
        c_tst = _is_v21_valid(p_out / "ICCV_PROTOCOL_TEST_RESULTS.json") if (resume and allow_test) else None
        if c_dev is not None and ((not allow_test) or (c_tst is not None)):
            per_proto[p] = c_dev
            if c_tst is not None:
                per_proto_test[p] = c_tst
        else:
            pcfg = {**config, "protocol": p, "experiment": f"iccv_same_bank_{config.get('dataset', 'crisismmd')}_{backbone}_{p.lower()}"}
            per_proto[p] = run_single_protocol_same_bank(pcfg, seq_inputs, images_dir, p_out, vlm, protos_full, unseen_protos, store, dev_cache, base0_cache=base0_cache, confirm_freeze=allow_test)
            if allow_test:
                loaded_t = _is_v21_valid(p_out / "ICCV_PROTOCOL_TEST_RESULTS.json")
                if loaded_t is not None:
                    per_proto_test[p] = loaded_t
        if write_suite:
            write_same_bank_suite_outputs(config, per_proto, out_dir, split_name="dev")
            if allow_test and per_proto_test:
                write_same_bank_suite_outputs({**config, "allow_test": True}, per_proto_test, out_dir, split_name="test")
    if write_suite:
        if allow_test and per_proto_test:
            write_same_bank_suite_outputs({**config, "allow_test": True}, per_proto_test, out_dir, split_name="test")
        return write_same_bank_suite_outputs(config, per_proto, out_dir, split_name="dev")
    return {"schema": "mnarm-iccv-same-bank-v2", "backbone": backbone, "allow_test": allow_test, "protocols": list(per_proto.keys())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seq-inputs", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["P4", "P3", "P2", "P1"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-suite-summary", action="store_true")
    ap.add_argument("--confirm-freeze", action="store_true")
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()
    summary = run_iccv_same_bank(load_yaml(args.config), args.seq_inputs, args.images, args.out, protocols=args.protocols, resume=args.resume, model_override=args.model_path, write_suite=not args.skip_suite_summary, confirm_freeze=args.confirm_freeze)
    p_done = list(summary["protocols"].keys()) if isinstance(summary["protocols"], dict) else list(summary["protocols"])
    print(f"[ICCV SAME-BANK DONE] {summary['backbone']} -> {p_done}")


if __name__ == "__main__":
    main()
