"""Pilot F1 falsification runner (dev-only, 0 D_cal labels, L_MNAR + SelfNormalizedBayesTransportHead)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from mnarm.core import build_run_lock, load_yaml, seed_everything, write_json_atomic
from mnarm.data import PreprocessedTensorStore, load_crisismmd_contracts
from mnarm.eval import Profiler, eval_view_suite, measure_cone_drift, paired_bootstrap_diff
from mnarm.methods import SelfNormalizedBayesTransportHead, StreamingClassBank, merge_task_deltas
from mnarm.models import MultimodalVLM, proto_logits
from mnarm.runners.stage_ops import (
    collect_train_features_by_modality,
    extract_paired_features_cached,
    train_lora_stage,
)

PILOT_V2_LMNAR_SPECS = (
    ("cross_modal_transport", "spectral_floor", "bayes_cm_sfloor_0lab"),
    ("hybrid_bridge", "spectral_floor", "bayes_hybrid_sfloor_0lab"),
    ("cross_modal_transport", "isotropic_vmf", "bayes_cm_vmf_0lab"),
    ("hybrid_bridge", "isotropic_vmf", "bayes_hybrid_vmf_0lab"),
    ("prompt_bridge", "isotropic_vmf", "bayes_prompt_vmf_0lab"),
    ("none_zero_shot", "isotropic_vmf", "bayes_none_vmf_0lab"),
)
PILOT_V2_STDCE_SPECS = (
    ("cross_modal_transport", "spectral_floor", "bayes_cm_sfloor_0lab"),
    ("cross_modal_transport", "isotropic_vmf", "bayes_cm_vmf_0lab"),
)


def _evaluate_arm(
    *,
    arm_name: str,
    objective: str,
    arm_type: str,
    kind: str,
    payload: dict[str, torch.Tensor],
    vlm: MultimodalVLM,
    protos_full: torch.Tensor,
    seen_classes: list[int],
    early_classes: list[int],
    late_classes: list[int],
    dev_rows: list[dict[str, Any]],
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

    cone_t = measure_cone_drift(dev_t, dev0_t)
    cone_i = measure_cone_drift(dev_i, dev0_i)
    zt_raw = proto_logits(dev_t, protos_full, seen_classes, vlm.logit_scale)
    zi_raw = proto_logits(dev_i, protos_full, seen_classes, vlm.logit_scale)
    raw_logits = {"text": zt_raw, "image": zi_raw, "paired": 0.5 * (zt_raw + zi_raw)}
    raw_eval = eval_view_suite(raw_logits, dev_y, seen_classes, early_classes, late_classes)

    variants: dict[str, Any] = {"raw_proto": raw_eval}
    logits_for_boot: dict[str, torch.Tensor] = {"raw_proto": raw_logits["paired"]}

    tf = None
    bank_stats = None
    if arm_type in ("joint", "seq_replay"):
        tf = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=train_recs,
            tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
    else:
        assert streaming_bank is not None
        f4_base = collect_train_features_by_modality(
            vlm=vlm, kind="lora", payload={}, records=fit4_recs,
            tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
        f4_arm = collect_train_features_by_modality(
            vlm=vlm, kind=kind, payload=payload, records=fit4_recs,
            tensor_store=store, batch_size=eval_batch_size, use_amp=use_amp,
        )
        bank_stats = streaming_bank.transport_to_target_stats(
            bridge_u0_t=f4_base["text_x"], bridge_um_t=f4_arm["text_x"],
            bridge_u0_i=f4_base["image_x"], bridge_um_i=f4_arm["image_x"],
            seen_classes=seen_classes, probe_y_t=f4_arm["text_y"], probe_y_i=f4_arm["image_y"],
        )

    specs = PILOT_V2_LMNAR_SPECS if objective == "l_mnar" else PILOT_V2_STDCE_SPECS
    for bmode, pcmode, vkey in specs:
        with profiler.track(f"fit_bayes:{arm_name}:{vkey}"):
            head = SelfNormalizedBayesTransportHead(bridge_mode=bmode, partial_cov_mode=pcmode, shrinkage=shrinkage)
            if tf is not None:
                head.fit(
                    train_feats_t=tf["text_x"], train_y_t=tf["text_y"],
                    train_feats_i=tf["image_x"], train_y_i=tf["image_y"],
                    paired_feats_t=tf["paired_t"], paired_feats_i=tf["paired_i"],
                    protos_full=protos_full, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
                )
            else:
                assert bank_stats is not None
                head.fit_from_sufficient_stats(
                    tower_stats=bank_stats["towers"],
                    paired_feats_t=bank_stats["paired"][0], paired_feats_i=bank_stats["paired"][1],
                    protos_full=protos_full, seen_classes=seen_classes, fallback_scale=vlm.logit_scale,
                )
        blogits = head.predict_views(dev_t, dev_i)
        variants[vkey] = {**eval_view_suite(blogits, dev_y, seen_classes, early_classes, late_classes), "diagnostics": head.diagnostics}
        logits_for_boot[vkey] = blogits["paired"]

    return {"objective": objective, "cone_drift": {"text": cone_t, "image": cone_i}, "variants": variants}, logits_for_boot


def _render_report(results: dict[str, Any], out_path: Path) -> None:
    cfg = results["config"]
    verdict = results["verdict"]
    lines = [
        f"# Pilot F1 Report — {cfg['backbone']} {cfg['protocol']} (dev-Only, 0 D_cal Labels)\n",
        f"- **Verdict:** `{verdict['status']}` (Gate 1 Collider Kill: `{verdict['gate1_pass']}`, Gate 2 0-Label Parity/Win: `{verdict['gate2_pass']}`)",
        f"- **Max theta_{str(verdict.get('mnar_tower', 'text'))[0].upper()} (replay/joint vs tsvm):** `{verdict['max_lmnar_theta_t_deg']:.2f} deg` vs `{verdict.get('tsvm_theta_deg', verdict['max_lmnar_theta_t_deg']):.2f} deg` (target `< {cfg['gates']['max_theta_deg']:.1f} deg`)",
        f"- **Best 0-Label H_ZERO-MNAR:** `{verdict['best_arm']}` / `{verdict['best_variant']}` -> Paired Macro-F1 = **`{verdict['best_paired_f1']*100:.2f}%`**, 3-View Mean = **`{verdict['best_view3_mean_f1']*100:.2f}%`**\n",
        "| Arm | Variant | Text F1 | Image F1 | Paired F1 | 7-Cls F1 | 3-View Mean | ECE (P) | theta_T / theta_I |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, arec in results["arms"].items():
        th_t = arec["cone_drift"]["text"]["theta_deg"]
        th_i = arec["cone_drift"]["image"]["theta_deg"]
        for vname, vrec in arec["variants"].items():
            f7 = vrec["paired"]["macro_f1_7cls"]
            f7s = f"{f7*100:.2f}" if f7 is not None else "--"
            lines.append(
                f"| `{arm_name}` | `{vname}` | {vrec['text']['macro_f1']*100:.2f} | {vrec['image']['macro_f1']*100:.2f} | "
                f"**{vrec['paired']['macro_f1']*100:.2f}** | {f7s} | {vrec['summary']['view3_mean_f1']*100:.2f} | "
                f"{vrec['paired']['ece']:.4f} | {th_t:.1f}/{th_i:.1f} |"
            )
    lines.append("\n## Paired Bootstrap Comparisons vs Standard-CE Raw Baseline\n")
    for arm_name, brec in results.get("bootstrap_vs_std_ce", {}).items():
        lines.append(
            f"- `{arm_name}`: delta = `{brec['delta_macro_f1']*100:+.2f}` pts, "
            f"95% CI = `[{brec['ci95_low']*100:+.2f}, {brec['ci95_high']*100:+.2f}]`, p = `{brec['p_value']:.4f}`"
        )
    text = "\n".join(lines) + "\n"
    if "$" in text:
        raise AssertionError("Forbidden dollar character detected in PILOT_F1_REPORT.md")
    out_path.write_text(text, encoding="utf8")


def run_pilot_f1(
    config: dict[str, Any], seq_inputs: Path, images_dir: Path, out_dir: Path, model_override: str | None = None,
    vlm: MultimodalVLM | None = None, protos_full: torch.Tensor | None = None,
    store: PreprocessedTensorStore | None = None, dev_cache: dict[tuple[str, ...], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Execute strict dev-only Pilot F1 comparison of Standard CE vs H_ZERO-MNAR."""
    if bool(config.get("allow_test", False)):
        raise ValueError("Pilot F1 strictly forbids allow_test=True! Only Stage-frozen final runs may open test.")

    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(config.get("use_amp", True)) and device == "cuda"

    ds = load_crisismmd_contracts(seq_inputs, images_dir, config["protocol"], allow_test=False)
    vlm = vlm or MultimodalVLM(
        config["backbone"], model_override or config["model_id"], device=device,
        rank=int(config["lora_rank"]), alpha=int(config["lora_alpha"]),
        revision=None if model_override else config.get("revision"), local_files_only=bool(model_override),
    )
    protos_full = vlm.encode_prompts(ds["orig_prompts"]) if protos_full is None else protos_full
    store = store or PreprocessedTensorStore(ds["lookup"], vlm, max_cached_images=int(config.get("max_cached_images", 25000)), workers=int(config.get("workers", 4)))
    profiler = Profiler()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "RUN_LOCK.json", build_run_lock(config, repo_root=Path(__file__).resolve().parents[3], dataset_hashes=ds["hashes"]))

    st4 = ds["stages"][-1]
    seen = st4["seen_classes"]
    dev_rows = [r for r in ds["dev"] if int(r["label"]) in set(seen)]
    dev_labels = [int(r["label"]) for r in dev_rows]
    t0 = time.perf_counter()
    dev_cache = {} if dev_cache is None else dev_cache
    dev0_t, dev0_i, dev_y = extract_paired_features_cached(
        vlm=vlm, kind="lora", payload={}, records=dev_rows, tensor_store=store,
        batch_size=int(config["eval_batch_size"]), cache=dev_cache, use_amp=use_amp,
    )
    print(f"[INIT] {config['backbone']} {config['protocol']} ready on {device} ({time.perf_counter() - t0:.1f}s) | dev={len(dev_rows)}", flush=True)

    objectives = list(config.get("objectives", ["std_ce", "l_mnar"]))
    mnar_tw = "image" if config["protocol"] in ("P1", "P3") else "text"
    arms_out: dict[str, Any] = {}
    boot_logits: dict[str, torch.Tensor] = {}
    traj: dict[str, list[dict[str, Any]]] = {f"{obj}_seq_replay": [] for obj in objectives}
    for obj in objectives:
        exp_deltas: list[dict[str, torch.Tensor]] = []
        bank = StreamingClassBank()
        rep_state: dict[str, torch.Tensor] | None = None
        for st in ds["stages"]:
            t = int(st["stage"])
            exp_s, _ = train_lora_stage(
                vlm=vlm, init_lora=None, records=st["fit"], protos_full=protos_full,
                seen_classes=st["new_classes"], objective=obj, stage_idx=t,
                epochs=int(config["epochs"]), effective_batch_size=int(config["effective_batch_size"]),
                micro_batch_size=int(config["micro_batch_size"]), lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]), seed=seed + (0 if obj == "std_ce" else 5000),
                use_amp=use_amp, tensor_store=store,
            )
            exp_deltas.append(vlm.extract_lora_delta(exp_s))
            tf_st = collect_train_features_by_modality(
                vlm=vlm, kind="lora", payload=exp_s, records=st["fit"],
                tensor_store=store, batch_size=int(config["eval_batch_size"]), use_amp=use_amp,
            )
            tf_b0 = collect_train_features_by_modality(
                vlm=vlm, kind="lora", payload={}, records=st["fit"],
                tensor_store=store, batch_size=int(config["eval_batch_size"]), use_amp=use_amp,
            )
            bank.update_stage(
                u_expert_t=tf_st["text_x"], u_base_t=tf_b0["text_x"], y_t=tf_st["text_y"],
                u_expert_i=tf_st["image_x"], u_base_i=tf_b0["image_x"], y_i=tf_st["image_y"],
                paired_base_t=tf_b0["paired_t"], paired_base_i=tf_b0["paired_i"],
            )
            rep_state, _ = train_lora_stage(
                vlm=vlm, init_lora=rep_state, records=st["fit"] + st["buffer_before"], protos_full=protos_full,
                seen_classes=st["seen_classes"], objective=obj, stage_idx=t,
                epochs=int(config["epochs"]), effective_batch_size=int(config["effective_batch_size"]),
                micro_batch_size=int(config["micro_batch_size"]), lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]), seed=seed + 2000 + (0 if obj == "std_ce" else 5000),
                use_amp=use_amp, tensor_store=store,
            )
            seen_set = set(st["seen_classes"])
            keep_idx = torch.tensor([i for i, yv in enumerate(dev_labels) if yv in seen_set], dtype=torch.long)
            dt_all, di_all, dy_all = extract_paired_features_cached(
                vlm=vlm, kind="lora", payload=rep_state, records=dev_rows,
                tensor_store=store, batch_size=int(config["eval_batch_size"]), cache=dev_cache, use_amp=use_amp,
            )
            dt_s, di_s, dy_s = dt_all.index_select(0, keep_idx), di_all.index_select(0, keep_idx), dy_all.index_select(0, keep_idx)
            st_z = {"text": proto_logits(dt_s, protos_full, st["seen_classes"], vlm.logit_scale), "image": proto_logits(di_s, protos_full, st["seen_classes"], vlm.logit_scale)}
            st_z["paired"] = 0.5 * (st_z["text"] + st_z["image"])
            st_f1 = eval_view_suite(st_z, dy_s, st["seen_classes"])["paired"]["macro_f1"]
            st_th = measure_cone_drift(dt_s, dev0_t.index_select(0, keep_idx))["theta_deg"]
            traj[f"{obj}_seq_replay"].append({"stage": t, "raw_paired_f1": st_f1, "theta_t_deg": st_th})
            print(f"[{obj} stage {t}/4] seq_replay raw_paired={st_f1*100:.2f}% | theta_T={st_th:.2f} deg ({time.perf_counter() - t0:.1f}s)", flush=True)

        joint_recs = st4["fit"] + st4["joint_before"]
        joint_state, _ = train_lora_stage(
            vlm=vlm, init_lora=None, records=joint_recs, protos_full=protos_full,
            seen_classes=seen, objective=obj, stage_idx=4,
            epochs=int(config["epochs"]), effective_batch_size=int(config["effective_batch_size"]),
            micro_batch_size=int(config["micro_batch_size"]), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]), seed=seed + 4000 + (0 if obj == "std_ce" else 5000),
            use_amp=use_amp, tensor_store=store,
        )
        print(f"[{obj} joint] finished ({time.perf_counter() - t0:.1f}s)", flush=True)
        arm_specs = [
            (f"{obj}_tsvm", "merge", "delta", merge_task_deltas(exp_deltas, "tsvm"), [], bank),
            (f"{obj}_ta_sum", "merge", "delta", merge_task_deltas(exp_deltas, "ta_sum"), [], bank),
            (f"{obj}_seq_replay", "seq_replay", "lora", rep_state, st4["fit"] + st4["buffer_before"], None),
            (f"{obj}_joint", "joint", "lora", joint_state, joint_recs, None),
        ]
        for aname, atype, akind, apayload, tr_recs, bnk in arm_specs:
            arec, alogits = _evaluate_arm(
                arm_name=aname, objective=obj, arm_type=atype, kind=akind, payload=apayload,
                vlm=vlm, protos_full=protos_full, seen_classes=seen,
                early_classes=ds["early_classes"], late_classes=ds["late_classes"],
                dev_rows=dev_rows, dev_cache=dev_cache, dev0_t=dev0_t, dev0_i=dev0_i,
                train_recs=tr_recs, streaming_bank=bnk, fit4_recs=st4["fit"], store=store,
                eval_batch_size=int(config["eval_batch_size"]), use_amp=use_amp,
                shrinkage=config.get("shrinkage"), profiler=profiler,
            )
            arms_out[aname] = arec
            for vk, lg in alogits.items():
                boot_logits[f"{aname}:{vk}"] = lg
            best_v_f1 = max(v["paired"]["macro_f1"] for v in arec["variants"].values())
            print(f"[EVAL {aname}] best_paired={best_v_f1*100:.2f}% | theta_T={arec['cone_drift']['text']['theta_deg']:.2f} deg ({time.perf_counter() - t0:.1f}s)", flush=True)

    best_arm, best_var, best_p, best_3m = None, None, -1.0, -1.0
    best_lmnar_p: float | None = None
    for aname, arec in arms_out.items():
        for vname, vrec in arec["variants"].items():
            if not vname.startswith("bayes_") or "bayes_none_" in vname or "bayes_prompt_" in vname:
                continue
            p_f1 = vrec["paired"]["macro_f1"]
            if aname.startswith("l_mnar"):
                best_lmnar_p = p_f1 if best_lmnar_p is None else max(best_lmnar_p, p_f1)
            if p_f1 > best_p:
                best_arm, best_var, best_p, best_3m = aname, vname, p_f1, vrec["summary"]["view3_mean_f1"]

    boot_vs_ce: dict[str, Any] = {}
    for aname in arms_out:
        for prefix, btags in (("std_ce", ("bayes_cm_vmf_0lab",)), ("l_mnar", ("bayes_cm_sfloor_0lab", "bayes_cm_vmf_0lab"))):
            if not aname.startswith(f"{prefix}_"):
                continue
            suffix = aname[len(prefix) + 1 :]
            if prefix == "std_ce" and suffix not in ("tsvm", "ta_sum"):
                continue
            ref_key = f"std_ce_{suffix}:raw_proto"
            for btag in btags:
                if ref_key in boot_logits and f"{aname}:{btag}" in boot_logits:
                    boot_vs_ce[f"{aname}:{btag}_vs_{suffix}_raw"] = paired_bootstrap_diff(boot_logits[f"{aname}:{btag}"], boot_logits[ref_key], dev_y, seen, seed=seed)

    if "l_mnar_seq_replay" in arms_out and "l_mnar_joint" in arms_out:
        max_th = max(arms_out[a]["cone_drift"][mnar_tw]["theta_deg"] for a in ("l_mnar_seq_replay", "l_mnar_joint"))
    else:
        max_th = arms_out["std_ce_tsvm"]["cone_drift"][mnar_tw]["theta_deg"]
    tsvm_th = arms_out["l_mnar_tsvm" if "l_mnar_tsvm" in arms_out else "std_ce_tsvm"]["cone_drift"][mnar_tw]["theta_deg"]
    gates = config["gates"]
    g1 = max_th < float(gates["max_theta_deg"])
    g2 = best_p >= float(gates["min_paired_f1"]) and best_3m >= float(gates["min_view3_mean_f1"])
    verdict = {
        "status": "PASS" if (g1 and g2) else ("PARTIAL" if (g1 or g2) else "FAIL"),
        "gate1_pass": g1, "gate2_pass": g2, "max_lmnar_theta_t_deg": float(max_th),
        "tsvm_theta_deg": float(tsvm_th), "mnar_tower": mnar_tw,
        "best_arm": best_arm, "best_variant": best_var, "best_paired_f1": float(best_p),
        "best_view3_mean_f1": float(best_3m), "best_lmnar_paired_f1": best_lmnar_p,
    }
    payload_out = {"schema": "mnarm-pilot-f1-v1", "config": config, "verdict": verdict, "arms": arms_out, "stage_trajectory": traj, "bootstrap_vs_std_ce": boot_vs_ce, "profiling": profiler.records}
    write_json_atomic(out_dir / "PILOT_F1_DEV_RESULTS.json", payload_out)
    _render_report(payload_out, out_dir / "PILOT_F1_REPORT.md")
    return payload_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seq-inputs", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()
    res = run_pilot_f1(load_yaml(args.config), args.seq_inputs, args.images, args.out, model_override=args.model_path)
    print(f"[PILOT F1 DONE] {res['verdict']}")


if __name__ == "__main__":
    main()
