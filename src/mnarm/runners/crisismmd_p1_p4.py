"""Shared-VLM CrisisMMD P1-P4 dev-only benchmark runner (0 D_cal labels, 0 exemplars)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mnarm.core import load_yaml, seed_everything, write_json_atomic
from mnarm.data import PreprocessedTensorStore, load_crisismmd_contracts
from mnarm.models import MultimodalVLM
from mnarm.runners.pilot_f1 import run_pilot_f1

DEFAULT_PROTOCOLS = ("P4", "P3", "P2", "P1")
SUMMARY_ROWS = (
    ("std_ce_tsvm", "raw_proto"),
    ("std_ce_tsvm", "bayes_cm_vmf_0lab"),
    ("std_ce_ta_sum", "bayes_cm_vmf_0lab"),
    ("l_mnar_tsvm", "bayes_cm_vmf_0lab"),
    ("std_ce_joint", "bayes_cm_vmf_0lab"),
)


def _extract_entry(res: dict[str, Any], arm: str, variant: str) -> dict[str, Any] | None:
    arec = res.get("arms", {}).get(arm)
    if not arec or variant not in arec.get("variants", {}):
        return None
    vrec = arec["variants"][variant]
    return {
        "text_f1": float(vrec["text"]["macro_f1"]),
        "image_f1": float(vrec["image"]["macro_f1"]),
        "paired_f1": float(vrec["paired"]["macro_f1"]),
        "view3_mean_f1": float(vrec["summary"]["view3_mean_f1"]),
        "paired_ece": float(vrec["paired"]["ece"]),
        "theta_t_deg": float(arec["cone_drift"]["text"]["theta_deg"]),
        "theta_i_deg": float(arec["cone_drift"]["image"]["theta_deg"]),
    }


def _write_suite_summary(config: dict[str, Any], per_proto: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    backbone = str(config["backbone"])
    summary_rows: dict[str, dict[str, Any]] = {}
    lines = [
        f"# CrisisMMD P1-P4 Summary — {backbone} (dev-Only, 0 D_cal Labels, 0 Exemplars)\n",
        "| Protocol | Method (`arm:variant`) | Text F1 | Image F1 | Paired F1 | 3-View Mean | ECE (P) | theta_T / theta_I |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for proto, res in per_proto.items():
        summary_rows[proto] = {"verdict": res.get("verdict"), "entries": {}}
        for arm, var in SUMMARY_ROWS:
            ent = _extract_entry(res, arm, var)
            if ent is None:
                continue
            summary_rows[proto]["entries"][f"{arm}:{var}"] = ent
            lines.append(
                f"| `{proto}` | `{arm}:{var}` | {ent['text_f1']*100:.2f} | {ent['image_f1']*100:.2f} | "
                f"**{ent['paired_f1']*100:.2f}** | {ent['view3_mean_f1']*100:.2f} | {ent['paired_ece']:.4f} | "
                f"{ent['theta_t_deg']:.1f}/{ent['theta_i_deg']:.1f} |"
            )
    md_text = "\n".join(lines) + "\n"
    if "$" in md_text:
        raise AssertionError("Forbidden dollar character detected in CrisisMMD P1-P4 summary")
    (out_dir / f"CRISISMMD_P1_P4_{backbone}_SUMMARY.md").write_text(md_text, encoding="utf8")
    payload = {
        "schema": "mnarm-crisismmd-p1-p4-v1",
        "backbone": backbone,
        "allow_test": False,
        "protocols": summary_rows,
    }
    write_json_atomic(out_dir / f"CRISISMMD_P1_P4_{backbone}_RESULTS.json", payload)
    return payload


def run_crisismmd_p1_p4(
    config: dict[str, Any],
    seq_inputs: Path,
    images_dir: Path,
    out_dir: Path,
    protocols: list[str] | tuple[str, ...] | None = None,
    include_lmnar_p1_p2: bool = False,
    resume: bool = False,
    model_override: str | None = None,
    vlm: MultimodalVLM | None = None,
    protos_full: torch.Tensor | None = None,
    store: PreprocessedTensorStore | None = None,
) -> dict[str, Any]:
    """Run CrisisMMD protocols sequentially with a single shared VLM, store, and dev feature cache."""
    if bool(config.get("allow_test", False)):
        raise ValueError("CrisisMMD dev suite strictly forbids allow_test=True!")
    protos = [str(p).upper() for p in (protocols or DEFAULT_PROTOCOLS)]
    for p in protos:
        if p not in ("P1", "P2", "P3", "P4"):
            raise ValueError(f"Unsupported protocol={p!r}")

    seed_everything(int(config["seed"]), deterministic=bool(config.get("deterministic", True)))
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    backbone = str(config["backbone"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # All P1-P4 protocols share the exact train+dev lookup rows loaded under P4.
    base_ds = load_crisismmd_contracts(seq_inputs, images_dir, "P4", allow_test=False)
    vlm = vlm or MultimodalVLM(
        backbone, model_override or config["model_id"], device=device,
        rank=int(config["lora_rank"]), alpha=int(config["lora_alpha"]),
        revision=None if model_override else config.get("revision"), local_files_only=bool(model_override),
    )
    protos_full = vlm.encode_prompts(base_ds["orig_prompts"]) if protos_full is None else protos_full
    store = store or PreprocessedTensorStore(
        base_ds["lookup"], vlm,
        max_cached_images=int(config.get("max_cached_images", 25000)),
        workers=int(config.get("workers", 4)),
    )
    dev_cache: dict[tuple[str, ...], torch.Tensor] = {}

    per_proto: dict[str, dict[str, Any]] = {}
    for proto in protos:
        proto_out = out_dir / f"{backbone}_{proto.lower()}"
        res_file = proto_out / "PILOT_F1_DEV_RESULTS.json"
        if resume and res_file.is_file():
            print(f"[RESUME] Loading completed {backbone} {proto} from {res_file}", flush=True)
            proto_res = json.loads(res_file.read_text(encoding="utf8"))
        else:
            proto_cfg = dict(config)
            proto_cfg["protocol"] = proto
            proto_cfg["experiment"] = f"crisismmd_{backbone}_{proto.lower()}"
            proto_cfg["objectives"] = (
                ["std_ce", "l_mnar"] if (proto in ("P3", "P4") or include_lmnar_p1_p2) else ["std_ce"]
            )
            proto_res = run_pilot_f1(
                proto_cfg, seq_inputs, images_dir, proto_out,
                model_override=model_override, vlm=vlm, protos_full=protos_full,
                store=store, dev_cache=dev_cache,
            )
        per_proto[proto] = proto_res
        _write_suite_summary(config, per_proto, out_dir)
    return _write_suite_summary(config, per_proto, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seq-inputs", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    ap.add_argument("--include-lmnar-p1-p2", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-path", type=str, default=None)
    args = ap.parse_args()
    summary = run_crisismmd_p1_p4(
        load_yaml(args.config), args.seq_inputs, args.images, args.out,
        protocols=args.protocols, include_lmnar_p1_p2=args.include_lmnar_p1_p2,
        resume=args.resume, model_override=args.model_path,
    )
    print(f"[CRISISMMD P1-P4 DONE] {summary['backbone']} -> {list(summary['protocols'].keys())}")


if __name__ == "__main__":
    main()
