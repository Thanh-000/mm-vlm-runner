"""CrisisMMD P1-P4 contract loader with strict split-overlap and dev-only guards."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from mnarm.core.io import sha256_bytes
from mnarm.data.tensor_store import MaskedRowsDataset, PreprocessedTensorStore

CRISISMMD_CLASSES = [
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "infrastructure_and_utility_damage",
    "affected_individuals",
    "injured_or_dead_people",
    "vehicle_damage",
    "missing_or_found_people",
]

ORIGINAL_PROMPTS = [
    "a social media post unrelated to humanitarian disaster response.",
    "a social media post with other relevant information about a disaster.",
    "a social media post about rescue, volunteering or donation efforts.",
    "a social media post about damage to infrastructure and utilities.",
    "a social media post about people affected by a disaster.",
    "a social media post about injured or dead people.",
    "a social media post about damaged vehicles.",
    "a social media post about missing or found people.",
]

UNSEEN_PROMPTS = [
    "an ordinary photo or update not describing a crisis or emergency event",
    "general breaking news, alerts, or background details during a calamity",
    "volunteers, emergency responders, shelters, or charitable relief supplies",
    "collapsed buildings, broken bridges, flooded roads, or downed power lines",
    "displaced residents or survivors impacted by an emergency situation",
    "casualties, wounded victims, medical stretchers, or loss of life",
    "wrecked cars, crushed trucks, or damaged boats after a disaster",
    "missing persons alerts, search efforts, or reunited family members",
]


def uid_hash(*parts: Any) -> str:
    """Compute canonical SHA-256 sample identifier matching CrisisMMD manifests."""
    raw = json.dumps(list(parts), separators=(",", ":"), ensure_ascii=False).encode("utf8")
    return sha256_bytes(raw)


def _row_uid(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("uid"))


def assert_zero_split_overlap(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Assert zero uid, image-path, or text-SHA overlap across every split pair."""
    splits = sorted(k for k, v in rows_by_split.items() if v is not None)
    uid_sets = {s: {_row_uid(r) for r in rows_by_split[s]} for s in splits}
    img_sets = {s: {str(r["image"]) for r in rows_by_split[s] if "image" in r} for s in splits}
    txt_sets = {s: {str(r["text_sha256"]) for r in rows_by_split[s] if "text_sha256" in r} for s in splits}

    audit: dict[str, int] = {}
    for idx_a, sp_a in enumerate(splits):
        for sp_b in splits[idx_a + 1 :]:
            uid_ov = len(uid_sets[sp_a] & uid_sets[sp_b])
            img_ov = len(img_sets[sp_a] & img_sets[sp_b])
            txt_ov = len(txt_sets[sp_a] & txt_sets[sp_b])
            audit[f"{sp_a}_{sp_b}_uid_overlap"] = uid_ov
            audit[f"{sp_a}_{sp_b}_image_overlap"] = img_ov
            audit[f"{sp_a}_{sp_b}_text_overlap"] = txt_ov
            if uid_ov or img_ov or txt_ov:
                raise ValueError(
                    f"Data leakage detected between {sp_a} and {sp_b}: "
                    f"uid={uid_ov}, image={img_ov}, text_sha256={txt_ov}"
                )
    return audit


def _read_crisismmd_tsv_texts(splits_zip: Path, split_name: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    with zipfile.ZipFile(splits_zip) as zf:
        matches = [
            name
            for name in zf.namelist()
            if not name.startswith("__MACOSX") and name.endswith(f"task_humanitarian_text_img_{split_name}.tsv")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected 1 TSV for split={split_name} in {splits_zip}, found {matches}")
        with zf.open(matches[0]) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t"):
                texts[uid_hash(row["tweet_id"], row["image_id"])] = row["tweet_text"]
    return texts


def extract_contracts_from_a2z_zip(a2z_zip: Path, dest_dir: Path) -> Path:
    """Extract and SHA-verify contract files from a full A2Z bundle or lite contract ZIP."""
    if not a2z_zip.is_file():
        raise FileNotFoundError(f"Contracts ZIP not found: {a2z_zip}")
    if not zipfile.is_zipfile(a2z_zip):
        size = a2z_zip.stat().st_size
        with a2z_zip.open("rb") as fh:
            head = fh.read(120)
        head_low = head.lower()
        if b"<html" in head_low or b"<!doctype" in head_low:
            raise zipfile.BadZipFile(
                f"{a2z_zip} is an HTML page (size={size} B), not a ZIP "
                "(common when curling a Google Drive share URL; use gdown or Drive mount)."
            )
        raise zipfile.BadZipFile(
            f"{a2z_zip} is not a valid ZIP file (size={size} B, header={head[:32]!r}); "
            "upload may be truncated or wrong file."
        )

    prefix = "dense_bundle/kit/inputs/"
    required = ("manifest.json", "PLANS.json", "partition_contract.json", "splits.zip")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a2z_zip) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/") and not n.endswith("/")]
        if "BUNDLE_MANIFEST.json" in names:
            bm = json.loads(zf.read("BUNDLE_MANIFEST.json"))
            for rel, meta in bm["files"].items():
                if not rel.startswith(prefix):
                    continue
                blob = zf.read(f"blobs/{meta['sha256']}")
                if sha256_bytes(blob) != meta["sha256"]:
                    raise ValueError(f"SHA mismatch extracting {rel} from {a2z_zip}")
                sub = rel.split(prefix, 1)[1]
                out_path = dest_dir / sub
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(blob)
        else:
            by_base = {Path(n).name: n for n in names}
            expected_shas: dict[str, str] = {}
            if "CONTRACT_SHA256.json" in by_base:
                expected_shas = json.loads(zf.read(by_base["CONTRACT_SHA256.json"]))
            for req in required:
                member = by_base.get(req)
                if member is None:
                    raise FileNotFoundError(f"Missing {req} inside lightweight contracts ZIP {a2z_zip}")
                blob = zf.read(member)
                if req in expected_shas and sha256_bytes(blob) != expected_shas[req]:
                    raise ValueError(f"SHA mismatch extracting {req} from {a2z_zip}")
                (dest_dir / req).write_bytes(blob)

    missing = [req for req in required if not (dest_dir / req).is_file()]
    if missing:
        raise FileNotFoundError(f"Extraction from {a2z_zip} missing required files: {missing}")
    return dest_dir


def load_crisismmd_contracts(
    seq_inputs_dir: Path,
    images_dir: Path,
    protocol: str,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Load CrisisMMD P1-P4 stage contracts with zero split leakage and hard test isolation."""
    if protocol not in ("P1", "P2", "P3", "P4"):
        raise ValueError(f"Unsupported CrisisMMD protocol: {protocol}")

    manifest_bytes = (seq_inputs_dir / "manifest.json").read_bytes()
    plans_bytes = (seq_inputs_dir / "PLANS.json").read_bytes()
    pc_path = seq_inputs_dir / "partition_contract.json"
    if not pc_path.is_file():
        raise FileNotFoundError(f"Missing partition_contract.json in {seq_inputs_dir}")
    pc_bytes = pc_path.read_bytes()

    manifest = json.loads(manifest_bytes)
    plans = json.loads(plans_bytes)[protocol]
    pc = json.loads(pc_bytes)
    overlap_audit = assert_zero_split_overlap(manifest["rows"])

    splits_zip = seq_inputs_dir / "splits.zip"
    active_splits = ("train", "dev", "test") if allow_test else ("train", "dev")
    texts_by_split = {sp: _read_crisismmd_tsv_texts(splits_zip, sp) for sp in active_splits}

    lookup: dict[str, dict[str, Any]] = {}
    for split_name in active_splits:
        tmap = texts_by_split[split_name]
        for row in manifest["rows"][split_name]:
            sid = row["sample_id"]
            txt = tmap[sid]
            if sha256_bytes(txt.encode("utf8")) != row["text_sha256"]:
                raise ValueError(f"Text SHA mismatch on {split_name} uid={sid}")
            img_p = images_dir / row["image"]
            if not img_p.is_file():
                raise FileNotFoundError(f"Missing CrisisMMD image for {split_name}: {img_p}")
            lookup[sid] = {"uid": sid, "label": int(row["label"]), "text": txt, "image": img_p}

    dev_rows = [{"uid": r["sample_id"], "label": int(r["label"]), "mask": ["text", "image"]} for r in manifest["rows"]["dev"]]
    test_rows = (
        [{"uid": r["sample_id"], "label": int(r["label"]), "mask": ["text", "image"]} for r in manifest["rows"]["test"]]
        if allow_test
        else None
    )

    tr_by_uid = {r["sample_id"]: r for r in manifest["rows"]["train"]}
    stage_plans: list[dict[str, Any]] = []
    joint_so_far: dict[str, dict[str, Any]] = {}
    for t_idx in range(4):
        st = t_idx + 1
        pl = plans[t_idx]
        spec = pc["protocols"][protocol][t_idx]
        fit_t = [
            {"uid": uid, "label": int(tr_by_uid[uid]["label"]), "mask": list(spec["modalities"])}
            for uid in spec["train_ids"]
        ]
        buf_before = [
            {"uid": r["uid"], "label": int(r["label"]), "mask": list(r["mask"])}
            for r in (plans[t_idx - 1]["buffer"] if t_idx > 0 else [])
        ]
        buf_after = [{"uid": r["uid"], "label": int(r["label"]), "mask": list(r["mask"])} for r in pl["buffer"]]
        cal_latest = [
            {"uid": r["uid"], "label": int(r["label"]), "mask": list(r["mask"])}
            for r in pl["calibration_latest_views"]
        ]
        joint_before = [dict(v) for v in joint_so_far.values()]
        for rec in fit_t:
            joint_so_far[rec["uid"]] = dict(rec)
        if protocol in ("P1", "P2") and st >= 2:
            for rec in joint_so_far.values():
                rec["mask"] = ["text", "image"]
        stage_plans.append(
            {
                "stage": st,
                "new_classes": sorted({r["label"] for r in fit_t}),
                "seen_classes": [int(c) for c in pl["seen_classes"]],
                "fit": fit_t,
                "buffer_before": buf_before,
                "buffer_after": buf_after,
                "joint_before": joint_before,
                "calibration_latest_views": cal_latest,
            }
        )

    return {
        "name": "crisismmd",
        "protocol": protocol,
        "allow_test": bool(allow_test),
        "num_classes": 8,
        "num_stages": 4,
        "class_names": list(CRISISMMD_CLASSES),
        "orig_prompts": list(ORIGINAL_PROMPTS),
        "unseen_prompts": list(UNSEEN_PROMPTS),
        "early_classes": [0, 3, 5] if protocol in ("P1", "P2") else [0, 1, 4, 5],
        "late_classes": [1, 2, 4, 6, 7] if protocol in ("P1", "P2") else [2, 3, 6, 7],
        "stages": stage_plans,
        "dev": dev_rows,
        "test": test_rows,
        "lookup": lookup,
        "hashes": {
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "plans_sha256": sha256_bytes(plans_bytes),
            "partition_contract_sha256": sha256_bytes(pc_bytes),
            **overlap_audit,
        },
    }


__all__ = [
    "CRISISMMD_CLASSES",
    "MaskedRowsDataset",
    "ORIGINAL_PROMPTS",
    "PreprocessedTensorStore",
    "UNSEEN_PROMPTS",
    "assert_zero_split_overlap",
    "extract_contracts_from_a2z_zip",
    "load_crisismmd_contracts",
    "uid_hash",
]
