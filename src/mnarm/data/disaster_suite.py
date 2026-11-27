"""Multi-altitude disaster benchmarks (DMD UCI #456, DisasterVQA-Incidents, DisasterM3 NeurIPS'25)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mnarm.data.archive_ops import aria2_download_file, extract_archive_safe, hf_lfs_batch_download
from mnarm.data.crisismmd import load_crisismmd_contracts
from mnarm.data.general_suite import (
    SCIENCEQA_CLASSES,
    SCIENCEQA_ORIG_PROMPTS,
    SCIENCEQA_UNSEEN_PROMPTS,
    scan_scienceqa_mm8_records,
)
from mnarm.data.protocol_builder import build_p1_p4_contracts

INCIDENTS_HF_SHA = "f341ce34814a93bad8abfe73da898a309116a59e"
DISASTER_M3_HF_SHA = "1da6da73fad6bd100454c099edd0cee972823810"
DMD_UCI_URL = "https://archive.ics.uci.edu/static/public/456/multimodal+damage+identification+for+humanitarian+computing.zip"

DMD_CLASSES = ["fires", "floods", "natural_landscape", "infrastructural", "human_damage", "non_damage"]
DMD_ORIG_PROMPTS = [
    "a disaster post depicting active fires, smoke plumes, or wildfire damage.",
    "a disaster post depicting flooded streets, rising water, or inundation.",
    "a disaster post depicting landslide, storm, or damaged natural landscape.",
    "a disaster post depicting collapsed buildings, bridges, or infrastructural damage.",
    "a disaster post depicting injured people, victims, or human casualties.",
    "a social media post unrelated to physical disaster damage or casualties.",
]
DMD_UNSEEN_PROMPTS = [
    "flames burning structures or heavy wildfire smoke in the scene",
    "submerged roads, floodwater currents, or inundated neighborhoods",
    "uprooted trees, mudslides, or storm-damaged outdoor terrain",
    "shattered masonry, broken bridges, or destroyed civic utilities",
    "civilians needing medical aid or emergency trauma rescue",
    "normal daily photo without emergency destruction or crisis",
]

INCIDENTS_CLASSES = ["earthquake", "fire", "accidents", "flood", "storm", "other_disaster"]
INCIDENTS_ORIG_PROMPTS = [
    "an emergency incident scene depicting earthquake rubble and seismic structural collapse.",
    "an emergency incident scene depicting fire flames, smoke, or burned structures.",
    "an emergency incident scene depicting transportation crash or industrial accidents.",
    "an emergency incident scene depicting floodwater inundation and submerged streets.",
    "an emergency incident scene depicting severe storm winds, heavy rain, or debris.",
    "an emergency incident scene depicting landslides, hurricanes, or other hazardous disasters.",
]
INCIDENTS_UNSEEN_PROMPTS = [
    "seismic ruins, cracked concrete slabs, and collapsed walls",
    "blazing flames, thick black smoke, or firefighting response",
    "wrecked vehicles, road collisions, or mechanical mishaps",
    "inundated roadways, deep water levels, and stranded boats",
    "violent weather winds, blown roofs, or downed power poles",
    "mudslides, tropical cyclone impacts, or general hazard emergencies",
]

M3_CLASSES = ["earthquake", "wildfire", "tornado", "hurricane", "flood", "tsunami", "explosion", "volcano"]
M3_ORIG_PROMPTS = [
    "overhead remote sensing view and damage assessment after an earthquake.",
    "overhead remote sensing view and damage assessment after a wildfire.",
    "overhead remote sensing view and damage assessment after a tornado.",
    "overhead remote sensing view and damage assessment after a hurricane.",
    "overhead remote sensing view and damage assessment after a flood.",
    "overhead remote sensing view and damage assessment after a tsunami.",
    "overhead remote sensing view and damage assessment after an explosion.",
    "overhead remote sensing view and damage assessment after a volcano eruption.",
]
M3_UNSEEN_PROMPTS = [
    "satellite observation of seismic ground rupture and urban building collapse",
    "satellite observation of burned forest canopies and fire scar zones",
    "satellite observation of narrow wind-swath structural destruction path",
    "satellite observation of coastal tropical cyclone wind and surge devastation",
    "satellite observation of inundated river plains and submerged settlements",
    "satellite observation of massive ocean wave coastal wash-out destruction",
    "satellite observation of blast crater and shattered industrial roofs",
    "satellite observation of volcanic ash fall, lava flow, and pyroclastic zone",
]

_DMD_DIR_MAP = {
    "fires": 0, "fire": 0,
    "floods": 1, "flood": 1,
    "natural_landscape": 2, "damaged_nature": 2, "nature": 2,
    "infrastructural": 3, "damaged_infrastructure": 3, "infrastructure": 3,
    "human_damage": 4, "damaged_human": 4, "human": 4,
    "non_damage": 5, "nondamage": 5, "non_damaged": 5,
}
_INC_MAP = {
    "earthquake": 0, "fire": 1, "accidents": 2, "flood": 3, "storm": 4,
    "other_disasters": 5, "other": 5, "landslide": 5, "hurricane": 5,
}
_M3_MAP = {
    "earthquake": 0, "wildfire": 1, "fire": 1, "bushfire": 1, "tornado": 2,
    "hurricane": 3, "flood": 4, "flooding": 4, "tsunami": 5, "explosion": 6, "volcano": 7,
}
_M3_MASK_RE = re.compile(
    r"\b(earthquakes?|wildfires?|bushfires?|fires?|accidents?|storms?|landslides?|tornad(?:o|oes)|hurricanes?|flood(?:ing|s)?|tsunamis?|explosions?|volcan(?:o|oes|ic))\b",
    re.IGNORECASE,
)
_VQA_BOILER_RE = re.compile(r"\s*(?:Select all that apply\.|Answer with Yes or No\.)", re.IGNORECASE)


def prepare_dmd(dest_dir: Path, archive_path: Path | None = None) -> Path:
    """Download/unpack official UCI #456 DMD (.tar/.tar.gz/.zip) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.rglob("*.txt")) and any(dest_dir.rglob("*.jpg")):
        return dest_dir
    arc = archive_path
    if arc is None:
        arc = aria2_download_file(DMD_UCI_URL, dest_dir / "dmd_uci_456.zip", min_bytes=500_000)
    extract_archive_safe(arc, dest_dir)
    for sub_zip in list(dest_dir.rglob("*.zip")):
        if sub_zip.resolve() != arc.resolve():
            extract_archive_safe(sub_zip, dest_dir)
    return dest_dir


def prepare_incidents_vqa(dest_dir: Path, archive_path: Path | None = None) -> Path:
    """Unpack (.tar/.zip) or fetch 0%-overlapping QCRI/DisasterVQA incidents1m subset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.rglob("disastervqa_annotations.jsonl")) and any(dest_dir.rglob("*.jpg")):
        return dest_dir
    if archive_path is not None and archive_path.is_file():
        extract_archive_safe(archive_path, dest_dir)
        ann_hits = list(dest_dir.rglob("disastervqa_annotations.jsonl"))
        if ann_hits:
            vqa_root = ann_hits[0].parent
            for drop_name in ("CrisisMMD", "MEDIC"):
                for sub_d in (vqa_root / "DisasterVQA" / drop_name, dest_dir / "DisasterVQA" / drop_name):
                    if sub_d.is_dir():
                        shutil.rmtree(sub_d, ignore_errors=True)
        return dest_dir

    ann_p = dest_dir / "disastervqa_annotations.jsonl"
    hf_lfs_batch_download("QCRI/DisasterVQA", INCIDENTS_HF_SHA, ["disastervqa_annotations.jsonl"], dest_dir)
    img_rels: list[str] = []
    seen_imgs: set[str] = set()
    for ln in ann_p.read_text(encoding="utf8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("dataset_source") != "incidents1m":
            continue
        rel = str(row["image_path"]).replace("\\", "/").lstrip("/")
        if rel not in seen_imgs:
            seen_imgs.add(rel)
            img_rels.append(rel)
    hf_lfs_batch_download("QCRI/DisasterVQA", INCIDENTS_HF_SHA, img_rels, dest_dir)
    return dest_dir


def prepare_disaster_m3(dest_dir: Path, archive_path: Path | None = None) -> Path:
    """Unpack (.tar/.zip) or fetch NeurIPS'25 DisasterM3 Optical+SAR tiles via LFS Batch CDN."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if any(dest_dir.rglob("benchmark_release.json")) and any(dest_dir.rglob("*.png")):
        return dest_dir
    if archive_path is not None and archive_path.is_file():
        extract_archive_safe(archive_path, dest_dir)
        return dest_dir

    bench_rel = "DisasterM3_Bench/benchmark_release.json"
    hf_lfs_batch_download("Kingdrone-Junjue/DisasterM3", DISASTER_M3_HF_SHA, [bench_rel], dest_dir)
    rows = json.loads((dest_dir / bench_rel).read_text(encoding="utf8"))
    img_rels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("task") != "Disaster Report":
            continue
        rel = str(row.get("post_image_path", "")).replace("\\", "/").lstrip("/")
        full_rel = f"DisasterM3_Bench/{rel}"
        if rel and full_rel not in seen:
            seen.add(full_rel)
            img_rels.append(full_rel)
    hf_lfs_batch_download("Kingdrone-Junjue/DisasterM3", DISASTER_M3_HF_SHA, img_rels, dest_dir)
    return dest_dir


def _is_valid_image(p: Path) -> bool:
    if not p.is_file() or p.stat().st_size < 64:
        return False
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(p) as im:
            return im.width > 0 and im.height > 0
    except Exception:
        return False


def _scan_dmd_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for img_p in sorted(root.rglob("*")):
        if img_p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp") or not _is_valid_image(img_p):
            continue
        cls_idx = next((_DMD_DIR_MAP[p.lower()] for p in reversed(img_p.parts[:-1]) if p.lower() in _DMD_DIR_MAP), None)
        if cls_idx is None:
            continue
        txt_p = next(
            (c for c in (img_p.with_suffix(".txt"), img_p.parent / "text" / f"{img_p.stem}.txt", img_p.parent.parent / "text" / f"{img_p.stem}.txt") if c.is_file()),
            None,
        )
        if txt_p is None:
            continue
        txt = " ".join(txt_p.read_text(encoding="utf8", errors="replace").split())
        if txt:
            records.append({"uid": f"dmd:{img_p.stem}_{cls_idx}", "group_id": f"{img_p.stem}", "label": cls_idx, "text": txt, "image": img_p})
    return records


def _scan_incidents_vqa_records(root: Path) -> list[dict[str, Any]]:
    ann_p = root / "disastervqa_annotations.jsonl"
    if not ann_p.is_file():
        ann_hits = sorted(root.rglob("disastervqa_annotations.jsonl"))
        if not ann_hits:
            raise FileNotFoundError(f"disastervqa_annotations.jsonl not found under {root}")
        ann_p = ann_hits[0]
    base_dir = ann_p.parent
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ln in ann_p.read_text(encoding="utf8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("dataset_source") != "incidents1m":
            continue
        grouped.setdefault(str(row["image_id"]), []).append(row)
    records: list[dict[str, Any]] = []
    for img_id, rows in sorted(grouped.items()):
        dtype = str(rows[0].get("disaster_type", "")).strip().lower()
        if dtype not in _INC_MAP:
            continue
        img_p = base_dir / str(rows[0]["image_path"]).replace("\\", "/").lstrip("/")
        if not _is_valid_image(img_p):
            continue
        qa_parts = []
        for r in rows:
            if r.get("crisis_info_code") == "SA-1":
                continue
            ch = r.get("choices") or {}
            raw_ans = r.get("groundtruth_answer")
            ans_items = raw_ans if isinstance(raw_ans, (list, tuple)) else ([raw_ans] if raw_ans not in (None, "") else [])
            ans_list = [str(ch.get(a, a)) if isinstance(ch, dict) else str(a) for a in ans_items]
            q_clean = _VQA_BOILER_RE.sub("", _M3_MASK_RE.sub("hazard event", str(r.get("question", "")))).strip()
            a_clean = _M3_MASK_RE.sub("hazard event", ", ".join(ans_list)).strip()
            qa_parts.append(f"{q_clean} {a_clean}".strip())
        if qa_parts:
            records.append({"uid": f"incvqa:{img_id}", "group_id": img_id, "label": _INC_MAP[dtype], "text": " | ".join(qa_parts), "image": img_p})
    return records


def _scan_disaster_m3_records(root: Path) -> list[dict[str, Any]]:
    bench_p = root / "DisasterM3_Bench" / "benchmark_release.json"
    if not bench_p.is_file():
        bench_p = root / "benchmark_release.json"
    if not bench_p.is_file():
        hits = sorted(root.rglob("benchmark_release.json"))
        if not hits:
            raise FileNotFoundError(f"benchmark_release.json not found under {root}")
        bench_p = hits[0]
    rows = json.loads(bench_p.read_text(encoding="utf8"))
    advice_by_img = {
        str(r.get("post_image_path", "")).replace("\\", "/"): str(r.get("ground_truth", ""))
        for r in rows if r.get("task") == "Disaster Restoration Advice"
    }
    img_base = bench_p.parent
    records: list[dict[str, Any]] = []
    for r in rows:
        if r.get("task") != "Disaster Report":
            continue
        raw_rep = str(r.get("ground_truth", ""))
        m_haz = re.search(r"DISASTER:\s*([^\n\r]+)", raw_rep)
        if not m_haz:
            continue
        cls_idx = _M3_MAP.get(m_haz.group(1).strip().lower())
        if cls_idx is None:
            continue
        rel = str(r.get("post_image_path", "")).replace("\\", "/").lstrip("/")
        img_p = img_base / rel
        if not _is_valid_image(img_p):
            continue
        rep_body = re.sub(r"^DISASTER:[^\n\r]*[\n\r]*", "", raw_rep, flags=re.IGNORECASE)
        adv_body = advice_by_img.get(str(r.get("post_image_path", "")).replace("\\", "/"), "")
        txt = _M3_MASK_RE.sub("hazard event", " ".join(f"{rep_body} {adv_body}".split()))
        pre_gid = str(r.get("pre_image_path", rel)).replace("\\", "/")
        records.append({"uid": f"m3:{rel}", "group_id": pre_gid, "label": cls_idx, "text": txt, "image": img_p})
    return records


def load_benchmark_contracts(
    dataset_name: str,
    seq_inputs_dir: Path,
    images_dir: Path,
    protocol: str,
    allow_test: bool = False,
    seed: int = 11,
    split_seed: int | None = None,
    class_order_seed: int | None = None,
    mnar_missing_rate: float = 1.0,
) -> dict[str, Any]:
    """Unified dispatcher loading CrisisMMD, DMD, DisasterVQA-Incidents, DisasterM3, or ScienceQA-MM8."""
    ds_key = str(dataset_name).strip().lower()
    if ds_key == "crisismmd":
        return load_crisismmd_contracts(
            seq_inputs_dir, images_dir, protocol, allow_test=allow_test,
            seed=int(seed), class_order_seed=class_order_seed, mnar_missing_rate=float(mnar_missing_rate),
        )
    kw_ext = {"split_seed": split_seed, "class_order_seed": class_order_seed, "mnar_missing_rate": float(mnar_missing_rate)}
    if ds_key == "dmd":
        return build_p1_p4_contracts(
            dataset_name="dmd", raw_records=_scan_dmd_records(images_dir), class_names=DMD_CLASSES,
            orig_prompts=DMD_ORIG_PROMPTS, unseen_prompts=DMD_UNSEEN_PROMPTS, protocol=protocol, allow_test=allow_test, seed=seed, **kw_ext,
        )
    if ds_key == "incidents_vqa":
        return build_p1_p4_contracts(
            dataset_name="incidents_vqa", raw_records=_scan_incidents_vqa_records(images_dir), class_names=INCIDENTS_CLASSES,
            orig_prompts=INCIDENTS_ORIG_PROMPTS, unseen_prompts=INCIDENTS_UNSEEN_PROMPTS, protocol=protocol, allow_test=allow_test, seed=seed, **kw_ext,
        )
    if ds_key == "disaster_m3":
        return build_p1_p4_contracts(
            dataset_name="disaster_m3", raw_records=_scan_disaster_m3_records(images_dir), class_names=M3_CLASSES,
            orig_prompts=M3_ORIG_PROMPTS, unseen_prompts=M3_UNSEEN_PROMPTS, protocol=protocol, allow_test=allow_test, seed=seed, **kw_ext,
        )
    if ds_key == "scienceqa_mm8":
        return build_p1_p4_contracts(
            dataset_name="scienceqa_mm8", raw_records=scan_scienceqa_mm8_records(images_dir), class_names=SCIENCEQA_CLASSES,
            orig_prompts=SCIENCEQA_ORIG_PROMPTS, unseen_prompts=SCIENCEQA_UNSEEN_PROMPTS, protocol=protocol, allow_test=allow_test, seed=seed, **kw_ext,
        )
    raise ValueError(f"Unsupported dataset={dataset_name!r}")
