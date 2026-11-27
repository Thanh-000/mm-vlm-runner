"""Out-of-domain general multimodal benchmark: ScienceQA-MM8 (8 academic image-text topics)."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from mnarm.core.io import sha256_bytes
from mnarm.data.archive_ops import aria2_download_file, extract_archive_safe

SCIENCEQA_HF_REPO = "derek-thomas/ScienceQA"
SCIENCEQA_HF_SHA = "f18b0a70359ebfb41f658fd564208d0355b013f4"
SCIENCEQA_PARQUETS = (
    "data/train-00000-of-00001-1028f23e353fbe3e.parquet",
    "data/validation-00000-of-00001-6c7328ff6c84284c.parquet",
    "data/test-00000-of-00001-f0e719df791966ff.parquet",
)

SCIENCEQA_CLASSES = [
    "geography",
    "biology",
    "physics",
    "engineering_practices",
    "earth_science",
    "chemistry",
    "us_history",
    "economics",
]

SCIENCEQA_ORIG_PROMPTS = [
    "an educational diagram and visual question about maps, continents, states, or geography.",
    "an educational diagram and visual question about organisms, cells, ecosystems, or biology.",
    "an educational diagram and visual question about forces, motion, energy, circuits, or physics.",
    "an educational diagram and visual question about scientific experiments, hypotheses, and engineering practices.",
    "an educational diagram and visual question about weather, rocks, fossils, astronomy, or earth science.",
    "an educational diagram and visual question about atoms, molecules, chemical reactions, or states of matter.",
    "an educational diagram and visual question about historical timelines, documents, or United States history.",
    "an educational diagram and visual question about trade, scarcity, supply and demand, or economics.",
]

SCIENCEQA_UNSEEN_PROMPTS = [
    "academic classroom illustration paired with cartography or regional spatial reasoning",
    "life science figure paired with anatomy, genetics, or food-web question",
    "mechanics, optics, thermodynamics, or electricity instructional figure",
    "controlled variable design and laboratory inquiry reasoning schematic",
    "geological strata, meteorology, or planetary science instructional plate",
    "molecular structure, periodic property, or mixture transformation graphic",
    "primary historical map, civic artifact, or American chronological question",
    "marketplace exchange, budgeting, or resource allocation scenario figure",
]

_SQA_TOPIC_MAP = {
    "geography": 0,
    "biology": 1,
    "physics": 2,
    "science-and-engineering-practices": 3,
    "earth-science": 4,
    "chemistry": 5,
    "us-history": 6,
    "economics": 7,
}

_SQA_MASK_RE = re.compile(
    r"\b(geograph(?:y|ic|ical)|biolog(?:y|ical)|physic(?:s|al)|science and engineering practices|"
    r"engineering practices|earth[-\s]science|chemist(?:ry|s)|(?:us|u\.s\.|united states)\s+history|"
    r"econom(?:ics|ic))\b",
    re.IGNORECASE,
)


def _format_sqa_text(row: dict[str, Any]) -> str:
    q = str(row.get("question") or "").strip()
    hint = str(row.get("hint") or "").strip()
    choices = [str(c).strip() for c in (row.get("choices") or []) if str(c).strip()]
    parts = [p for p in (q, f"Choices: {'; '.join(choices)}" if choices else "", hint) if p]
    clean = _SQA_MASK_RE.sub("academic concept", " ".join(" ".join(parts).split()))
    return clean[:512]


def prepare_scienceqa_mm8(dest_dir: Path, archive_path: Path | None = None) -> Path:
    """Download official ScienceQA Parquets via aria2c and materialize all 8-topic image-paired records."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    jsonl_p = dest_dir / "scienceqa_mm8.jsonl"
    img_dir = dest_dir / "images"
    if jsonl_p.is_file() and any(img_dir.glob("*.png")):
        return dest_dir

    if archive_path is not None and archive_path.is_file():
        extract_archive_safe(archive_path, dest_dir)
        if any(dest_dir.rglob("scienceqa_mm8.jsonl")) and any(dest_dir.rglob("*.png")):
            return dest_dir

    import pyarrow.parquet as pq

    img_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[str] = []
    idx_global = 0
    for rel_pq in SCIENCEQA_PARQUETS:
        pq_local = dest_dir / Path(rel_pq).name
        if not pq_local.is_file():
            url = f"https://huggingface.co/datasets/{SCIENCEQA_HF_REPO}/resolve/{SCIENCEQA_HF_SHA}/{rel_pq}"
            aria2_download_file(url, pq_local, min_bytes=50_000_000, max_conn=16)
        pf = pq.ParquetFile(pq_local)
        for batch in pf.iter_batches(batch_size=512, columns=["image", "question", "choices", "hint", "topic"]):
            for row in batch.to_pylist():
                topic_key = str(row.get("topic") or "").strip().lower()
                cls_idx = _SQA_TOPIC_MAP.get(topic_key)
                img_obj = row.get("image")
                img_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None
                if cls_idx is None or not img_bytes:
                    continue
                img_sha = sha256_bytes(img_bytes)[:20]
                img_p = img_dir / f"{img_sha}.png"
                if not img_p.is_file():
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as im:
                            im.convert("RGB").save(img_p, format="PNG")
                    except OSError:
                        continue
                txt = _format_sqa_text(row)
                if not txt:
                    continue
                idx_global += 1
                rec = {
                    "uid": f"sqa:{idx_global:06d}_{img_sha}",
                    "group_id": img_sha,
                    "label": int(cls_idx),
                    "text": txt,
                    "image_rel": f"images/{img_sha}.png",
                }
                rows_out.append(json.dumps(rec, ensure_ascii=False))
        pq_local.unlink(missing_ok=True)

    if not rows_out:
        raise RuntimeError(f"Zero valid ScienceQA-MM8 image-text records extracted into {dest_dir}")
    jsonl_p.write_text("\n".join(rows_out) + "\n", encoding="utf8")
    return dest_dir


def scan_scienceqa_mm8_records(root: Path) -> list[dict[str, Any]]:
    """Scan local materialized scienceqa_mm8.jsonl and images/ directory."""
    jsonl_p = root / "scienceqa_mm8.jsonl"
    if not jsonl_p.is_file():
        hits = sorted(root.rglob("scienceqa_mm8.jsonl"))
        if not hits:
            raise FileNotFoundError(f"scienceqa_mm8.jsonl not found under {root}")
        jsonl_p = hits[0]
    base = jsonl_p.parent
    records: list[dict[str, Any]] = []
    for ln in jsonl_p.read_text(encoding="utf8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        img_p = base / str(row["image_rel"]).replace("\\", "/").lstrip("/")
        if img_p.is_file() and img_p.stat().st_size > 0:
            records.append(
                {
                    "uid": str(row["uid"]),
                    "group_id": str(row["group_id"]),
                    "label": int(row["label"]),
                    "text": str(row["text"]),
                    "image": img_p,
                }
            )
    return records
