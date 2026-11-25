"""Deterministic zero-leakage P1-P4 MNAR protocol builder across multimodal disaster datasets."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mnarm.core.io import sha256_bytes
from mnarm.data.crisismmd import assert_zero_split_overlap, uid_hash


def _cluster_by_identity(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    n = len(records)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    seen_gid: dict[str, int] = {}
    seen_img: dict[str, int] = {}
    seen_txt: dict[str, int] = {}
    for idx, rec in enumerate(records):
        gid = str(rec["group_id"])
        img_k = str(Path(rec["image"]))
        txt_k = str(rec["text_sha256"])
        for mp, key in ((seen_gid, gid), (seen_img, img_k), (seen_txt, txt_k)):
            if key in mp:
                _union(idx, mp[key])
            else:
                mp[key] = idx

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        groups[_find(idx)].append(rec)
    return [sorted(g, key=lambda r: str(r["uid"])) for _, g in sorted(groups.items())]


def _group_stratified_split(
    records: list[dict[str, Any]], num_classes: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    clusters = _cluster_by_identity(records)
    by_cls: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    for cl in clusters:
        maj = Counter(int(r["label"]) for r in cl).most_common(1)[0][0]
        by_cls[int(maj)].append(cl)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for cls in range(int(num_classes)):
        cls_clusters = list(by_cls.get(cls, []))
        rng.shuffle(cls_clusters)
        k = len(cls_clusters)
        if k == 0:
            raise ValueError(f"Class {cls} has 0 clusters; cannot build valid P1-P4 split.")
        if k < 3:
            split_rows["train"].extend(r for cl in cls_clusters for r in cl)
            continue
        n_dev = max(1, round(0.15 * k))
        n_test = max(1, round(0.15 * k))
        if n_dev + n_test >= k:
            n_dev, n_test = 1, 1
        n_tr = k - n_dev - n_test
        for cl in cls_clusters[:n_tr]:
            split_rows["train"].extend(cl)
        for cl in cls_clusters[n_tr : n_tr + n_dev]:
            split_rows["dev"].extend(cl)
        for cl in cls_clusters[n_tr + n_dev :]:
            split_rows["test"].extend(cl)

    return split_rows["train"], split_rows["dev"], split_rows["test"]


def _reservoir_buffer(records: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= cap:
        return [dict(r) for r in records]
    rng = random.Random(int(seed))
    by_c: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_c[int(r["label"])].append(r)
    for vals in by_c.values():
        rng.shuffle(vals)
    out: list[dict[str, Any]] = []
    pos = 0
    while len(out) < cap:
        progressed = False
        for c in sorted(by_c):
            if pos < len(by_c[c]):
                out.append(dict(by_c[c][pos]))
                progressed = True
                if len(out) >= cap:
                    break
        if not progressed:
            break
        pos += 1
    return out


def build_p1_p4_contracts(
    *,
    dataset_name: str,
    raw_records: list[dict[str, Any]],
    class_names: list[str],
    orig_prompts: list[str],
    unseen_prompts: list[str],
    protocol: str,
    allow_test: bool = False,
    seed: int = 11,
) -> dict[str, Any]:
    """Construct zero-leakage P1-P4 continual stages & MNAR modality contracts for any disaster dataset."""
    if protocol not in ("P1", "P2", "P3", "P4"):
        raise ValueError(f"Unsupported protocol: {protocol}")
    k_cls = len(class_names)
    if k_cls < 4:
        raise ValueError(f"Expected >= 4 classes, got {k_cls}")

    norm_recs: list[dict[str, Any]] = []
    for r in raw_records:
        txt = str(r["text"]).strip()
        uid = str(r.get("uid") or uid_hash(dataset_name, str(r["image"]), txt))
        norm_recs.append(
            {
                "uid": uid,
                "sample_id": uid,
                "group_id": str(r.get("group_id") or uid),
                "label": int(r["label"]),
                "text": txt,
                "text_sha256": sha256_bytes(txt.encode("utf8")),
                "image": Path(r["image"]),
            }
        )

    tr_rows, dev_rows_raw, test_rows_raw = _group_stratified_split(norm_recs, k_cls, seed=seed)
    overlap_audit = assert_zero_split_overlap({"train": tr_rows, "dev": dev_rows_raw, "test": test_rows_raw})

    num_stages = max(3, k_cls // 2)
    stage_cls: list[list[int]] = [[] for _ in range(num_stages)]
    for c in range(k_cls):
        stage_cls[min(num_stages - 1, c // 2 if k_cls // 2 == num_stages else (c * num_stages) // k_cls)].append(c)
    n_early_st = 2 if num_stages >= 4 else 1
    early_classes = [c for st_i in range(n_early_st) for c in stage_cls[st_i]]
    late_classes = [c for st_i in range(n_early_st, num_stages) for c in stage_cls[st_i]]

    rng = random.Random(int(seed) + 71)
    by_cls_tr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in tr_rows:
        by_cls_tr[int(r["label"])].append(r)

    fit_by_cls: dict[int, list[dict[str, Any]]] = {}
    cal_by_cls: dict[int, list[dict[str, Any]]] = {}
    for c in range(k_cls):
        grp = sorted(by_cls_tr[c], key=lambda x: x["uid"])
        rng.shuffle(grp)
        n_cal = max(1, round(0.12 * len(grp))) if len(grp) >= 5 else (1 if len(grp) >= 3 else 0)
        cal_by_cls[c] = grp[:n_cal]
        fit_by_cls[c] = grp[n_cal:]

    stage_plans: list[dict[str, Any]] = []
    seen_so_far: list[int] = []
    all_fit_hist: list[dict[str, Any]] = []
    cal_hist: list[dict[str, Any]] = []
    joint_so_far: dict[str, dict[str, Any]] = {}
    buf_before: list[dict[str, Any]] = []

    for s_idx, cls_list in enumerate(stage_cls):
        st_num = s_idx + 1
        seen_so_far.extend(cls_list)
        is_early = s_idx < n_early_st
        fit_t: list[dict[str, Any]] = []
        cal_t_new: list[dict[str, Any]] = []

        for c in cls_list:
            c_fit = fit_by_cls[c]
            c_cal = cal_by_cls[c]
            for pos, r in enumerate(c_fit):
                if protocol == "P3":
                    msk = ["text"] if is_early else ["text", "image"]
                elif protocol == "P4":
                    msk = ["image"] if is_early else ["text", "image"]
                elif protocol == "P1":
                    msk = ["text"] if (s_idx == 0 and pos % 2 == 0) else ["text", "image"]
                else:
                    msk = ["image"] if (s_idx == 0 and pos % 2 == 0) else ["text", "image"]
                fit_t.append({"uid": r["uid"], "label": int(r["label"]), "mask": msk})

            for pos, r in enumerate(c_cal):
                if protocol == "P3":
                    cmsk = ["text"] if is_early else ["text", "image"]
                elif protocol == "P4":
                    cmsk = ["image"] if is_early else ["text", "image"]
                else:
                    cmsk = ["text", "image"] if st_num >= 2 else (["text"] if (protocol == "P1" and pos % 2 == 0) else (["image"] if (protocol == "P2" and pos % 2 == 0) else ["text", "image"]))
                cal_t_new.append({"uid": r["uid"], "label": int(r["label"]), "mask": cmsk})

        if protocol in ("P1", "P2") and st_num >= 2:
            cal_hist = [{"uid": r["uid"], "label": r["label"], "mask": ["text", "image"]} for r in cal_hist]
        cal_hist.extend(cal_t_new)
        all_fit_hist.extend(fit_t)
        buf_after = _reservoir_buffer(all_fit_hist, 256, seed + st_num * 17)
        joint_before = [dict(v) for v in joint_so_far.values()]
        for rec in fit_t:
            joint_so_far[rec["uid"]] = dict(rec)
        if protocol in ("P1", "P2") and st_num >= 2:
            for rec in joint_so_far.values():
                rec["mask"] = ["text", "image"]

        stage_plans.append(
            {
                "stage": st_num,
                "new_classes": sorted(cls_list),
                "seen_classes": sorted(seen_so_far),
                "fit": fit_t,
                "buffer_before": [dict(r) for r in buf_before],
                "buffer_after": buf_after,
                "joint_before": joint_before,
                "calibration_latest_views": [dict(r) for r in cal_hist],
            }
        )
        buf_before = buf_after

    active_rows = tr_rows + dev_rows_raw + (test_rows_raw if allow_test else [])
    lookup = {r["uid"]: {"uid": r["uid"], "label": int(r["label"]), "text": r["text"], "image": r["image"]} for r in active_rows}
    dev_out = [{"uid": r["uid"], "label": int(r["label"]), "mask": ["text", "image"]} for r in dev_rows_raw]
    test_out = [{"uid": r["uid"], "label": int(r["label"]), "mask": ["text", "image"]} for r in test_rows_raw] if allow_test else None
    manifest_sig = sha256_bytes(json.dumps([(r["uid"], r["label"], r["text_sha256"]) for r in sorted(norm_recs, key=lambda x: x["uid"])]).encode("utf8"))

    return {
        "name": dataset_name,
        "protocol": protocol,
        "allow_test": bool(allow_test),
        "num_classes": k_cls,
        "num_stages": num_stages,
        "class_names": list(class_names),
        "orig_prompts": list(orig_prompts),
        "unseen_prompts": list(unseen_prompts),
        "early_classes": early_classes,
        "late_classes": late_classes,
        "stages": stage_plans,
        "dev": dev_out,
        "test": test_out,
        "lookup": lookup,
        "hashes": {"dataset_manifest_sha256": manifest_sig, **overlap_audit},
    }
