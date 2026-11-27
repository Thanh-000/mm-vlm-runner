"""Unit tests for the 3-piece ICCV supplementary suite (class-order, severity sweep, ScienceQA-MM8, summary/plots)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from mnarm.data import build_p1_p4_contracts, load_benchmark_contracts
from mnarm.runners.iccv_supplementary import merge_supplementary_zips, write_supplementary_suite


def _make_dummy_records(root: Path, n_classes: int = 8, per_cls: int = 8) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    recs = []
    for c in range(n_classes):
        for i in range(per_cls):
            ip = root / f"img_{c}_{i}.png"
            Image.new("RGB", (16, 16), color=(c * 25 % 255, i * 30 % 255, 80)).save(ip)
            recs.append({"uid": f"u_{c}_{i}", "group_id": f"g_{c}_{i}", "label": c, "text": f"sample class {c} item {i}", "image": ip})
    return recs


def test_class_order_permutation_keeps_identical_dev_test(tmp_path: Path) -> None:
    recs = _make_dummy_records(tmp_path / "imgs", n_classes=8, per_cls=8)
    names = [f"c{i}" for i in range(8)]
    p0 = [f"prompt {i}" for i in range(8)]
    pu = [f"unseen {i}" for i in range(8)]
    c11 = build_p1_p4_contracts(dataset_name="synthetic", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", allow_test=True, seed=11, split_seed=11, class_order_seed=11)
    c11_def = build_p1_p4_contracts(dataset_name="synthetic", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", allow_test=True, seed=11)
    assert c11["early_classes"] == c11_def["early_classes"] == [0, 1, 2, 3]
    assert [r["uid"] for r in c11["test"]] == [r["uid"] for r in c11_def["test"]]

    for extra_s in (22, 33, 44, 55):
        c_s = build_p1_p4_contracts(dataset_name="synthetic", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", allow_test=True, seed=extra_s, split_seed=11, class_order_seed=extra_s)
        assert [r["uid"] for r in c_s["dev"]] == [r["uid"] for r in c11["dev"]]
        assert [r["uid"] for r in c_s["test"]] == [r["uid"] for r in c11["test"]]
        assert set(c_s["early_classes"]) != set(c11["early_classes"])


def test_mnar_missing_rate_sweep(tmp_path: Path) -> None:
    recs = _make_dummy_records(tmp_path / "imgs", n_classes=6, per_cls=10)
    names = [f"c{i}" for i in range(6)]
    p0 = [f"prompt {i}" for i in range(6)]
    pu = [f"unseen {i}" for i in range(6)]
    c0 = build_p1_p4_contracts(dataset_name="dmd", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", seed=11, mnar_missing_rate=0.0)
    early_fit0 = [r["mask"] for s in c0["stages"] for r in s["fit"] if r["label"] in c0["early_classes"]]
    assert all(m == ["text", "image"] for m in early_fit0)

    c50 = build_p1_p4_contracts(dataset_name="dmd", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P3", seed=11, mnar_missing_rate=0.5)
    early_fit50 = [r["mask"] for s in c50["stages"] for r in s["fit"] if r["label"] in c50["early_classes"]]
    n_miss = sum(1 for m in early_fit50 if m == ["text"])
    assert 0 < n_miss < len(early_fit50)

    c100 = build_p1_p4_contracts(dataset_name="dmd", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", seed=11, mnar_missing_rate=1.0)
    early_fit100 = [r["mask"] for s in c100["stages"] for r in s["fit"] if r["label"] in c100["early_classes"]]
    assert all(m == ["image"] for m in early_fit100)


def test_scienceqa_mm8_loader_and_supplementary_aggregator(tmp_path: Path) -> None:
    sqa_dir = tmp_path / "scienceqa_mm8"
    (sqa_dir / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for c in range(8):
        for i in range(6):
            img_rel = f"images/s_{c}_{i}.png"
            Image.new("RGB", (16, 16), color=(c * 31 % 255, i * 41 % 255, 120)).save(sqa_dir / img_rel)
            lines.append(json.dumps({"uid": f"sqa:{c}_{i}", "group_id": f"sg_{c}_{i}", "label": c, "text": f"question diagram {c} {i}", "image_rel": img_rel}))
    (sqa_dir / "scienceqa_mm8.jsonl").write_text("\n".join(lines) + "\n", encoding="utf8")

    ds = load_benchmark_contracts("scienceqa_mm8", sqa_dir, sqa_dir, "P4", allow_test=True, seed=11)
    assert ds["num_classes"] == 8 and ds["num_stages"] == 4
    assert all(len(s["new_classes"]) == 2 for s in ds["stages"])

    supp_root = tmp_path / "supp_out"
    def _mock_json(path: Path, ds_name: str, bb: str, pr: str, boost: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arms = {}
        for arm in ("opcm", "knots", "tsvm", "dc_lora", "iso_c", "magmax", "seq_ft", "seq_magmax", "seq_lmnar", "seq_replay256", "mg_clip", "pal_rls", "joint_oracle"):
            regs = {"raw": {"paired": {"macro_f1": 0.45 + boost, "early_recall": 0.30}, "summary": {"view3_mean_f1": 0.42}}}
            if arm != "pal_rls":
                regs.update({
                    "ncm": {"paired": {"macro_f1": 0.40 + boost, "early_recall": 0.15}, "summary": {"view3_mean_f1": 0.38}},
                    "cal_bias": {"paired": {"macro_f1": 0.50 + boost, "early_recall": 0.35}, "summary": {"view3_mean_f1": 0.45}},
                    "zero_mnar": {"paired": {"macro_f1": 0.72 + boost, "early_recall": 0.65}, "summary": {"view3_mean_f1": 0.68}},
                    "gr_clip": {"paired": {"macro_f1": 0.48 + boost, "early_recall": 0.31}, "summary": {"view3_mean_f1": 0.44}},
                })
            else:
                regs["cal_bias"] = {"paired": {"macro_f1": 0.55 + boost, "early_recall": 0.32}, "summary": {"view3_mean_f1": 0.48}}
            arms[arm] = {"regimes": regs}
        path.write_text(json.dumps({"config": {"dataset": ds_name, "backbone": bb, "protocol": pr}, "arms": arms}), encoding="utf8")

    for s, b in ((11, 0.0), (22, 0.01), (33, -0.01), (44, 0.005), (55, -0.005)):
        for pr in ("P1", "P2", "P3", "P4"):
            _mock_json(supp_root / "multiseed" / "dmd" / f"seed_{s}" / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, b)
    for pr in ("P1", "P2", "P3", "P4"):
        _mock_json(supp_root / "scaling_b32" / "dmd" / f"clip_b32_same_bank/clip_b32_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b32", pr, -0.02)
        _mock_json(supp_root / "scaling_l14" / "dmd" / f"clip_l14_same_bank/clip_l14_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_l14", pr, 0.04)
        _mock_json(supp_root / "ood_general" / "scienceqa_mm8" / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "scienceqa_mm8", "clip_b16", pr, 0.02)
    for r_tag, b in (("rho_000", 0.15), ("rho_050", 0.05), ("rho_100", 0.0)):
        for pr in ("P3", "P4"):
            _mock_json(supp_root / "mnar_sweep" / "dmd" / r_tag / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, b)

    master_md = write_supplementary_suite(supp_root)
    assert master_md.is_file() and "$" not in master_md.read_text(encoding="utf8")
    assert "B/32" in master_md.read_text(encoding="utf8")
    assert "scaling_b32" in json.loads((supp_root / "ICCV_SUPPLEMENTARY_MASTER_RESULTS.json").read_text(encoding="utf8"))["piece2_scaling_ood"]
    assert (supp_root / "ICCV_SUPP_PIECE1_MULTISEED.tex").is_file()
    assert (supp_root / "ICCV_SUPP_PIECE2_SCALING_OOD.tex").is_file()
    assert (supp_root / "ICCV_SUPP_PIECE3_MNAR_SWEEP.tex").is_file()
    assert (supp_root / "figures" / "fig_mnar_severity_sweep.png").is_file()
    assert (supp_root / "figures" / "fig_mnar_severity_sweep.pdf").is_file()


def test_crisismmd_multiseed_class_order_keeps_splits_fixed(tmp_path: Path) -> None:
    import csv, io, zipfile
    from mnarm.data.crisismmd import CRISISMMD_CLASSES, load_crisismmd_contracts, uid_hash

    seq_dir, img_dir = tmp_path / "seq", tmp_path / "imgs"
    seq_dir.mkdir(parents=True); (img_dir / "data_image").mkdir(parents=True)
    rows_sp = {"train": [], "dev": [], "test": []}
    plans_p4 = [{"stage": s + 1, "buffer": [], "calibration_latest_views": []} for s in range(4)]
    pc_stages = []
    for sp, per_cls in (("train", 6), ("dev", 2), ("test", 2)):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["tweet_id", "image_id", "tweet_text"], delimiter="\t")
        writer.writeheader()
        for c in range(8):
            for i in range(per_cls):
                tid, iid = f"{sp}_{c}_{i}", f"img_{sp}_{c}_{i}"
                txt = f"disaster tweet class {CRISISMMD_CLASSES[c]} sample {sp} {i}"
                sid = uid_hash(tid, iid)
                rel = f"data_image/f_{sid[:12]}.jpg"
                Image.new("RGB", (16, 16), color=(c * 25 % 255, i * 40 % 255, 60)).save(img_dir / rel)
                writer.writerow({"tweet_id": tid, "image_id": iid, "tweet_text": txt})
                from mnarm.core.io import sha256_bytes
                rec = {"sample_id": sid, "label": c, "image": rel, "text_sha256": sha256_bytes(txt.encode("utf8"))}
                rows_sp[sp].append(rec)
                if sp == "train" and len(pc_stages) <= c // 2:
                    pc_stages.append({"train_ids": [], "modalities": ["image"] if c // 2 < 2 else ["text", "image"]})
                if sp == "train":
                    pc_stages[c // 2]["train_ids"].append(sid)
                    plans_p4[c // 2]["calibration_latest_views"].append({"uid": sid, "label": c, "mask": ["image"] if c // 2 < 2 else ["text", "image"]})
        with zipfile.ZipFile(seq_dir / "splits.zip", "a") as zf:
            zf.writestr(f"task_humanitarian_text_img_{sp}.tsv", buf.getvalue())
    for st_i in range(4):
        plans_p4[st_i]["seen_classes"] = list(range((st_i + 1) * 2))
    (seq_dir / "manifest.json").write_text(json.dumps({"rows": rows_sp}), encoding="utf8")
    (seq_dir / "PLANS.json").write_text(json.dumps({"P4": plans_p4}), encoding="utf8")
    (seq_dir / "partition_contract.json").write_text(json.dumps({"protocols": {"P4": pc_stages}}), encoding="utf8")

    c11 = load_crisismmd_contracts(seq_dir, img_dir, "P4", allow_test=True, seed=11)
    c22 = load_crisismmd_contracts(seq_dir, img_dir, "P4", allow_test=True, seed=22, class_order_seed=22)
    assert [r["uid"] for r in c22["dev"]] == [r["uid"] for r in c11["dev"]]
    assert [r["uid"] for r in c22["test"]] == [r["uid"] for r in c11["test"]]
    assert set(c22["early_classes"]) != set(c11["early_classes"])
    assert len(c22["stages"]) == 4


def test_merge_supplementary_zips_combines_part_a_and_part_b(tmp_path: Path) -> None:
    import zipfile

    def _mk_json(root: Path, rel: str, ds: str, bb: str, pr: str, val: float) -> None:
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        arms = {
            arm: {
                "regimes": {
                    "raw": {"paired": {"macro_f1": val, "early_recall": 0.3}, "summary": {"view3_mean_f1": val}},
                    **({} if arm == "pal_rls" else {
                        "ncm": {"paired": {"macro_f1": val - 0.05, "early_recall": 0.2}, "summary": {"view3_mean_f1": val - 0.05}},
                        "cal_bias": {"paired": {"macro_f1": val + 0.02, "early_recall": 0.35}, "summary": {"view3_mean_f1": val}},
                        "zero_mnar": {"paired": {"macro_f1": val + 0.25, "early_recall": 0.7}, "summary": {"view3_mean_f1": val + 0.2}},
                    }),
                }
            }
            for arm in ("opcm", "knots", "tsvm", "dc_lora", "iso_c", "magmax", "seq_ft", "seq_magmax", "seq_lmnar", "seq_replay256", "mg_clip", "pal_rls", "joint_oracle")
        }
        fp.write_text(json.dumps({"config": {"dataset": ds, "backbone": bb, "protocol": pr}, "arms": arms}), encoding="utf8")

    part_a_dir, part_b_dir = tmp_path / "pa_raw", tmp_path / "pb_raw"
    for s in (11, 22):
        for pr in ("P1", "P3", "P4"):
            _mk_json(part_a_dir, f"multiseed/dmd/seed_{s}/clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, 0.50)
    for rt, v in (("rho_000", 0.65), ("rho_100", 0.50)):
        _mk_json(part_a_dir, f"mnar_sweep/dmd/{rt}/clip_b16_same_bank/clip_b16_p4/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", "P4", v)
    for s in (33, 44, 55):
        for pr in ("P1", "P3", "P4"):
            _mk_json(part_b_dir, f"multiseed/dmd/seed_{s}/clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, 0.52)

    pa_zip, pb_zip = tmp_path / "part_a.zip", tmp_path / "part_b.zip"
    for d, zp in ((part_a_dir, pa_zip), (part_b_dir, pb_zip)):
        with zipfile.ZipFile(zp, "w") as zf:
            for fp in sorted(d.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(d).as_posix())

    merged_dir = tmp_path / "merged_out"
    master_md = merge_supplementary_zips([pa_zip, pb_zip], merged_dir)
    res_json = json.loads((merged_dir / "ICCV_SUPPLEMENTARY_MASTER_RESULTS.json").read_text(encoding="utf8"))
    assert res_json["piece1_multiseed"]["seeds"] == [11, 22, 33, 44, 55]
    assert (merged_dir / "figures" / "fig_mnar_severity_sweep.pdf").is_file()
    assert "$" not in master_md.read_text(encoding="utf8")
