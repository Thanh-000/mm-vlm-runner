"""Unit tests for the 3-piece ICCV supplementary suite (class-order, severity sweep, ScienceQA-MM8, summary/plots)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from mnarm.data import build_p1_p4_contracts, load_benchmark_contracts
from mnarm.runners.iccv_supplementary import write_supplementary_suite


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

    c22 = build_p1_p4_contracts(dataset_name="synthetic", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", allow_test=True, seed=22, split_seed=11, class_order_seed=22)
    c33 = build_p1_p4_contracts(dataset_name="synthetic", raw_records=recs, class_names=names, orig_prompts=p0, unseen_prompts=pu, protocol="P4", allow_test=True, seed=33, split_seed=11, class_order_seed=33)
    assert [r["uid"] for r in c22["dev"]] == [r["uid"] for r in c11["dev"]]
    assert [r["uid"] for r in c22["test"]] == [r["uid"] for r in c11["test"]]
    assert set(c22["early_classes"]) != set(c11["early_classes"])
    assert set(c33["early_classes"]) != set(c11["early_classes"])


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

    for s, b in ((11, 0.0), (22, 0.01), (33, -0.01)):
        for pr in ("P1", "P2", "P3", "P4"):
            _mock_json(supp_root / "multiseed" / "dmd" / f"seed_{s}" / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, b)
    for pr in ("P1", "P2", "P3", "P4"):
        _mock_json(supp_root / "scaling_l14" / "dmd" / f"clip_l14_same_bank/clip_l14_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_l14", pr, 0.04)
        _mock_json(supp_root / "ood_general" / "scienceqa_mm8" / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "scienceqa_mm8", "clip_b16", pr, 0.02)
    for r_tag, b in (("rho_000", 0.15), ("rho_050", 0.05), ("rho_100", 0.0)):
        for pr in ("P3", "P4"):
            _mock_json(supp_root / "mnar_sweep" / "dmd" / r_tag / f"clip_b16_same_bank/clip_b16_{pr.lower()}/ICCV_PROTOCOL_TEST_RESULTS.json", "dmd", "clip_b16", pr, b)

    master_md = write_supplementary_suite(supp_root)
    assert master_md.is_file() and "$" not in master_md.read_text(encoding="utf8")
    assert (supp_root / "ICCV_SUPP_PIECE1_MULTISEED.tex").is_file()
    assert (supp_root / "ICCV_SUPP_PIECE2_SCALING_OOD.tex").is_file()
    assert (supp_root / "ICCV_SUPP_PIECE3_MNAR_SWEEP.tex").is_file()
    assert (supp_root / "figures" / "fig_mnar_severity_sweep.png").is_file()
    assert (supp_root / "figures" / "fig_mnar_severity_sweep.pdf").is_file()
