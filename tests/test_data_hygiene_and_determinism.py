"""Tests for split-overlap guards, test isolation (allow_test=False), determinism, and tower digests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from mnarm.core import seed_everything
from mnarm.data import assert_zero_split_overlap, load_crisismmd_contracts, uid_hash
from mnarm.models import tower_digest


def test_assert_zero_split_overlap_catches_all_leak_types() -> None:
    base = {
        "train": [{"sample_id": "a", "image": "img_a.jpg", "text_sha256": "ta"}],
        "dev": [{"sample_id": "b", "image": "img_b.jpg", "text_sha256": "tb"}],
        "test": [{"sample_id": "c", "image": "img_c.jpg", "text_sha256": "tc"}],
    }
    audit = assert_zero_split_overlap(base)
    assert all(v == 0 for v in audit.values())

    for field, leak_val in (("sample_id", "a"), ("image", "img_a.jpg"), ("text_sha256", "ta")):
        bad = {k: [dict(r) for r in v] for k, v in base.items()}
        bad["test"][0][field] = leak_val
        with pytest.raises(ValueError, match="Data leakage detected"):
            assert_zero_split_overlap(bad)


def test_real_crisismmd_manifest_zero_overlap_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    a2z = (
        repo_root.parent
        / "research_baseline_2026_11_05"
        / "related_baselines_2026"
        / "benchmark_v1"
        / "COLAB"
        / "JCMIL_A2Z_INPUTS.zip"
    )
    if not a2z.is_file():
        pytest.skip("JCMIL_A2Z_INPUTS.zip not present in isolated runtime")
    with zipfile.ZipFile(a2z) as zf:
        bm = json.loads(zf.read("BUNDLE_MANIFEST.json"))
        mf = json.loads(zf.read("blobs/" + bm["files"]["dense_bundle/kit/inputs/manifest.json"]["sha256"]))
    audit = assert_zero_split_overlap(mf["rows"])
    assert all(v == 0 for v in audit.values())


def test_load_crisismmd_contracts_never_requires_test_when_disallowed(tmp_path: Path) -> None:
    seq_dir = tmp_path / "seq"
    img_dir = tmp_path / "imgs"
    seq_dir.mkdir()
    img_dir.mkdir()

    # Build 2 synthetic train/dev items (1 early label 0, 1 late label 2) + missing test image
    from mnarm.core import sha256_bytes

    def _mk_row(idx: int, label: int, prefix: str):
        tw, im_id, txt = f"tw{idx}", f"im{idx}", f"sample text {prefix} {idx}"
        uid = uid_hash(tw, im_id)
        rel = f"data_image/{prefix}_{idx}.jpg"
        return {"sample_id": uid, "label": label, "image": rel, "text_sha256": sha256_bytes(txt.encode("utf8")), "tweet_id": tw, "image_id": im_id, "tweet_text": txt}

    r_tr0 = _mk_row(1, 0, "tr")
    r_tr2 = _mk_row(2, 2, "tr")
    r_dv0 = _mk_row(3, 0, "dv")
    r_te0 = _mk_row(4, 0, "te")  # Deliberately do NOT write test image to disk!

    for r in (r_tr0, r_tr2, r_dv0):
        p = img_dir / r["image"]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(10, 20, 30)).save(p)

    manifest = {"rows": {"train": [r_tr0, r_tr2], "dev": [r_dv0], "test": [r_te0]}}
    (seq_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    plans = {
        "P4": [
            {"stage": s, "seen_classes": [0] if s <= 2 else [0, 2], "buffer": [], "calibration_latest_views": []}
            for s in range(1, 5)
        ]
    }
    (seq_dir / "PLANS.json").write_text(json.dumps(plans), encoding="utf8")
    pc = {
        "protocols": {
            "P4": [
                {"stage": 1, "modalities": ["image"], "train_ids": [r_tr0["sample_id"]]},
                {"stage": 2, "modalities": ["image"], "train_ids": []},
                {"stage": 3, "modalities": ["text", "image"], "train_ids": [r_tr2["sample_id"]]},
                {"stage": 4, "modalities": ["text", "image"], "train_ids": []},
            ]
        }
    }
    (seq_dir / "partition_contract.json").write_text(json.dumps(pc), encoding="utf8")

    with zipfile.ZipFile(seq_dir / "splits.zip", "w") as zf:
        for sp, rlist in (("train", [r_tr0, r_tr2]), ("dev", [r_dv0])):
            lines = ["tweet_id\timage_id\ttweet_text"] + [f"{r['tweet_id']}\t{r['image_id']}\t{r['tweet_text']}" for r in rlist]
            zf.writestr(f"task_humanitarian_text_img_{sp}.tsv", "\n".join(lines))

    loaded = load_crisismmd_contracts(seq_dir, img_dir, "P4", allow_test=False)
    assert loaded["test"] is None
    assert loaded["allow_test"] is False
    assert len(loaded["dev"]) == 1


def test_seed_everything_and_collision_free_tower_digest() -> None:
    seed_everything(42, deterministic=True)
    a = torch.randn(4, 8)
    seed_everything(42, deterministic=True)
    b = torch.randn(4, 8)
    assert torch.equal(a, b)

    # Regression check against old L1-sum collision bug: +w vs -w share identical sum(abs)
    w_pos = {"text_model.layer0.q_proj.weight": torch.tensor([[1.0, -2.0], [3.0, -4.0]])}
    w_neg = {"text_model.layer0.q_proj.weight": torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])}
    w_zero = {"text_model.layer0.q_proj.weight": torch.zeros((2, 2))}
    assert tower_digest("text", "delta", w_zero) == "base_zero"
    assert tower_digest("text", "delta", w_pos) != tower_digest("text", "delta", w_neg)


def test_grad_accum_optimizer_steps_and_report_render(tmp_path: Path) -> None:
    from mnarm.runners.pilot_f1 import _render_report
    from mnarm.runners.stage_ops import train_lora_stage

    class _DummyStore:
        def make_batch_specs(self, records, batch_size, shuffle=False, seed=11):
            return [(("image",), records[i : i + batch_size]) for i in range(0, len(records), batch_size)]

        def collate_chunk(self, views, chunk):
            return {
                "pixel_values": torch.stack([torch.full((8,), float(r["uid"][-1])) for r in chunk]),
                "label": torch.tensor([int(r["label"]) for r in chunk], dtype=torch.long),
                "sample_id": [r["uid"] for r in chunk],
            }

    class _DummyVLM:
        device = "cpu"
        logit_scale = 10.0

        def __init__(self):
            self.w_a = torch.nn.Parameter(torch.eye(8))
            self.w_b = torch.nn.Parameter(torch.zeros(8, 8))
            self.clip = torch.nn.Linear(8, 8)

        def restore_base_weights(self): pass
        def reset_lora(self):
            torch.nn.init.eye_(self.w_a); torch.nn.init.zeros_(self.w_b)
        def load_lora_state(self, s): pass
        def lora_parameters(self): return [self.w_a, self.w_b]
        def encode_image(self, inp): return inp["pixel_values"] @ self.w_a + inp["pixel_values"] @ self.w_b
        def lora_state(self): return {"w_a": self.w_a.detach().clone(), "w_b": self.w_b.detach().clone()}

    recs = [{"uid": f"u{i}", "label": i % 2, "mask": ["image"]} for i in range(4)]
    protos = torch.eye(8)[:2]
    vlm1 = _DummyVLM()
    _, m1 = train_lora_stage(
        vlm=vlm1, init_lora=None, records=recs, protos_full=protos, seen_classes=[0, 1],
        objective="l_mnar", stage_idx=1, epochs=2, effective_batch_size=4, micro_batch_size=4,
        lr=1e-3, weight_decay=0.0, seed=11, use_amp=False, tensor_store=_DummyStore(),
    )
    vlm2 = _DummyVLM()
    _, m2 = train_lora_stage(
        vlm=vlm2, init_lora=None, records=recs, protos_full=protos, seen_classes=[0, 1],
        objective="l_mnar", stage_idx=1, epochs=2, effective_batch_size=4, micro_batch_size=2,
        lr=1e-3, weight_decay=0.0, seed=11, use_amp=False, tensor_store=_DummyStore(),
    )
    assert m1["optimizer_steps"] == m2["optimizer_steps"] == 2
    assert m1["accum_steps"] == 1 and m2["accum_steps"] == 2

    # Check report rendering emits clean Markdown with 0 '$' signs
    sample_m = {
        "macro_f1": 0.5, "macro_f1_7cls": 0.52, "accuracy": 0.6,
        "early_recall": 0.5, "late_recall": 0.5, "worst_class_f1": 0.3,
        "ece": 0.05, "nll": 0.7, "brier": 0.3,
    }
    sample_res = {
        "config": {"backbone": "clip_b16", "protocol": "P4", "gates": {"max_theta_deg": 15.0}},
        "verdict": {"status": "PASS", "gate1_pass": True, "gate2_pass": True, "max_lmnar_theta_t_deg": 5.0, "best_arm": "l_mnar_joint", "best_variant": "bayes_hybrid_bridge_0lab", "best_paired_f1": 0.5, "best_view3_mean_f1": 0.5},
        "arms": {"l_mnar_joint": {"cone_drift": {"text": {"theta_deg": 5.0}, "image": {"theta_deg": 4.0}}, "variants": {"bayes_hybrid_bridge_0lab": {"text": sample_m, "image": sample_m, "paired": sample_m, "summary": {"view3_mean_f1": 0.5}}}}},
        "bootstrap_vs_std_ce": {"l_mnar_joint": {"delta_macro_f1": 0.1, "ci95_low": 0.05, "ci95_high": 0.15, "p_value": 0.001}},
    }
    rep = tmp_path / "PILOT_F1_REPORT.md"
    _render_report(sample_res, rep)
    assert "$" not in rep.read_text(encoding="utf8")
