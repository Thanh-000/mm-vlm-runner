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


def test_extract_contracts_from_a2z_zip_supports_lite_a2z_and_rejects_html(tmp_path: Path) -> None:
    from mnarm.core import sha256_bytes
    from mnarm.data import extract_contracts_from_a2z_zip

    payloads = {
        "manifest.json": b'{"rows":{"train":[],"dev":[],"test":[]}}',
        "PLANS.json": b'{"P4":[]}',
        "partition_contract.json": b'{"protocols":{}}',
        "splits.zip": b"PK\x05\x06" + b"\x00" * 18,
    }
    shas = {k: sha256_bytes(v) for k, v in payloads.items()}

    # 1) Lightweight flat contract ZIP with CONTRACT_SHA256.json
    lite_zip = tmp_path / "lite.zip"
    with zipfile.ZipFile(lite_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CONTRACT_SHA256.json", json.dumps(shas))
        for k, v in payloads.items():
            zf.writestr(k, v)
    out_lite = extract_contracts_from_a2z_zip(lite_zip, tmp_path / "out_lite")
    for k, v in payloads.items():
        assert (out_lite / k).read_bytes() == v

    # Tampered SHA inside lite ZIP must raise ValueError
    tampered_zip = tmp_path / "tampered_lite.zip"
    bad_shas = {**shas, "PLANS.json": "0" * 64}
    with zipfile.ZipFile(tampered_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CONTRACT_SHA256.json", json.dumps(bad_shas))
        for k, v in payloads.items():
            zf.writestr(k, v)
    with pytest.raises(ValueError, match="SHA mismatch"):
        extract_contracts_from_a2z_zip(tampered_zip, tmp_path / "out_tampered")

    # 2) Legacy BUNDLE_MANIFEST.json + blobs/<sha> A2Z ZIP
    a2z_zip = tmp_path / "a2z.zip"
    bm = {"files": {f"dense_bundle/kit/inputs/{k}": {"sha256": shas[k]} for k in payloads}}
    with zipfile.ZipFile(a2z_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BUNDLE_MANIFEST.json", json.dumps(bm))
        for k, v in payloads.items():
            zf.writestr(f"blobs/{shas[k]}", v)
    out_a2z = extract_contracts_from_a2z_zip(a2z_zip, tmp_path / "out_a2z")
    for k, v in payloads.items():
        assert (out_a2z / k).read_bytes() == v

    # 3) HTML page saved as .zip must raise BadZipFile mentioning HTML page
    html_zip = tmp_path / "bad_drive.zip"
    html_zip.write_bytes(b"<!DOCTYPE html><html><head><title>Google Drive</title></head></html>")
    with pytest.raises(zipfile.BadZipFile, match="HTML page"):
        extract_contracts_from_a2z_zip(html_zip, tmp_path / "out_html")


def test_run_pilot_f1_end_to_end_p4_mock_vlm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise all 4 stages of P4 end-to-end (including image-only Stage 1 base_zero cache check)."""
    from torch import nn
    from mnarm.core import sha256_bytes
    from mnarm.models.vlm import attach_qv_lora, tower_digest
    from mnarm.runners import pilot_f1 as pf1_mod

    seq_dir = tmp_path / "seq"
    img_dir = tmp_path / "imgs"
    out_dir = tmp_path / "out"
    seq_dir.mkdir()
    img_dir.mkdir()

    def _mk_row(idx: int, label: int, prefix: str, write_img: bool = True) -> dict[str, object]:
        tw, im_id, txt = f"tw{idx}", f"im{idx}", f"cls_{label} {prefix} {idx}"
        uid = uid_hash(tw, im_id)
        rel = f"data_image/{prefix}_{label}_{idx}.png"
        if write_img:
            p = img_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            color = ((label * 31 + 17) % 250, (label * 67 + 43) % 250, (label * 97 + 71) % 250)
            Image.new("RGB", (8, 8), color=color).save(p)
        return {
            "sample_id": uid, "label": label, "image": rel,
            "text_sha256": sha256_bytes(txt.encode("utf8")),
            "tweet_id": tw, "image_id": im_id, "tweet_text": txt,
        }

    train_rows, dev_rows, test_rows = [], [], []
    by_cls_tr: dict[int, list[str]] = {c: [] for c in range(8)}
    seq_id = 0
    for c in range(8):
        for _ in range(4):
            r = _mk_row(seq_id, c, "tr")
            train_rows.append(r)
            by_cls_tr[c].append(str(r["sample_id"]))
            seq_id += 1
        for _ in range(2):
            dev_rows.append(_mk_row(seq_id, c, "dv"))
            seq_id += 1
        test_rows.append(_mk_row(seq_id, c, "te", write_img=False))
        seq_id += 1

    manifest = {"rows": {"train": train_rows, "dev": dev_rows, "test": test_rows}}
    (seq_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    # CrisisMMD P4 schedule: St1 image-only [0,1], St2 image-only [4,5], St3 paired [2,3], St4 paired [6,7]
    stages_cls = [([0, 1], ["image"]), ([4, 5], ["image"]), ([2, 3], ["text", "image"]), ([6, 7], ["text", "image"])]
    seen_acc: list[int] = []
    plans_p4, pc_p4 = [], []
    for st_i, (new_c, mods) in enumerate(stages_cls, 1):
        seen_acc = sorted(seen_acc + new_c)
        t_ids = [uid for c in new_c for uid in by_cls_tr[c]]
        buf_rows = [{"uid": by_cls_tr[c][0], "label": c, "mask": ["image" if c in (0, 1, 4, 5) else "text", "image"] if c not in (0, 1, 4, 5) else ["image"]} for c in seen_acc]
        plans_p4.append({"stage": st_i, "seen_classes": list(seen_acc), "buffer": buf_rows, "calibration_latest_views": []})
        pc_p4.append({"stage": st_i, "modalities": mods, "train_ids": t_ids})

    (seq_dir / "PLANS.json").write_text(json.dumps({"P4": plans_p4, "P3": plans_p4}), encoding="utf8")
    pc_p3 = [{"stage": s["stage"], "modalities": ["text"] if s["modalities"] == ["image"] else s["modalities"], "train_ids": s["train_ids"]} for s in pc_p4]
    (seq_dir / "partition_contract.json").write_text(json.dumps({"protocols": {"P4": pc_p4, "P3": pc_p3}}), encoding="utf8")
    with zipfile.ZipFile(seq_dir / "splits.zip", "w") as zf:
        for sp, rlist in (("train", train_rows), ("dev", dev_rows)):
            lines = ["tweet_id\timage_id\ttweet_text"] + [f"{r['tweet_id']}\t{r['image_id']}\t{r['tweet_text']}" for r in rlist]
            zf.writestr(f"task_humanitarian_text_img_{sp}.tsv", "\n".join(lines))

    class _MockClipNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_model = nn.Module()
            self.text_model.l0 = nn.Module()
            self.text_model.l0.q_proj = nn.Linear(16, 16, bias=False)
            self.vision_model = nn.Module()
            self.vision_model.l0 = nn.Module()
            self.vision_model.l0.q_proj = nn.Linear(16, 16, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.text_model.l0.q_proj.weight)
                nn.init.eye_(self.vision_model.l0.q_proj.weight)

    class _MockVLM:
        def __init__(self, backbone_key: str, model_path: str, device: str = "cpu", rank: int = 4, alpha: int = 8, **_kw: object) -> None:
            self.backbone_key = backbone_key
            self.device = "cpu"
            self.rank = rank
            self.alpha = alpha
            self.embed_dim = 16
            self.logit_scale = 12.0
            self.clip = _MockClipNet()
            self.qv_modules = attach_qv_lora(self.clip, rank=rank, alpha=alpha)
            self.base_qv_weights = {n: mod.base.weight.detach().clone() for n, mod in self.qv_modules.items()}

        @staticmethod
        def _vec_for_cls(c: int) -> torch.Tensor:
            v = torch.zeros(16, dtype=torch.float32)
            v[c % 8] = 1.0
            v[8 + (c % 8)] = 0.5
            return v / v.norm()

        def tokenize_texts(self, texts: list[str], **_kw: object) -> dict[str, torch.Tensor]:
            rows = []
            for txt in texts:
                cls_id = int(txt.split("_", 1)[1].split(" ", 1)[0])
                rows.append(self._vec_for_cls(cls_id))
            return {"input_ids": torch.stack(rows, dim=0)}

        def processor(self, images: list[Image.Image], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            del return_tensors
            rows = []
            for im in images:
                r0, g0, b0 = im.getpixel((0, 0))
                cls_id = min(range(8), key=lambda c: abs(r0 - (c * 31 + 17) % 250) + abs(g0 - (c * 67 + 43) % 250) + abs(b0 - (c * 97 + 71) % 250))
                pv = self._vec_for_cls(cls_id).view(1, 4, 4).repeat(3, 1, 1)
                rows.append(pv)
            return {"pixel_values": torch.stack(rows, dim=0)}

        def encode_text(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.clip.text_model.l0.q_proj(inputs["input_ids"].float())

        def encode_image(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            x = inputs["pixel_values"].float()[:, 0, :, :].reshape(inputs["pixel_values"].shape[0], 16)
            return self.clip.vision_model.l0.q_proj(x)

        @torch.no_grad()
        def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
            return torch.stack([self._vec_for_cls(i) for i in range(len(prompts))], dim=0)

        def lora_parameters(self) -> list[nn.Parameter]:
            return [p for m in self.qv_modules.values() for p in (m.lora_a, m.lora_b)]

        def lora_state(self) -> dict[str, torch.Tensor]:
            return {f"{n}.{p}": getattr(m, p).detach().cpu().clone() for n, m in self.qv_modules.items() for p in ("lora_a", "lora_b")}

        def load_lora_state(self, state: dict[str, torch.Tensor]) -> None:
            if not state:
                self.zero_lora()
                return
            with torch.no_grad():
                for n, m in self.qv_modules.items():
                    m.lora_a.data.copy_(state[f"{n}.lora_a"])
                    m.lora_b.data.copy_(state[f"{n}.lora_b"])

        def zero_lora(self) -> None:
            with torch.no_grad():
                for m in self.qv_modules.values():
                    m.lora_b.zero_()

        def reset_lora(self) -> None:
            self.restore_base_weights()
            with torch.no_grad():
                for m in self.qv_modules.values():
                    torch.nn.init.kaiming_uniform_(m.lora_a, a=2.0)
                    m.lora_b.zero_()

        def restore_base_weights(self) -> None:
            with torch.no_grad():
                for n, m in self.qv_modules.items():
                    m.base.weight.data.copy_(self.base_qv_weights[n])

        def apply_dense_delta(self, delta: dict[str, torch.Tensor]) -> None:
            with torch.no_grad():
                self.zero_lora()
                for n, m in self.qv_modules.items():
                    w0 = self.base_qv_weights[n]
                    m.base.weight.data.copy_(w0 + delta[n] if n in delta else w0)

        def extract_lora_delta(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            sc = float(self.alpha) / float(self.rank)
            return {n: sc * (state[f"{n}.lora_b"].float() @ state[f"{n}.lora_a"].float()).cpu() for n in self.qv_modules}

        def tower_digest(self, tower: str, kind: str, payload: dict[str, torch.Tensor]) -> str:
            return tower_digest(tower, kind, payload)

    monkeypatch.setattr(pf1_mod, "MultimodalVLM", _MockVLM)
    cfg = {
        "schema": "mnarm-config-v1", "experiment": "test_p4", "dataset": "crisismmd", "protocol": "P4",
        "allow_test": False, "backbone": "clip_b16", "model_id": "mock", "seed": 11, "deterministic": True,
        "device": "cpu", "use_amp": False, "lora_rank": 4, "lora_alpha": 8, "epochs": 1,
        "effective_batch_size": 4, "micro_batch_size": 4, "eval_batch_size": 8, "lr": 1e-3,
        "weight_decay": 0.0, "max_cached_images": 100, "workers": 1, "shrinkage": None,
        "gates": {"max_theta_deg": 15.0, "min_paired_f1": 0.5, "min_view3_mean_f1": 0.5},
    }
    res = pf1_mod.run_pilot_f1(cfg, seq_dir, img_dir, out_dir)
    assert (out_dir / "PILOT_F1_DEV_RESULTS.json").is_file()
    assert (out_dir / "PILOT_F1_REPORT.md").is_file()
    assert (out_dir / "RUN_LOCK.json").is_file()
    assert "$" not in (out_dir / "PILOT_F1_REPORT.md").read_text(encoding="utf8")
    expected_arms = {f"{obj}_{suf}" for obj in ("std_ce", "l_mnar") for suf in ("tsvm", "ta_sum", "seq_replay", "joint")}
    assert set(res["arms"].keys()) == expected_arms
    assert len(res["stage_trajectory"]["std_ce_seq_replay"]) == 4
    assert len(res["stage_trajectory"]["l_mnar_seq_replay"]) == 4
    assert res["verdict"]["best_paired_f1"] > 0.7

    # Also verify shared-VLM multi-protocol runner across P4 and P3 without reloading model weights
    from mnarm.runners import crisismmd_p1_p4 as p14_mod

    init_calls = []
    orig_init = _MockVLM.__init__

    def _counting_init(self, *a: object, **kw: object) -> None:
        init_calls.append(1)
        orig_init(self, *a, **kw)

    monkeypatch.setattr(_MockVLM, "__init__", _counting_init)
    monkeypatch.setattr(p14_mod, "MultimodalVLM", _MockVLM)
    suite_out = tmp_path / "suite_out"
    suite = p14_mod.run_crisismmd_p1_p4(cfg, seq_dir, img_dir, suite_out, protocols=["P4", "P3"])
    assert len(init_calls) == 1
    assert set(suite["protocols"].keys()) == {"P4", "P3"}
    assert (suite_out / "CRISISMMD_P1_P4_clip_b16_SUMMARY.md").is_file()
    assert "$" not in (suite_out / "CRISISMMD_P1_P4_clip_b16_SUMMARY.md").read_text(encoding="utf8")


