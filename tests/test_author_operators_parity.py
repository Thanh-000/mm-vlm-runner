"""Tests for 9 verbatim author merging operators, logit controls, retrieval R@1, and ICCV Same-Bank E2E."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from mnarm.core import sha256_bytes
from mnarm.data import uid_hash
from mnarm.eval import eval_retrieval_r1
from mnarm.methods import (
    ALL_MERGE_METHODS,
    apply_bias,
    fit_cal_bias,
    fit_unlabeled_center,
    merge_task_deltas,
)
from mnarm.methods.subspace_merging import _dop_merge, _wudi_matrix, _wudi_merge
from mnarm.models.vlm import attach_qv_lora, tower_digest


def test_all_nine_author_mergers_and_mnar_zero_towers() -> None:
    torch.manual_seed(17)
    # 4 stages with MNAR pattern: text_model zero in stages 1-2, both active in stages 3-4
    deltas = []
    for st in range(4):
        d_txt = torch.zeros(32, 32) if st < 2 else torch.randn(32, 32) * 0.05
        d_vis = torch.randn(32, 32) * 0.05
        deltas.append({"text_model.l0.q_proj": d_txt, "vision_model.l0.q_proj": d_vis})

    for m in ALL_MERGE_METHODS:
        merged = merge_task_deltas(deltas, m)
        assert set(merged.keys()) == set(deltas[0].keys())
        for k, ten in merged.items():
            assert ten.shape == (32, 32)
            assert torch.isfinite(ten).all(), f"{m} produced nonfinite values on {k}"

    # Verify exact exposure denominator (2 active stages for text, 4 for vision)
    m_exp = merge_task_deltas(deltas, "exposure")
    exp_txt = 0.5 * (deltas[2]["text_model.l0.q_proj"] + deltas[3]["text_model.l0.q_proj"])
    assert float((m_exp["text_model.l0.q_proj"] - exp_txt).abs().max().item()) < 1e-6

    # Verify exact magmax scale (0.5 vs 1.0) and >= tie-break
    m_half = merge_task_deltas(deltas, "magmax")
    m_full = merge_task_deltas(deltas, "magmax_s1")
    assert float((m_full["vision_model.l0.q_proj"] - 2.0 * m_half["vision_model.l0.q_proj"]).abs().max().item()) < 1e-6
    tie_deltas = [{"k": torch.tensor([[2.0, -3.0]])}, {"k": torch.tensor([[-2.0, 3.0]])}]
    assert torch.equal(merge_task_deltas(tie_deltas, "magmax_s1")["k"], torch.tensor([[-2.0, 3.0]]))

    # Verify dc_lora single-active-stage bypass preserves exact tensor without SVD pruning
    d_single_txt = [
        {"t": torch.zeros(32, 32), "v": torch.randn(32, 32) * 0.05},
        {"t": torch.zeros(32, 32), "v": torch.randn(32, 32) * 0.05},
        {"t": torch.randn(32, 32) * 0.05, "v": torch.randn(32, 32) * 0.05},
    ]
    m_dcl = merge_task_deltas(d_single_txt, "dc_lora")
    assert float((m_dcl["t"] - d_single_txt[2]["t"]).abs().max().item()) < 1e-6


def test_batched_wudi_and_sync_free_dop_exact_parity() -> None:
    torch.manual_seed(19)
    deltas = []
    for st in range(4):
        row = {
            "v.l0": torch.zeros(16, 16) if st == 1 else (torch.randn(16, 4) @ torch.randn(4, 16)) * 0.05,
            "v.l1": torch.zeros(16, 16) if st == 1 else (torch.randn(16, 4) @ torch.randn(4, 16)) * 0.05,
            "t.l0": torch.zeros(16, 16) if st == 0 else (torch.randn(16, 4) @ torch.randn(4, 16)) * 0.05,
        }
        deltas.append(row)
    keys = ["v.l0", "v.l1", "t.l0"]
    batched = _wudi_merge(deltas, keys, iter_num=40, lr=1e-4)
    for k in keys:
        single = _wudi_matrix([d[k] for d in deltas], iter_num=40, lr=1e-4)
        assert float((batched[k] - single).abs().max().item()) < 1e-5
    dop_a = _dop_merge(deltas, keys)
    dop_b = _dop_merge(deltas, keys)
    for k in keys:
        assert torch.isfinite(dop_a[k]).all() and torch.equal(dop_a[k], dop_b[k])


def test_logit_controls_and_retrieval_r1() -> None:
    torch.manual_seed(23)
    # Unlabeled centering zeroes class column means cleanly
    scores = torch.randn(100, 6) + torch.tensor([3.0, -2.0, 1.0, 0.0, -1.5, 4.0])
    uc = fit_unlabeled_center(scores)
    assert uc["labels_used"] is False
    centered = apply_bias(scores, uc["bias"])
    assert float(centered.mean(dim=0).abs().max().item()) < 1e-5

    # Calibration bias converges to convex optimum and recovers shifted early class accuracy
    feats = torch.eye(4).repeat(25, 1) * 4.0 + 0.1 * torch.randn(100, 4)
    labels = torch.arange(4).repeat(25)
    shifted = feats + torch.tensor([-6.0, -6.0, 2.0, 2.0])
    cb = fit_cal_bias(shifted, labels, [0, 1, 2, 3])
    assert cb["labels_used"] is True and cb["grad_inf"] < 1e-4
    fixed = apply_bias(shifted, cb["bias"])
    assert float((fixed.argmax(dim=-1) == labels).float().mean().item()) > 0.98

    # Retrieval R@1 is 1.0 for exact pairs and low when permuted
    emb = torch.nn.functional.normalize(torch.randn(64, 32), dim=-1)
    r_exact = eval_retrieval_r1(emb, emb)
    assert r_exact["i2t_r1"] == 1.0 and r_exact["t2i_r1"] == 1.0
    r_perm = eval_retrieval_r1(emb, emb[torch.randperm(64)])
    assert r_perm["i2t_r1"] < 0.25


def test_run_iccv_same_bank_end_to_end_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mnarm.runners import iccv_same_bank as sb_mod

    seq_dir, img_dir, out_dir = tmp_path / "seq", tmp_path / "imgs", tmp_path / "iccv_out"
    seq_dir.mkdir(); img_dir.mkdir()

    def _mk_row(idx: int, label: int, prefix: str, write_img: bool = True) -> dict[str, object]:
        tw, im_id, txt = f"tw{idx}", f"im{idx}", f"cls_{label} {prefix} {idx}"
        uid = uid_hash(tw, im_id)
        rel = f"data_image/{prefix}_{label}_{idx}.png"
        if write_img:
            p = img_dir / rel; p.parent.mkdir(parents=True, exist_ok=True)
            color = ((label * 31 + 17) % 250, (label * 67 + 43) % 250, (label * 97 + 71) % 250)
            Image.new("RGB", (8, 8), color=color).save(p)
        return {"sample_id": uid, "label": label, "image": rel, "text_sha256": sha256_bytes(txt.encode("utf8")), "tweet_id": tw, "image_id": im_id, "tweet_text": txt}

    train_rows, dev_rows, test_rows = [], [], []
    by_cls_tr: dict[int, list[str]] = {c: [] for c in range(8)}
    seq_id = 0
    for c in range(8):
        for _ in range(4):
            r = _mk_row(seq_id, c, "tr"); train_rows.append(r); by_cls_tr[c].append(str(r["sample_id"])); seq_id += 1
        for _ in range(2):
            dev_rows.append(_mk_row(seq_id, c, "dv")); seq_id += 1
        test_rows.append(_mk_row(seq_id, c, "te", write_img=False)); seq_id += 1

    (seq_dir / "manifest.json").write_text(json.dumps({"rows": {"train": train_rows, "dev": dev_rows, "test": test_rows}}), encoding="utf8")
    stages_cls = [([0, 1], ["image"]), ([4, 5], ["image"]), ([2, 3], ["text", "image"]), ([6, 7], ["text", "image"])]
    seen_acc: list[int] = []
    plans_p4, pc_p4 = [], []
    for st_i, (new_c, mods) in enumerate(stages_cls, 1):
        seen_acc = sorted(seen_acc + new_c)
        t_ids = [uid for c in new_c for uid in by_cls_tr[c]]
        buf_rows = [{"uid": by_cls_tr[c][0], "label": c, "mask": ["image"] if c in (0, 1, 4, 5) else ["text", "image"]} for c in seen_acc]
        cal_rows = [{"uid": by_cls_tr[c][1], "label": c, "mask": ["image"] if c in (0, 1, 4, 5) else ["text", "image"]} for c in seen_acc]
        plans_p4.append({"stage": st_i, "seen_classes": list(seen_acc), "buffer": buf_rows, "calibration_latest_views": cal_rows})
        pc_p4.append({"stage": st_i, "modalities": mods, "train_ids": t_ids})
    plans_p3 = [{**pl, "calibration_latest_views": [{"uid": r["uid"], "label": r["label"], "mask": ["text"] if r["label"] in (0, 1, 4, 5) else ["text", "image"]} for r in pl["calibration_latest_views"]]} for pl in plans_p4]
    pc_p3 = [{**s, "modalities": ["text"] if s["modalities"] == ["image"] else s["modalities"]} for s in pc_p4]
    (seq_dir / "PLANS.json").write_text(json.dumps({"P4": plans_p4, "P3": plans_p3}), encoding="utf8")
    (seq_dir / "partition_contract.json").write_text(json.dumps({"protocols": {"P4": pc_p4, "P3": pc_p3}}), encoding="utf8")
    with zipfile.ZipFile(seq_dir / "splits.zip", "w") as zf:
        for sp, rlist in (("train", train_rows), ("dev", dev_rows)):
            zf.writestr(f"task_humanitarian_text_img_{sp}.tsv", "\n".join(["tweet_id\timage_id\ttweet_text"] + [f"{r['tweet_id']}\t{r['image_id']}\t{r['tweet_text']}" for r in rlist]))

    init_calls = []

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_model = nn.Module(); self.text_model.l0 = nn.Module(); self.text_model.l0.q_proj = nn.Linear(16, 16, bias=False)
            self.vision_model = nn.Module(); self.vision_model.l0 = nn.Module(); self.vision_model.l0.q_proj = nn.Linear(16, 16, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.text_model.l0.q_proj.weight); nn.init.eye_(self.vision_model.l0.q_proj.weight)

    class _VLM:
        def __init__(self, backbone_key: str, *_a: object, **_kw: object) -> None:
            init_calls.append(1)
            self.backbone_key, self.device, self.rank, self.alpha, self.embed_dim, self.logit_scale = backbone_key, "cpu", 4, 8, 16, 12.0
            self.clip = _Net()
            self.qv_modules = attach_qv_lora(self.clip, rank=4, alpha=8)
            self.base_qv_weights = {n: m.base.weight.detach().clone() for n, m in self.qv_modules.items()}

        @staticmethod
        def _vec(c: int) -> torch.Tensor:
            v = torch.zeros(16); v[c % 8] = 1.0; v[8 + (c % 8)] = 0.5; return v / v.norm()

        def tokenize_texts(self, texts: list[str], **_kw: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.stack([self._vec(int(t.split("_", 1)[1].split(" ", 1)[0])) for t in texts])}

        def processor(self, images: list[Image.Image], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            del return_tensors
            rows = []
            for im in images:
                r0, g0, b0 = im.getpixel((0, 0))
                cid = min(range(8), key=lambda c: abs(r0 - (c * 31 + 17) % 250) + abs(g0 - (c * 67 + 43) % 250) + abs(b0 - (c * 97 + 71) % 250))
                rows.append(self._vec(cid).view(1, 4, 4).repeat(3, 1, 1))
            return {"pixel_values": torch.stack(rows)}

        def encode_text(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.clip.text_model.l0.q_proj(inp["input_ids"].float())

        def encode_image(self, inp: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.clip.vision_model.l0.q_proj(inp["pixel_values"].float()[:, 0].reshape(inp["pixel_values"].shape[0], 16))

        @torch.no_grad()
        def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
            return torch.stack([self._vec(i) for i in range(len(prompts))])

        @torch.no_grad()
        def encode_prompts_current(self, prompts: list[str]) -> torch.Tensor:
            tok = self.tokenize_texts(list(prompts)) if all("_" in p for p in prompts) else None
            if tok is not None:
                return torch.nn.functional.normalize(self.encode_text(tok).float(), dim=-1)
            return torch.nn.functional.normalize(self.clip.text_model.l0.q_proj(self.encode_prompts(prompts)), dim=-1)

        def lora_parameters(self) -> list[nn.Parameter]:
            return [p for m in self.qv_modules.values() for p in (m.lora_a, m.lora_b)]

        def lora_state(self) -> dict[str, torch.Tensor]:
            return {f"{n}.{p}": getattr(m, p).detach().cpu().clone() for n, m in self.qv_modules.items() for p in ("lora_a", "lora_b")}

        def load_lora_state(self, state: dict[str, torch.Tensor]) -> None:
            if not state: self.zero_lora(); return
            with torch.no_grad():
                for n, m in self.qv_modules.items():
                    m.lora_a.data.copy_(state[f"{n}.lora_a"]); m.lora_b.data.copy_(state[f"{n}.lora_b"])

        def zero_lora(self) -> None:
            with torch.no_grad():
                for m in self.qv_modules.values(): m.lora_b.zero_()

        def reset_lora(self) -> None:
            self.restore_base_weights()
            with torch.no_grad():
                for m in self.qv_modules.values(): torch.nn.init.kaiming_uniform_(m.lora_a, a=2.0); m.lora_b.zero_()

        def restore_base_weights(self) -> None:
            with torch.no_grad():
                for n, m in self.qv_modules.items(): m.base.weight.data.copy_(self.base_qv_weights[n])

        def apply_dense_delta(self, delta: dict[str, torch.Tensor]) -> None:
            with torch.no_grad():
                self.zero_lora()
                for n, m in self.qv_modules.items():
                    w0 = self.base_qv_weights[n]
                    m.base.weight.data.copy_(w0 + delta[n] if n in delta else w0)

        def extract_lora_delta(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {n: 2.0 * (state[f"{n}.lora_b"].float() @ state[f"{n}.lora_a"].float()).cpu() for n in self.qv_modules}

        def tower_digest(self, tower: str, kind: str, payload: dict[str, torch.Tensor]) -> str:
            return tower_digest(tower, kind, payload)

    monkeypatch.setattr(sb_mod, "MultimodalVLM", _VLM)
    cfg = {
        "schema": "mnarm-config-v1", "experiment": "test_iccv", "dataset": "crisismmd", "protocol": "P4",
        "allow_test": False, "backbone": "clip_b16", "model_id": "mock", "seed": 11, "deterministic": True,
        "device": "cpu", "use_amp": False, "lora_rank": 4, "lora_alpha": 8, "epochs": 1,
        "effective_batch_size": 4, "micro_batch_size": 4, "eval_batch_size": 8, "lr": 1e-3,
        "weight_decay": 0.0, "max_cached_images": 100, "workers": 1, "shrinkage": None,
        "include_mg_clip": True,
    }
    res = sb_mod.run_iccv_same_bank(cfg, seq_dir, img_dir, out_dir, protocols=["P4", "P3"])
    assert len(init_calls) == 1
    assert set(res["protocols"].keys()) == {"P4", "P3"}
    for p in ("P4", "P3"):
        p_res = json.loads((out_dir / f"clip_b16_{p.lower()}" / "ICCV_PROTOCOL_RESULTS.json").read_text(encoding="utf8"))
        assert len(p_res["arms"]) == len(ALL_MERGE_METHODS) + 5
        for arm in list(ALL_MERGE_METHODS) + ["seq_ft", "seq_replay256", "mg_clip", "joint_oracle"]:
            assert set(p_res["arms"][arm]["regimes"].keys()) == {"raw", "gr_clip", "unlab_center", "zero_mnar", "cal_bias"}
            assert "i2t_r1" in p_res["arms"][arm]["encoder_preservation"]["retrieval"]
            assert "i2t_r1" in p_res["arms"][arm]["encoder_preservation"]["retrieval_reanchored"]
        assert "raw" in p_res["arms"]["pal_rls"]["regimes"]
        assert {"cm_vmf", "css_ct"} <= set(p_res["ablation_tsvm"].keys())
    assert (out_dir / "ICCV_SAME_BANK_clip_b16_TABLE.tex").is_file()
    md_sum = (out_dir / "ICCV_SAME_BANK_clip_b16_SUMMARY.md").read_text(encoding="utf8")
    assert "$" not in md_sum
    fig_dir = out_dir / "figures"
    for stem in ("fig1_same_bank_regimes_clip_b16", "fig2_mnar_pathology_hubness_clip_b16", "fig3_encoder_preservation_pareto_clip_b16", "fig4_ablation_bootstrap_clip_b16"):
        assert (fig_dir / f"{stem}.png").stat().st_size > 1000
        assert (fig_dir / f"{stem}.pdf").stat().st_size > 1000
    assert len(res.get("figures", [])) == 8
