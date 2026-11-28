"""Mathematical-definition, protocol-equivalence, and author-parity regression tests."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from mnarm.core.determinism import build_run_lock
from mnarm.core.io import sha256_bytes
from mnarm.data.crisismmd import CRISISMMD_CLASSES, load_crisismmd_contracts, uid_hash
from mnarm.eval.metrics import holm_bonferroni_adjust, paired_bootstrap_diff
from mnarm.methods import PALRLSClassifier, StreamingClassBank, mg_clip_eval_logits, train_mg_clip_stage
from mnarm.methods.iterative_merging import _opcm_merge
from mnarm.methods.logit_controls import fit_gr_clip_gap, gr_clip_logits
from mnarm.methods.spherical_stats import _estimate_conjugate_weights, _identity_anchored_procrustes, _safe_precision_from_cov
from mnarm.models.vlm import attach_qv_lora, l2_normalize
from mnarm.runners.iccv_supplementary import _summarize_piece1
from mnarm.runners.stage_ops import train_lora_stage


def test_chan_welford_streaming_scatter_exact_under_class_reappearance_and_singleton() -> None:
    torch.manual_seed(101)
    dim, classes = 32, [0, 1, 2, 3]
    x_by_c = {c: l2_normalize(torch.randn(15, dim)) for c in classes}
    # Deliberate repeated classes across stages including a singleton (n_new == 1) in stage 3!
    batches = [
        ([(0, x_by_c[0][:6]), (1, x_by_c[1][:8])]),
        ([(0, x_by_c[0][6:14]), (2, x_by_c[2][:10]), (3, x_by_c[3][:15])]),
        ([(0, x_by_c[0][14:]), (1, x_by_c[1][8:]), (2, x_by_c[2][10:])]),
    ]
    bank = StreamingClassBank()
    for grp in batches:
        xs = torch.cat([x for _, x in grp], dim=0)
        ys = torch.cat([torch.full((x.shape[0],), c, dtype=torch.long) for c, x in grp], dim=0)
        bank.update_stage(u_expert_t=xs, u_base_t=xs, y_t=ys, u_expert_i=xs, u_base_i=xs, y_i=ys)

    stats = bank.canonical_stats(classes)["towers"]["text"]
    x_all = l2_normalize(torch.cat([x_by_c[c] for c in classes], dim=0)).double()
    y_all = torch.cat([torch.full((15,), c, dtype=torch.long) for c in classes], dim=0)
    sc_ref = torch.zeros(dim, dim, dtype=torch.float64)
    for c in classes:
        xc = x_all[y_all == c]
        mc = xc.mean(dim=0)
        assert float((stats["centroids"][c] - mc).abs().max().item()) < 1e-12
        assert stats["counts"][c] == 15
        sc_ref += (xc - mc.unsqueeze(0)).T @ (xc - mc.unsqueeze(0))
    assert stats["scatter_dof"] == sum(15 - 1 for _ in classes) == 56
    assert stats["total_samples"] == 60
    assert float((stats["within_scatter"] - sc_ref).abs().max().item()) < 1e-11


def test_crisismmd_multiseed_protocol_equivalence_across_seed11_and_seed22(tmp_path: Path) -> None:
    seq_dir, img_dir = tmp_path / "seq", tmp_path / "imgs"
    seq_dir.mkdir(parents=True)
    (img_dir / "data_image").mkdir(parents=True)
    rows_sp = {"train": [], "dev": [], "test": []}
    for sp, per_cls in (("train", 10), ("dev", 3), ("test", 3)):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["tweet_id", "image_id", "tweet_text"], delimiter="\t")
        writer.writeheader()
        for c in range(8):
            for i in range(per_cls):
                tid, iid = f"{sp}_{c}_{i}", f"img_{sp}_{c}_{i}"
                txt = f"tweet {CRISISMMD_CLASSES[c]} {sp} {i}"
                sid = uid_hash(tid, iid)
                rel = f"data_image/u_{sid[:12]}.jpg"
                Image.new("RGB", (8, 8), color=(c * 25 % 255, i * 20 % 255, 70)).save(img_dir / rel)
                writer.writerow({"tweet_id": tid, "image_id": iid, "tweet_text": txt})
                rows_sp[sp].append({"sample_id": sid, "label": c, "image": rel, "text_sha256": sha256_bytes(txt.encode("utf8"))})
        with zipfile.ZipFile(seq_dir / "splits.zip", "a") as zf:
            zf.writestr(f"task_humanitarian_text_img_{sp}.tsv", buf.getvalue())
    (seq_dir / "manifest.json").write_text(json.dumps({"rows": rows_sp}), encoding="utf8")
    (seq_dir / "PLANS.json").write_text(json.dumps({p: [{"stage": s + 1, "seen_classes": [], "buffer": [], "calibration_latest_views": []} for s in range(4)] for p in ("P1", "P2", "P3", "P4")}), encoding="utf8")
    (seq_dir / "partition_contract.json").write_text(json.dumps({"protocols": {p: [{"train_ids": [], "modalities": ["text", "image"]} for _ in range(4)] for p in ("P1", "P2", "P3", "P4")}}), encoding="utf8")

    for pr in ("P1", "P2", "P3", "P4"):
        c11 = load_crisismmd_contracts(seq_dir, img_dir, pr, allow_test=True, seed=11, split_seed=11, class_order_seed=11)
        c22 = load_crisismmd_contracts(seq_dir, img_dir, pr, allow_test=True, seed=22, split_seed=11, class_order_seed=22)
        fit11 = [r["uid"] for st in c11["stages"] for r in st["fit"]]
        fit22 = [r["uid"] for st in c22["stages"] for r in st["fit"]]
        cal11 = [r["uid"] for r in c11["stages"][-1]["calibration_latest_views"]]
        cal22 = [r["uid"] for r in c22["stages"][-1]["calibration_latest_views"]]
        assert len(fit11) == len(set(fit11)) and len(fit22) == len(set(fit22))
        assert set(fit11) == set(fit22) and set(cal11) == set(cal22)
        assert [len(st["new_classes"]) for st in c11["stages"]] == [2, 2, 2, 2]
        assert [len(st["new_classes"]) for st in c22["stages"]] == [2, 2, 2, 2]
        assert [r["uid"] for r in c11["dev"]] == [r["uid"] for r in c22["dev"]]
        assert [r["uid"] for r in c11["test"]] == [r["uid"] for r in c22["test"]]


def test_ncm_identity_precision_and_css_ct_loo_unbiased_k3_k4() -> None:
    g = torch.Generator().manual_seed(55)
    d = 64
    # Strongly anisotropic covariance should STILL give strict I_d when partial_cov_mode == "isotropic_vmf", even if is_partial=False!
    aniso = torch.diag(torch.linspace(0.1, 10.0, d, dtype=torch.float64))
    prec_full_vmf = _safe_precision_from_cov(aniso, is_partial=False, partial_cov_mode="isotropic_vmf")
    assert float((prec_full_vmf - torch.eye(d, dtype=torch.float64)).abs().max().item()) < 1e-12

    # Unbiased LOO on K=3 & K=4 partial towers: prompts rotated by orthogonal Q should NOT collapse to r_p_eff=0
    sk = torch.randn(d, d, dtype=torch.float64, generator=g) / (d ** 0.5)
    q_mat = torch.linalg.matrix_exp(0.15 * (sk - sk.T))
    for k_obs in (3, 4):
        obs = list(range(k_obs))
        p_all = l2_normalize(torch.randn(k_obs + 2, d, dtype=torch.float64, generator=g))
        centroids = {c: p_all[c].double() @ q_mat.T for c in obs}
        conj = _estimate_conjugate_weights(obs, centroids, p_all, {}, None, anchor_weight=1.0)
        assert conj["r_p_eff"] > 0.75, f"K={k_obs} LOO bias collapsed r_p_eff={conj['r_p_eff']}"


def test_seq_lmnar_cumulative_graph_retains_old_classes() -> None:
    from mnarm.models.vlm import tower_digest

    class _MockStore:
        def make_batch_specs(self, records, batch_size, shuffle=False, seed=11):
            by_m: dict[tuple[str, ...], list[dict]] = {}
            for r in records:
                by_m.setdefault(tuple(sorted(r["mask"])), []).append(r)
            return [(vw, rs[i : i + batch_size]) for vw, rs in by_m.items() for i in range(0, len(rs), batch_size)]

        def collate_chunk(self, views, chunk):
            return {"x": torch.eye(8)[[int(r["label"]) for r in chunk]], "label": torch.tensor([int(r["label"]) for r in chunk], dtype=torch.long), "sample_id": [r["uid"] for r in chunk]}

    class _MiniVLM:
        device, rank, alpha, embed_dim, logit_scale = "cpu", 2, 4, 8, 10.0
        def __init__(self) -> None:
            self.w = nn.Parameter(torch.eye(8))
            self.clip = nn.Identity()
        def restore_base_weights(self): pass
        def reset_lora(self): pass
        def load_lora_state(self, s): pass
        def lora_parameters(self): return [self.w]
        def encode_text(self, inp): return inp["x"] @ self.w
        def encode_image(self, inp): return inp["x"] @ self.w
        def lora_state(self): return {"w": self.w.detach().clone()}

    hist = [{"uid": f"u{c}", "label": c, "mask": ["image"]} for c in (0, 1)]
    st2_fit = [{"uid": f"u{c}", "label": c, "mask": ["text", "image"]} for c in (2, 3)]
    hist.extend(st2_fit)
    _, meta = train_lora_stage(
        vlm=_MiniVLM(), init_lora=None, records=st2_fit, protos_full=torch.eye(8), seen_classes=[0, 1, 2, 3],
        objective="l_mnar", stage_idx=2, epochs=1, effective_batch_size=2, micro_batch_size=2,
        lr=1e-3, weight_decay=0.0, seed=11, use_amp=False, tensor_store=_MockStore(), graph_records=hist,
    )
    _, meta_no_hist = train_lora_stage(
        vlm=_MiniVLM(), init_lora=None, records=st2_fit, protos_full=torch.eye(8), seen_classes=[0, 1, 2, 3],
        objective="l_mnar", stage_idx=2, epochs=1, effective_batch_size=2, micro_batch_size=2,
        lr=1e-3, weight_decay=0.0, seed=11, use_amp=False, tensor_store=_MockStore(), graph_records=None,
    )
    # Cumulative graph_records preserves [0, 1] image-only co-observation component instead of shattering into singletons [0], [1]
    assert meta["graph_components"] == [[0, 1], [2, 3]]
    assert meta_no_hist["graph_components"] == [[0], [1], [2, 3]]


def test_mg_clip_single_softmax_mnar_and_opcm_diagonal_and_gr_c2() -> None:
    torch.manual_seed(77)
    # 1. Single-softmax MG-CLIP allows unobserved class 0 with high cross_score to win across 6 classes!
    class _H6(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.eye(8)[:6])
            self.initialized = torch.tensor([False, True, True, True, True, True])
        def forward(self, u: torch.Tensor, classes: list[int]) -> torch.Tensor:
            return torch.zeros((u.shape[0], len(classes)))  # uniform over initialized [1..5] => 1/5 = 0.2 each
    heads = {"text": _H6(), "image": _H6()}
    cross = {"image": torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), "paired": torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}
    out = mg_clip_eval_logits(None, torch.randn(1, 8), cross, heads, [0, 1, 2, 3, 4, 5], 10.0)
    # Under single softmax: (1*1.0 + 4*0)/5 = 0.20 > (1*0 + 4*0.20)/5 = 0.16 -> class 0 WINS! (Double softmax would cap p_c at ~0.35 -> 0.07 < 0.16!)
    assert int(out["image"].argmax(dim=-1).item()) == 0

    # 2. OPCM in-place diagonal zeroing inside active signal subspace: verify projection u^T (cleaned) v^T has zero diagonal on [split_rank:pos_rank]!
    torch.manual_seed(78)
    u_orth, _ = torch.linalg.qr(torch.randn(12, 8))
    v_orth, _ = torch.linalg.qr(torch.randn(12, 8))
    m0 = u_orth @ torch.diag(torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])) @ v_orth.T
    # m1 is pure diagonal in m0's singular basis across indices [0..7] -> after removing [0..split_rank] and diagonal [split_rank..8], cleaned = 0!
    m1 = u_orth @ torch.diag(torch.ones(8)) @ v_orth.T
    merged_opcm = _opcm_merge([{"v": m0}, {"v": m1}], ["v"], alpha=0.5)["v"]
    cos_dir = float((merged_opcm.reshape(-1) @ m0.reshape(-1)).item() / (merged_opcm.norm().item() * m0.norm().item()))
    assert cos_dir > 0.99999, f"OPCM diagonal was not zeroed in-place (cos={cos_dir})"

    # 3. MG-CLIP epoch-level MGP early stopping stops before max epochs when gap_alpha is exceeded
    class _MGStore:
        def make_batch_specs(self, records, batch_size, shuffle=False, seed=11):
            return [(("image",), records)]
        def collate_chunk(self, views, chunk):
            return {"pixel_values": torch.eye(8)[: len(chunk)], "label": torch.arange(len(chunk)), "sample_id": [r["uid"] for r in chunk]}

    class _MGVLM:
        device, rank, alpha, embed_dim, logit_scale = "cpu", 2, 4, 8, 10.0
        def __init__(self) -> None:
            self.w = nn.Parameter(torch.eye(8))
            self.clip = nn.Module()
            self.clip.linear = nn.Linear(8, 8, bias=False)
        def restore_base_weights(self): pass
        def reset_lora(self):
            with torch.no_grad(): self.w.copy_(torch.eye(8))
        def load_lora_state(self, s):
            with torch.no_grad(): self.w.copy_(s["w"])
        def lora_parameters(self): return [self.w]
        def encode_text(self, inp): return inp["x"]
        def encode_image(self, inp): return inp["pixel_values"] @ self.w
        def encode_prompts_current(self, prompts): return torch.eye(8)[: len(prompts)]
        def lora_state(self): return {"w": self.w.detach().cpu().clone()}

    mg_heads: dict[str, nn.Module] = {}
    recs_mg = [{"uid": f"m{i}", "label": i, "mask": ["image"]} for i in range(4)]
    _, mg_heads_out = train_mg_clip_stage(
        vlm=_MGVLM(), prev_state=None, mg_heads=mg_heads, records=recs_mg, protos_full=torch.eye(8)[:4],
        old_classes=[], new_classes=[0, 1, 2, 3], stage_idx=1, epochs=4, effective_batch_size=4, micro_batch_size=4,
        lr=0.2, weight_decay=0.0, seed=11, use_amp=False, tensor_store=_MGStore(), gap_alpha=1e-6, mgc_epochs=1,
    )
    assert mg_heads_out["text"].mgp_trace[1]["stopped_epoch"] < 4

    # 4. GR-CLIP C=2 with full protos uses bank mean instead of degenerate antipodal half-diff
    gap = fit_gr_clip_gap(torch.randn(8, 12), torch.randn(8, 12))
    protos_full = l2_normalize(torch.eye(12)[:6])
    gl = gr_clip_logits(torch.randn(3, 12), torch.randn(3, 12), protos_full, [0, 1], 10.0, gap)
    assert torch.isfinite(gl["paired"]).all()


def test_bootstrap_finite_p_holm_balanced_piece1_and_run_lock(tmp_path: Path) -> None:
    a = torch.tensor([[10.0, 0.0]] * 20)
    b = torch.tensor([[0.0, 10.0]] * 20)
    y = torch.zeros(20, dtype=torch.long)
    res = paired_bootstrap_diff(a, b, y, [0, 1], n_resamples=99, seed=11)
    assert res["p_value"] >= 2.0 / 100.0
    adj = holm_bonferroni_adjust({"c1": dict(res), "c2": {"delta_macro_f1": 0.0, "ci95_low": -0.1, "ci95_high": 0.1, "p_value": 0.5, "n_resamples": 99}})
    assert adj["c1"]["p_value_holm"] >= adj["c1"]["p_value"] and adj["c2"]["p_value_holm"] <= 1.0

    # Balanced Piece 1 intersects datasets/backbones/protocols across seeds
    ms_root = tmp_path / "ms"
    for s, ds_list in ((11, ("dmd", "crisismmd")), (22, ("dmd",))):
        for ds_n in ds_list:
            p = ms_root / ds_n / f"seed_{s}" / "clip_b16_p4" / "ICCV_PROTOCOL_TEST_RESULTS.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            val = 0.80 if ds_n == "dmd" else 0.10
            p.write_text(json.dumps({
                "config": {"dataset": ds_n, "backbone": "clip_b16", "protocol": "P4"},
                "arms": {"tsvm": {"regimes": {"raw": {"paired": {"macro_f1": val}, "summary": {"view3_mean_f1": val}}, "zero_mnar": {"paired": {"macro_f1": val}, "summary": {"view3_mean_f1": val}}}}},
            }), encoding="utf8")
    p1 = _summarize_piece1(ms_root, tmp_path)
    assert p1["balanced"] is True and p1["n_common_cells"] == 1
    assert abs(p1["arms"]["tsvm"]["all_css"]["mean"] - 80.0) < 1e-6

    lk = build_run_lock({"allow_test": False, "deterministic": True})
    assert "turbo_kernels" in lk and "effective_deterministic" in lk
