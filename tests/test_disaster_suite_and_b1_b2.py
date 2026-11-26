"""Tests for Phase B1 math fixes, Phase B2 batched GPU DOP + tensor cache, and Phase B3 disaster suite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from mnarm.data import PreprocessedTensorStore, assert_zero_split_overlap, load_benchmark_contracts
from mnarm.eval import compute_balanced_ece, eval_classification_metrics, eval_view_suite
from mnarm.methods import SelfNormalizedBayesTransportHead, StreamingClassBank, merge_task_deltas
from mnarm.methods.iterative_merging import _dop_merge, _dop_merge_pair
from mnarm.models.vlm import l2_normalize


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_b1_partial_tower_epistemic_cov_and_unconditional_equalization() -> None:
    g = torch.Generator().manual_seed(2026)
    d = 128
    classes = list(range(8))
    early, late = [0, 1, 4, 5], [2, 3, 6, 7]
    mu_t = F.normalize(torch.randn(1, d, dtype=torch.float64, generator=g), dim=-1)
    mu_i = F.normalize(torch.randn(1, d, dtype=torch.float64, generator=g), dim=-1)
    mu_p = F.normalize(torch.randn(1, d, dtype=torch.float64, generator=g), dim=-1)
    sem = F.normalize(torch.randn(8, d, dtype=torch.float64, generator=g), dim=-1)
    basis_t = torch.randn(d, 16, dtype=torch.float64, generator=g) * 0.045
    basis_i = torch.randn(d, 20, dtype=torch.float64, generator=g) * 0.055
    true_t = F.normalize(0.95 * mu_t + 0.31 * sem, dim=-1)
    true_i = F.normalize(0.94 * mu_i + 0.34 * sem, dim=-1)
    protos = F.normalize(0.95 * mu_p + 0.25 * F.normalize(sem + 0.6 * torch.randn(8, d, dtype=torch.float64, generator=g), dim=-1), dim=-1).float()

    def _draw(cls_list: list[int], n_per: int, tw: str) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for c in cls_list:
            ctr = true_t[c] if tw == "text" else true_i[c]
            bas = basis_t if tw == "text" else basis_i
            z = torch.randn(n_per, bas.shape[1], dtype=torch.float64, generator=g) @ bas.T + 0.005 * torch.randn(n_per, d, dtype=torch.float64, generator=g)
            xs.append(l2_normalize((ctr.unsqueeze(0) + z).float()))
            ys.append(torch.full((n_per,), c, dtype=torch.long))
        return torch.cat(xs, 0), torch.cat(ys, 0)

    bank = StreamingClassBank()
    for st_cls, paired in [([0, 5], False), ([1, 4], False), ([2, 6], True), ([3, 7], True)]:
        xi, yi = _draw(st_cls, 60, "image")
        xt, yt = _draw(st_cls, 60, "text")
        bank.update_stage(
            u_expert_t=xt if paired else None, u_base_t=xt if paired else None, y_t=yt if paired else None,
            u_expert_i=xi, u_base_i=xi, y_i=yi,
            paired_base_t=xt if paired else None, paired_base_i=xi if paired else None,
        )
    cstats = bank.canonical_stats(classes)
    head = SelfNormalizedBayesTransportHead(bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic").fit_from_sufficient_stats(
        cstats["towers"], None, None, protos, classes, paired_moments=cstats["paired_moments"]
    )
    assert abs(head.diagnostics["towers"]["text"]["alpha_ep"] - 0.5) < 0.51
    assert head.diagnostics["towers"]["image"]["alpha_ep"] == 0.0
    dev_i, dev_y = _draw(classes, 35, "image")
    dev_t, _ = _draw(classes, 35, "text")
    ev = eval_view_suite(head.predict_views(dev_t, dev_i), dev_y, classes, early, late)
    assert ev["paired"]["macro_f1"] >= ev["image"]["macro_f1"] - 0.02
    assert ev["paired"]["macro_f1"] > 0.80 and ev["paired"]["balanced_ece"] < 0.25

    # Full-support unequal dispersion + tight cone test (dc_lora P1 regime)
    counts = [400, 220, 180, 140, 120, 100, 50, 10]
    centroids, scatter, dof = {}, torch.zeros(d, d, dtype=torch.float64), 0
    for c, nc in enumerate(counts):
        disp = 0.09 if c in early else 0.05
        xs = l2_normalize((true_t[c] + disp * torch.randn(nc, d, dtype=torch.float64, generator=g)).float()).double()
        mc = xs.mean(0)
        centroids[c] = mc
        res = xs - mc.unsqueeze(0)
        scatter += res.T @ res
        dof += nc - 1
    full_stats = {
        "text": {"centroids": centroids, "counts": dict(enumerate(counts)), "within_scatter": scatter, "scatter_dof": dof, "total_samples": sum(counts)},
        "image": {"centroids": centroids, "counts": dict(enumerate(counts)), "within_scatter": scatter, "scatter_dof": dof, "total_samples": sum(counts)},
    }
    h_eq = SelfNormalizedBayesTransportHead(bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic").fit_from_sufficient_stats(
        full_stats, None, None, protos, classes
    )
    tight_x = torch.cat([l2_normalize((F.normalize(0.98 * mu_t + 0.19 * sem[c], dim=-1) + 0.03 * torch.randn(25, d, dtype=torch.float64, generator=g)).float()) for c in classes], 0)
    tight_y = torch.arange(8).repeat_interleave(25)
    ev_eq = eval_view_suite(h_eq.predict_views(tight_x, tight_x), tight_y, classes, early, late)["paired"]
    assert ev_eq["macro_f1"] > 0.75 and abs(ev_eq["excess_gini"]) < 0.20

    # Check balanced ECE metric field
    probs = F.softmax(torch.randn(40, 4), dim=-1)
    y_skew = torch.tensor([0] * 30 + [1] * 5 + [2] * 3 + [3] * 2)
    assert 0.0 <= compute_balanced_ece(probs, y_skew, 4) <= 1.0
    m = eval_classification_metrics(torch.log(probs.clamp_min(1e-6)), y_skew, [0, 1, 2, 3])
    assert "balanced_ece" in m


def test_b2_batched_dop_matches_sequential_dop_pair() -> None:
    torch.manual_seed(31)
    deltas = []
    for st in range(4):
        row = {
            "v.q": (torch.randn(24, 4) @ torch.randn(4, 24)) * 0.05 if st != 1 else torch.zeros(24, 24),
            "v.v": (torch.randn(24, 4) @ torch.randn(4, 24)) * 0.05,
            "t.q": torch.zeros(24, 24) if st < 2 else (torch.randn(24, 4) @ torch.randn(4, 24)) * 0.05,
        }
        deltas.append(row)
    keys = ["v.q", "v.v", "t.q"]
    batched = _dop_merge(deltas, keys)
    for k in keys:
        cur = deltas[0][k].float().cpu().clone()
        for nxt in deltas[1:]:
            cur = _dop_merge_pair(cur, nxt[k])
        assert float((batched[k] - cur).abs().max().item()) < 1e-5


def test_b3_multi_altitude_disaster_suite_contracts_and_isolation(tmp_path: Path) -> None:
    # 1. Mock DMD (6 classes x 6 samples)
    dmd_root = tmp_path / "dmd"
    for cls_dir in ("fires", "floods", "natural_landscape", "infrastructural", "human_damage", "non_damage"):
        for i in range(6):
            p = dmd_root / cls_dir / f"{cls_dir}_{i}.jpg"
            _write_rgb(p, (20 * i, 40, 80))
            p.with_suffix(".txt").write_text(f"Field observation {cls_dir} sample {i} debris emergency.", encoding="utf8")

    ds_dmd = load_benchmark_contracts("dmd", dmd_root, dmd_root, "P4", allow_test=False, seed=11)
    assert ds_dmd["allow_test"] is False and ds_dmd["test"] is None and ds_dmd["num_classes"] == 6 and ds_dmd["num_stages"] == 3
    assert ds_dmd["stages"][0]["fit"] and all(r["mask"] == ["image"] for r in ds_dmd["stages"][0]["fit"])
    assert all(len(st["new_classes"]) >= 2 for st in ds_dmd["stages"])

    # 2. Mock DisasterVQA-Incidents (filters out crisismmd/medic overlap + drops SA-1 label leakage!)
    vqa_root = tmp_path / "vqa"
    inc_classes = ("earthquake", "fire", "accidents", "flood", "storm", "landslide")
    ann_rows = [
        {"dataset_source": "crisismmd", "image_id": "leak1", "image_path": "DisasterVQA/CrisisMMD/leak1.jpg", "disaster_type": "fire", "crisis_info_code": "SA-2", "question": "q?", "groundtruth_answer": ["Yes"], "choices": None},
        {"dataset_source": "medic", "image_id": "leak2", "image_path": "DisasterVQA/MEDIC/leak2.jpg", "disaster_type": "flood", "crisis_info_code": "SA-2", "question": "q?", "groundtruth_answer": ["No"], "choices": None},
    ]
    for c_i, dt in enumerate(inc_classes):
        for i in range(6):
            iid = f"inc_{c_i}_{i}"
            rel = f"DisasterVQA/Incidents1M/{iid}.jpg"
            _write_rgb(vqa_root / rel, (30, 30 * c_i % 250, 20 * i))
            ann_rows.append({"dataset_source": "incidents1m", "image_id": iid, "image_path": rel, "disaster_type": dt, "crisis_info_code": "SA-1", "question": "What type?", "groundtruth_answer": ["A"], "choices": {"A": dt}})
            ann_rows.append({"dataset_source": "incidents1m", "image_id": iid, "image_path": rel, "disaster_type": dt, "crisis_info_code": "SA-2", "question": "Visible structural damage? Select all that apply.", "groundtruth_answer": ["A"], "choices": {"A": f"Heavy rubble scene {iid}"}})
            ann_rows.append({"dataset_source": "incidents1m", "image_id": iid, "image_path": rel, "disaster_type": dt, "crisis_info_code": "OT-1", "question": f"Emergency responders in zone {iid}? Answer with Yes or No.", "groundtruth_answer": "Yes", "choices": None})
    vqa_root.mkdir(parents=True, exist_ok=True)
    (vqa_root / "disastervqa_annotations.jsonl").write_text("\n".join(json.dumps(r) for r in ann_rows), encoding="utf8")

    ds_vqa = load_benchmark_contracts("incidents_vqa", vqa_root, vqa_root, "P3", allow_test=False, seed=11)
    assert all(v["uid"].startswith("incvqa:inc_") for v in ds_vqa["lookup"].values())
    assert all("What type?" not in v["text"] for v in ds_vqa["lookup"].values())
    assert all("Y, e, s" not in v["text"] and "Yes" in v["text"] for v in ds_vqa["lookup"].values())
    assert all("Select all that apply." not in v["text"] and "Answer with Yes or No." not in v["text"] for v in ds_vqa["lookup"].values())
    assert all(r["mask"] == ["text"] for r in ds_vqa["stages"][0]["fit"])

    # 3. Mock DisasterM3 (8 classes, strips DISASTER header, groups Optical+SAR by pre_image_path)
    m3_root = tmp_path / "m3" / "DisasterM3_Bench"
    m3_rows = []
    for c_i, haz in enumerate(("earthquake", "wildfire", "tornado", "hurricane", "flooding", "tsunami", "explosion", "volcano")):
        for g_i in range(4):
            pre_rel = f"test_images/{haz}_{g_i}_pre.png"
            for mod_tag in ("post", "sar_post"):
                post_rel = f"test_images/{haz}_{g_i}_{mod_tag}.png"
                _write_rgb(m3_root / post_rel, (10 * c_i, 30 * g_i, 50))
                m3_rows.append({
                    "pre_image_path": pre_rel.replace("/", "\\"),
                    "post_image_path": post_rel.replace("/", "\\"),
                    "task": "Disaster Report",
                    "ground_truth": f"DISASTER: {haz}\nBUILDING: Collapsed roofs in grid_{c_i}_{g_i} from {haz} {mod_tag}.",
                })
                m3_rows.append({
                    "pre_image_path": pre_rel.replace("/", "\\"),
                    "post_image_path": post_rel.replace("/", "\\"),
                    "task": "Disaster Restoration Advice",
                    "ground_truth": f"IMMEDIATE_RECOVERY: Clear blocked roads in grid_{c_i}_{g_i} {mod_tag}.",
                })
    m3_root.mkdir(parents=True, exist_ok=True)
    (m3_root / "benchmark_release.json").write_text(json.dumps(m3_rows), encoding="utf8")
    ds_m3_p4 = load_benchmark_contracts("disaster_m3", m3_root.parent, m3_root.parent, "P4", allow_test=False, seed=11)
    ds_m3_test = load_benchmark_contracts("disaster_m3", m3_root.parent, m3_root.parent, "P4", allow_test=True, seed=11)
    assert ds_m3_p4["num_classes"] == 8 and ds_m3_p4["num_stages"] == 4
    assert ds_m3_p4["test"] is None and len(ds_m3_test["test"]) > 0
    for rec in ds_m3_p4["lookup"].values():
        assert "DISASTER:" not in rec["text"] and "earthquake" not in rec["text"].lower()
    # Optical and SAR pairs sharing a pre_image_path never span splits
    assert_zero_split_overlap({"train": [{"uid": r["uid"], "image": str(ds_m3_p4["lookup"][r["uid"]]["image"])} for st in ds_m3_p4["stages"] for r in st["fit"]], "dev": [{"uid": r["uid"], "image": str(ds_m3_p4["lookup"][r["uid"]]["image"])} for r in ds_m3_p4["dev"]]})


def test_tar_archive_unpacking_traversal_guard_and_wrapper_unwrap(tmp_path: Path) -> None:
    import io
    import tarfile
    import pytest
    from mnarm.data import extract_archive_safe, prepare_disaster_m3, prepare_dmd, prepare_incidents_vqa

    # 1. Verify path traversal rejection
    bad_tar = tmp_path / "evil.tar"
    with tarfile.open(bad_tar, "w") as tf:
        ti = tarfile.TarInfo("../escape.txt")
        ti.size = 3
        tf.addfile(ti, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        extract_archive_safe(bad_tar, tmp_path / "out_bad")

    # 2. Pack DMD (real UCI #456 multimodal/<cls>/{images,text} layout), nested DisasterVQA-Dataset-v3 (with CrisisMMD/MEDIC to strip), and DisasterM3 into .tar
    dmd_tar = tmp_path / "dmd.tar"
    with tarfile.open(dmd_tar, "w") as tf:
        for cls_dir in ("fires", "flood", "damaged_nature", "damaged_infrastructure", "human_damage", "non_damage"):
            for i in range(6):
                img_f = tmp_path / "src_dmd" / "multimodal" / cls_dir / "images" / f"{cls_dir}_{i}.jpg"
                _write_rgb(img_f, (25 * i, 50, 90))
                txt_f = tmp_path / "src_dmd" / "multimodal" / cls_dir / "text" / f"{cls_dir}_{i}.txt"
                txt_f.parent.mkdir(parents=True, exist_ok=True)
                txt_f.write_text(f"DMD tar sample {cls_dir} idx {i}", encoding="utf8")
                tf.add(img_f, arcname=f"./multimodal/{cls_dir}/images/{img_f.name}")
                tf.add(txt_f, arcname=f"./multimodal/{cls_dir}/text/{txt_f.name}")
    dmd_unpacked = prepare_dmd(tmp_path / "dmd_ext", archive_path=dmd_tar)
    assert load_benchmark_contracts("dmd", dmd_unpacked, dmd_unpacked, "P4", allow_test=False)["num_classes"] == 6

    vqa_src = tmp_path / "src_vqa" / "DisasterVQA-Dataset-v3"
    ann_rows = []
    for c_i, dt in enumerate(("earthquake", "fire", "accidents", "flood", "storm", "other")):
        for i in range(6):
            iid = f"tar_inc_{c_i}_{i}"
            rel = f"DisasterVQA/Incidents1M/{iid}.jpg"
            _write_rgb(vqa_src / rel, (40, 20 * c_i, 15 * i))
            ann_rows.append({"dataset_source": "incidents1m", "image_id": iid, "image_path": rel, "disaster_type": dt, "crisis_info_code": "SA-2", "question": f"Debris visible in zone {c_i}_{i}?", "groundtruth_answer": ["Yes"], "choices": None})
    _write_rgb(vqa_src / "DisasterVQA/CrisisMMD/drop_me.jpg", (1, 1, 1))
    (vqa_src / "disastervqa_annotations.jsonl").write_text("\n".join(json.dumps(r) for r in ann_rows), encoding="utf8")
    vqa_tar = tmp_path / "incidents_vqa.tar"
    with tarfile.open(vqa_tar, "w") as tf:
        tf.add(vqa_src, arcname="DisasterVQA-Dataset-v3")
    vqa_unpacked = prepare_incidents_vqa(tmp_path / "vqa_ext", archive_path=vqa_tar)
    assert not (vqa_unpacked / "DisasterVQA-Dataset-v3/DisasterVQA/CrisisMMD").exists()
    assert load_benchmark_contracts("incidents_vqa", vqa_unpacked, vqa_unpacked, "P4", allow_test=False)["num_classes"] == 6

    m3_src = tmp_path / "src_m3" / "DisasterM3_Bench"
    m3_rows = []
    for c_i, haz in enumerate(("earthquake", "wildfire", "tornado", "hurricane", "flood", "tsunami", "explosion", "volcano")):
        for g_i in range(4):
            rel = f"test_images/{haz}_{g_i}.png"
            _write_rgb(m3_src / rel, (12 * c_i, 25 * g_i, 60))
            m3_rows.append({"pre_image_path": f"{haz}_{g_i}_pre.png", "post_image_path": rel, "task": "Disaster Report", "ground_truth": f"DISASTER: {haz}\nDAMAGE: Sector {c_i}_{g_i}."})
    m3_src.mkdir(parents=True, exist_ok=True)
    (m3_src / "benchmark_release.json").write_text(json.dumps(m3_rows), encoding="utf8")
    m3_tar = tmp_path / "disaster_m3.tar"
    with tarfile.open(m3_tar, "w") as tf:
        tf.add(m3_src, arcname="DisasterM3_Bench")
    m3_unpacked = prepare_disaster_m3(tmp_path / "m3_ext", archive_path=m3_tar)
    assert load_benchmark_contracts("disaster_m3", m3_unpacked, m3_unpacked, "P4", allow_test=False)["num_classes"] == 8


def test_contiguous_tensor_bank_palette_and_mgclip_batch_invariance(tmp_path: Path) -> None:
    import warnings
    from mnarm.methods.vlm_cl_sota import _measure_mean_neg_by_mask

    class _MiniProc:
        def __call__(self, text=None, images=None, return_tensors="pt", **_kw):
            if text is not None:
                ids = torch.tensor([[len(t), len(t) % 7] for t in text], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            arrs = [torch.full((3, 16, 16), float(im.getpixel((0, 0))[0]) / 255.0) for im in images]
            return {"pixel_values": torch.stack(arrs, dim=0)}

    class _MiniVLM:
        device = "cpu"
        embed_dim = 8
        logit_scale = 10.0
        tok_kwargs = {}
        _text_fwd_params = {"input_ids", "attention_mask"}
        processor = _MiniProc()
        clip = type("_C", (), {"eval": lambda self: None})()

        def tokenize_texts(self, texts):
            return self.processor(text=texts)

        def encode_text(self, inp):
            x = inp["input_ids"].float()
            return F.normalize(torch.cat([x, x + 1.0, x + 2.0, x + 3.0], dim=-1), dim=-1)

        def encode_image(self, inp):
            m = inp["pixel_values"].float().mean(dim=(2, 3))
            return F.normalize(torch.cat([m, m * 2.0, torch.ones(m.shape[0], 2)], dim=-1), dim=-1)

    lookup = {}
    recs = []
    for i in range(5):
        p = tmp_path / f"img_{i}.png"
        if i == 0:
            pal = Image.new("P", (16, 16), color=i)
            pal.info["transparency"] = 0
            pal.save(p)
        else:
            _write_rgb(p, (20 * i, 40, 80))
        uid = f"u{i}"
        lookup[uid] = {"uid": uid, "label": i % 2, "text": f"text_{i}" + "x" * i, "image": p}
        recs.append({"uid": uid, "label": i % 2, "mask": ["text", "image"]})

    store = PreprocessedTensorStore(lookup, _MiniVLM(), max_cached_images=3, workers=1, gpu_cache=False)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", UserWarning)
        specs_s = store.make_batch_specs(recs, batch_size=2, shuffle=False)
        specs_l = store.make_batch_specs(recs, batch_size=5, shuffle=False)
        for vw, chk in specs_l:
            b = store.collate_chunk(vw, chk)
            assert b["pixel_values"].shape == (5, 3, 16, 16)
            assert b["input_ids"].shape == (5, 2)
        assert not [w for w in ws if "Palette" in str(w.message)]
    assert len(store._img_slot) == 3 and store._img_bank is not None and store._img_bank.shape == (3, 3, 16, 16)

    p_ord = F.normalize(torch.randn(2, 8), dim=-1)
    vlm = _MiniVLM()
    n_small = _measure_mean_neg_by_mask(vlm, specs_s, store, p_ord, {0: 0, 1: 1}, 2, torch.float32, False)
    n_large = _measure_mean_neg_by_mask(vlm, specs_l, store, p_ord, {0: 0, 1: 1}, 2, torch.float32, False)
    assert abs(n_small[("image", "text")] - n_large[("image", "text")]) < 1e-5


def test_truncated_jpeg_and_mixed_mask_base0_pal_alignment(tmp_path: Path) -> None:
    import io
    from mnarm.methods import PALRLSClassifier
    from mnarm.runners.iccv_same_bank import _cached_base0_features
    from mnarm.runners.stage_ops import collect_train_features_by_modality

    dmd_root = tmp_path / "dmd_trunc"
    for cls_dir in ("fires", "flood", "damaged_nature", "damaged_infrastructure", "human_damage", "non_damage"):
        for i in range(6):
            img_p = dmd_root / "multimodal" / cls_dir / "images" / f"{cls_dir}_{i}.jpg"
            txt_p = dmd_root / "multimodal" / cls_dir / "text" / f"{cls_dir}_{i}.txt"
            img_p.parent.mkdir(parents=True, exist_ok=True)
            txt_p.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            Image.new("RGB", (24, 24), color=(15 * i + 20, 60, 100)).save(buf, format="JPEG")
            raw_jpg = buf.getvalue()
            img_p.write_bytes(raw_jpg[:-16] if (cls_dir == "fires" and i == 0) else raw_jpg)
            txt_p.write_text(f"Trunc test {cls_dir} item {i}", encoding="utf8")

    # 0-byte stub must be skipped cleanly without failing split
    bad_p = dmd_root / "multimodal" / "fires" / "images" / "corrupt_stub.jpg"
    bad_p.write_bytes(b"")
    (dmd_root / "multimodal" / "fires" / "text" / "corrupt_stub.txt").write_text("bad stub", encoding="utf8")

    ds_p1 = load_benchmark_contracts("dmd", dmd_root, dmd_root, "P1", allow_test=False, seed=11)
    assert all("corrupt_stub" not in uid for uid in ds_p1["lookup"])
    st1_fit = ds_p1["stages"][0]["fit"]
    assert {tuple(sorted(r["mask"])) for r in st1_fit} == {("text",), ("image", "text")}

    class _MixVLM:
        device = "cpu"
        embed_dim = 8
        logit_scale = 10.0

        def tokenize_texts(self, texts):
            ids = torch.tensor([[len(t), len(t) % 5 + 1] for t in texts], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def processor(self, images, return_tensors="pt", **_kw):
            del return_tensors
            arrs = [torch.full((3, 24, 24), float(im.getpixel((0, 0))[0]) / 255.0) for im in images]
            return {"pixel_values": torch.stack(arrs, dim=0)}

        def restore_base_weights(self): return None
        def load_lora_state(self, _p): return None
        clip = type("_C", (), {"eval": lambda self: None})()

        def encode_text(self, inp):
            x = inp["input_ids"].float()
            return F.normalize(torch.cat([x, x + 1.0, x + 2.0, x + 3.0], dim=-1), dim=-1)

        def encode_image(self, inp):
            m = inp["pixel_values"].float().mean(dim=(2, 3))
            return F.normalize(torch.cat([m, m * 2.0, torch.ones(m.shape[0], 2)], dim=-1), dim=-1)

    vlm = _MixVLM()
    store = PreprocessedTensorStore(ds_p1["lookup"], vlm, max_cached_images=100, workers=1, gpu_cache=False)
    cache: dict[str, dict[str, torch.Tensor]] = {"text": {}, "image": {}}
    tf_direct = collect_train_features_by_modality(vlm=vlm, kind="lora", payload={}, records=st1_fit, tensor_store=store, batch_size=64, use_amp=False)
    tf_b0 = _cached_base0_features(vlm, st1_fit, store, eval_bs=64, use_amp=False, cache=cache)
    assert float((tf_b0["text_x"] - tf_direct["text_x"]).abs().max().item()) < 1e-6
    assert float((tf_b0["paired_t"] - tf_direct["paired_t"]).abs().max().item()) < 1e-6

    pal = PALRLSClassifier(embed_dim=8, num_classes=6, lam=1.0, seed=11)
    for m_views, (m_t, m_i, m_y) in tf_b0["by_mask"].items():
        pal.update(m_t, m_i, m_y, m_views)
    assert pal._dirty and pal.mod_count["text"] == len(st1_fit)
