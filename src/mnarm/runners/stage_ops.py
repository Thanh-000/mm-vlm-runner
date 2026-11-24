"""Stage training, gradient-accumulated optimization, and collision-free feature extraction."""

from __future__ import annotations

import random
from typing import Any

import torch

from mnarm.data.tensor_store import PreprocessedTensorStore
from mnarm.methods.l_mnar import (
    analyze_co_observation_graph,
    compute_multimodal_lmnar_loss,
    compute_standard_ce_loss,
)
from mnarm.models.vlm import MultimodalVLM


def train_lora_stage(
    *,
    vlm: MultimodalVLM,
    init_lora: dict[str, torch.Tensor] | None,
    records: list[dict[str, Any]],
    protos_full: torch.Tensor,
    seen_classes: list[int],
    objective: str,
    stage_idx: int,
    epochs: int,
    effective_batch_size: int,
    micro_batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    use_amp: bool,
    tensor_store: PreprocessedTensorStore,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train LoRA Q/V adapters with gradient accumulation preserving exact effective-batch steps."""
    if objective not in ("std_ce", "l_mnar"):
        raise ValueError(f"Unsupported objective={objective!r}")

    vlm.restore_base_weights()
    if init_lora is None:
        vlm.reset_lora()
    else:
        vlm.load_lora_state(init_lora)

    graph_info = analyze_co_observation_graph(records, seen_classes)
    eff_bs = max(1, int(effective_batch_size))
    micro_bs = min(eff_bs, max(1, int(micro_batch_size)))
    if eff_bs % micro_bs != 0:
        raise ValueError(f"effective_batch_size ({eff_bs}) must be divisible by micro_batch_size ({micro_bs})")
    accum_steps = eff_bs // micro_bs

    specs = tensor_store.make_batch_specs(records, micro_bs, shuffle=True, seed=seed + stage_idx * 101)
    opt = torch.optim.AdamW(vlm.lora_parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed) + stage_idx * 101)
    amp_on = bool(use_amp and vlm.device == "cuda")
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16

    vlm.clip.train()
    optimizer_steps = 0
    for _ in range(int(epochs)):
        rng.shuffle(specs)
        for group_start in range(0, len(specs), accum_steps):
            group = specs[group_start : group_start + accum_steps]
            opt.zero_grad()
            denom = float(len(group))
            for views, chunk in group:
                batch = tensor_store.collate_chunk(views, chunk)
                dev_inp = {
                    k: v.to(vlm.device, non_blocking=True)
                    for k, v in batch.items()
                    if k not in ("label", "sample_id")
                }
                y_global = batch["label"].to(vlm.device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    feats: dict[str, torch.Tensor] = {}
                    if "text" in views:
                        feats["text"] = vlm.encode_text(dev_inp)
                    if "image" in views:
                        feats["image"] = vlm.encode_image(dev_inp)
                    if objective == "l_mnar":
                        loss = compute_multimodal_lmnar_loss(
                            feats, protos_full, vlm.logit_scale, y_global, seen_classes, graph_info
                        )
                    else:
                        loss = compute_standard_ce_loss(
                            feats, protos_full, vlm.logit_scale, y_global, seen_classes
                        )
                    scaled_loss = loss / denom
                scaled_loss.backward()
            opt.step()
            optimizer_steps += 1

    vlm.clip.eval()
    meta = {
        "objective": objective,
        "effective_batch_size": eff_bs,
        "micro_batch_size": micro_bs,
        "accum_steps": accum_steps,
        "optimizer_steps": optimizer_steps,
        "graph_components": graph_info["components"],
        "has_zero_support_mnar": graph_info["has_zero_support_mnar"],
    }
    return vlm.lora_state(), meta


@torch.inference_mode()
def extract_paired_features_cached(
    *,
    vlm: MultimodalVLM,
    kind: str,
    payload: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    tensor_store: PreprocessedTensorStore,
    batch_size: int,
    cache: dict[tuple[str, str], torch.Tensor],
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode paired evaluation records using collision-free blake2b tower digests."""
    specs = tensor_store.make_batch_specs(records, batch_size=batch_size, shuffle=False)
    labels = torch.tensor([int(r["label"]) for _, recs in specs for r in recs], dtype=torch.long)
    sigs = {tw: vlm.tower_digest(tw, kind, payload) for tw in ("text", "image")}
    if any((tw, sigs[tw]) not in cache for tw in ("text", "image")):
        if kind == "lora":
            vlm.restore_base_weights()
            vlm.load_lora_state(payload)
        else:
            vlm.apply_dense_delta(payload)

    amp_on = bool(use_amp and vlm.device == "cuda")
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    vlm.clip.eval()

    for tw in ("text", "image"):
        ck = (tw, sigs[tw])
        if ck in cache:
            continue
        chunks: list[torch.Tensor] = []
        for views, item in specs:
            if tw not in views:
                continue
            raw_b = tensor_store.collate_chunk(views, item)
            dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in raw_b.items() if k not in ("label", "sample_id")}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                feat = vlm.encode_text(dev_inp) if tw == "text" else vlm.encode_image(dev_inp)
            chunks.append(feat.float().cpu())
        cache[ck] = torch.cat(chunks, dim=0) if chunks else torch.empty((0, vlm.embed_dim))

    return cache[("text", sigs["text"])], cache[("image", sigs["image"])], labels


@torch.inference_mode()
def collect_train_features_by_modality(
    *,
    vlm: MultimodalVLM,
    kind: str,
    payload: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    tensor_store: PreprocessedTensorStore,
    batch_size: int,
    use_amp: bool,
) -> dict[str, torch.Tensor | None]:
    """Extract authorized train-record embeddings separated by text/image/paired observations."""
    if not records:
        return {"text_x": None, "text_y": None, "image_x": None, "image_y": None, "paired_t": None, "paired_i": None}

    if kind == "lora":
        vlm.restore_base_weights()
        vlm.load_lora_state(payload)
    else:
        vlm.apply_dense_delta(payload)

    amp_on = bool(use_amp and vlm.device == "cuda")
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    vlm.clip.eval()

    tx_parts, ty_parts, ix_parts, iy_parts, pt_parts, pi_parts = [], [], [], [], [], []
    for views, batch in tensor_store.iter_batches(records, batch_size=batch_size, shuffle=False):
        dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in batch.items() if k not in ("label", "sample_id")}
        lbl = batch["label"].clone()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
            ut = vlm.encode_text(dev_inp).float().cpu() if "text" in views else None
            ui = vlm.encode_image(dev_inp).float().cpu() if "image" in views else None
        if ut is not None:
            tx_parts.append(ut)
            ty_parts.append(lbl)
        if ui is not None:
            ix_parts.append(ui)
            iy_parts.append(lbl)
        if ut is not None and ui is not None:
            pt_parts.append(ut)
            pi_parts.append(ui)

    return {
        "text_x": torch.cat(tx_parts, dim=0) if tx_parts else None,
        "text_y": torch.cat(ty_parts, dim=0) if ty_parts else None,
        "image_x": torch.cat(ix_parts, dim=0) if ix_parts else None,
        "image_y": torch.cat(iy_parts, dim=0) if iy_parts else None,
        "paired_t": torch.cat(pt_parts, dim=0) if pt_parts else None,
        "paired_i": torch.cat(pi_parts, dim=0) if pi_parts else None,
    }
