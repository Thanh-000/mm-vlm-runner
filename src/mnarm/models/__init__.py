"""Unified CLIP / SigLIP VLM adapters, LoRA Q/V wrappers, and normalized cosine utilities."""

from mnarm.models.vlm import (
    LoRALinear,
    MultimodalVLM,
    attach_qv_lora,
    l2_normalize,
    proto_logits,
    tower_digest,
)

__all__ = [
    "LoRALinear",
    "MultimodalVLM",
    "attach_qv_lora",
    "l2_normalize",
    "proto_logits",
    "tower_digest",
]
