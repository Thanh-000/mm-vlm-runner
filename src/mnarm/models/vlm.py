"""CLIP and SigLIP VLM backbone wrappers with LoRA Q/V adaptation and collision-free digests."""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, SiglipModel


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Safely L2-normalize the last dimension in float32."""
    x32 = x.float()
    return x32 / x32.norm(dim=-1, keepdim=True).clamp_min(eps)


def proto_logits(feats: torch.Tensor, protos_full: torch.Tensor, classes: list[int], scale: float) -> torch.Tensor:
    """Compute scaled cosine prototype logits [N, len(classes)] on CPU."""
    cols = torch.tensor([int(c) for c in classes], dtype=torch.long)
    p_seen = l2_normalize(protos_full.detach().cpu().float().index_select(0, cols))
    u = l2_normalize(feats.detach().cpu())
    return (float(scale) * (u @ p_seen.T)).float()


def tower_digest(tower: str, kind: str, payload: dict[str, torch.Tensor]) -> str:
    """Return collision-free blake2b digest identifying a tower's weights (fixes L1-sum collisions)."""
    prefix = "text_model." if tower == "text" else "vision_model."
    if kind == "lora":
        keys = [k for k in sorted(payload) if k.startswith(prefix)]
        active = [k for k in keys if k.endswith(".lora_b") and torch.count_nonzero(payload[k]) > 0]
    elif kind == "delta":
        keys = [k for k in sorted(payload) if k.startswith(prefix)]
        active = [k for k in keys if torch.count_nonzero(payload[k]) > 0]
    else:
        raise ValueError(f"Unknown tower payload kind: {kind}")

    if not active:
        return "base_zero"

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(f"{kind}:{tower}:".encode("ascii"))
    for key in keys:
        t_flat = payload[key].detach().cpu().to(torch.float32).contiguous()
        hasher.update(key.encode("utf8"))
        hasher.update(t_flat.numpy().tobytes())
    return f"{kind}:{hasher.hexdigest()}"


class LoRALinear(nn.Module):
    """Low-rank Q/V linear adapter preserving frozen pretrained weights."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, dtype=base.weight.dtype, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return out + (self.scale * delta).to(out.dtype)


def attach_qv_lora(vlm_model: nn.Module, rank: int = 8, alpha: int = 16) -> dict[str, LoRALinear]:
    """Attach LoRALinear to all self-attention q_proj and v_proj modules in both towers."""
    targets = [
        (name, mod)
        for name, mod in vlm_model.named_modules()
        if isinstance(mod, nn.Linear)
        and (name.startswith("text_model.") or name.startswith("vision_model."))
        and name.endswith((".q_proj", ".v_proj"))
    ]
    replaced: dict[str, LoRALinear] = {}
    for name, mod in targets:
        parent_name, attr = name.rsplit(".", 1)
        parent = vlm_model.get_submodule(parent_name)
        wrapped = LoRALinear(mod, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapped)
        replaced[name] = wrapped
    return replaced


def _unwrap_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Unexpected encoder output type: {type(out).__name__}")


def _load_hf_model(cls: Any, model_path: str, revision: str | None, local_files_only: bool) -> nn.Module:
    kw_rev: dict[str, Any] = {"local_files_only": local_files_only}
    if revision:
        kw_rev["revision"] = revision
    for attn_kw in ({"attn_implementation": "sdpa"}, {"attn_implementation": "eager"}, {}):
        try:
            return cls.from_pretrained(model_path, **kw_rev, **attn_kw).float()
        except (ValueError, TypeError, NotImplementedError, AttributeError, ImportError):
            if not attn_kw:
                raise
    raise RuntimeError(f"Unreachable loader fallthrough for {model_path}")


class MultimodalVLM:
    """Unified wrapper over HuggingFace CLIPModel and SiglipModel with LoRA Q/V adapters."""

    def __init__(
        self,
        backbone_key: str,
        model_path: str,
        device: str = "cuda",
        rank: int = 8,
        alpha: int = 16,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        self.backbone_key = backbone_key
        self.device = device
        self.rank = int(rank)
        self.alpha = int(alpha)

        proc_kw: dict[str, Any] = {"local_files_only": local_files_only}
        if revision:
            proc_kw["revision"] = revision

        if backbone_key == "clip_b16":
            self.clip = _load_hf_model(CLIPModel, model_path, revision, local_files_only)
            self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=False, **proc_kw)
            self.is_siglip = False
            self.tok_kwargs = {"padding": "max_length", "truncation": True, "max_length": 77}
        elif backbone_key == "siglip_b16":
            self.clip = _load_hf_model(SiglipModel, model_path, revision, local_files_only)
            self.processor = AutoProcessor.from_pretrained(model_path, **proc_kw)
            self.is_siglip = True
            self.tok_kwargs = {"padding": "max_length", "truncation": True, "max_length": 64}
        else:
            self.clip = _load_hf_model(AutoModel, model_path, revision, local_files_only)
            self.processor = AutoProcessor.from_pretrained(model_path, **proc_kw)
            self.is_siglip = hasattr(self.clip, "text_model") and not hasattr(self.clip, "text_projection")
            self.tok_kwargs = {"padding": "max_length", "truncation": True, "max_length": 64 if self.is_siglip else 77}

        for param in self.clip.parameters():
            param.requires_grad_(False)
        self.qv_modules = attach_qv_lora(self.clip, rank=self.rank, alpha=self.alpha)
        self.clip.to(device).eval()

        self.base_qv_weights = {n: mod.base.weight.detach().clone() for n, mod in self.qv_modules.items()}
        self._text_fwd_params = set(inspect.signature(self.clip.get_text_features).parameters.keys())
        self.logit_scale = float(self.clip.logit_scale.exp().detach().cpu().item()) if hasattr(self.clip, "logit_scale") else 100.0
        self.embed_dim = (
            int(self.clip.config.projection_dim)
            if hasattr(self.clip.config, "projection_dim")
            else int(self.clip.text_model.config.hidden_size)
        )

    def tokenize_texts(self, texts: list[str], chunk_size: int = 256) -> dict[str, torch.Tensor]:
        parts = [
            self.processor(text=texts[i : i + chunk_size], return_tensors="pt", **self.tok_kwargs)
            for i in range(0, len(texts), chunk_size)
        ]
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0] if k in self._text_fwd_params}

    def encode_text(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        fwd = {k: v for k, v in inputs.items() if k in self._text_fwd_params}
        return _unwrap_tensor(self.clip.get_text_features(**fwd))

    def encode_image(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        pv = inputs["pixel_values"]
        clip_dt = getattr(self.clip, "dtype", torch.float32)
        if pv.dtype != clip_dt:
            pv = pv.to(dtype=clip_dt)
        return _unwrap_tensor(self.clip.get_image_features(pixel_values=pv))

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Encode class text prompts under frozen pretrained base weights theta_0."""
        self.clip.eval()
        self.restore_base_weights()
        saved = self.lora_state()
        try:
            self.zero_lora()
            tok = self.tokenize_texts(list(prompts))
            tok_dev = {k: v.to(self.device) for k, v in tok.items()}
            return F.normalize(self.encode_text(tok_dev).float(), dim=-1).cpu()
        finally:
            self.load_lora_state(saved)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for mod in self.qv_modules.values() for p in (mod.lora_a, mod.lora_b)]

    def lora_state(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, mod in self.qv_modules.items():
            out[f"{name}.lora_a"] = mod.lora_a.detach().cpu().clone()
            out[f"{name}.lora_b"] = mod.lora_b.detach().cpu().clone()
        return out

    def load_lora_state(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            self.zero_lora()
            return
        with torch.no_grad():
            for name, mod in self.qv_modules.items():
                mod.lora_a.data.copy_(state[f"{name}.lora_a"].to(mod.lora_a.device))
                mod.lora_b.data.copy_(state[f"{name}.lora_b"].to(mod.lora_b.device))

    def zero_lora(self) -> None:
        with torch.no_grad():
            for mod in self.qv_modules.values():
                mod.lora_b.zero_()

    def reset_lora(self) -> None:
        self.restore_base_weights()
        with torch.no_grad():
            for mod in self.qv_modules.values():
                nn.init.kaiming_uniform_(mod.lora_a, a=math.sqrt(5))
                mod.lora_b.zero_()

    def restore_base_weights(self) -> None:
        with torch.no_grad():
            for name, mod in self.qv_modules.items():
                mod.base.weight.data.copy_(self.base_qv_weights[name])

    def apply_dense_delta(self, delta: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.zero_lora()
            for name, mod in self.qv_modules.items():
                w0 = self.base_qv_weights[name]
                if name in delta:
                    mod.base.weight.data.copy_(w0 + delta[name].to(w0.device, dtype=w0.dtype))
                else:
                    mod.base.weight.data.copy_(w0)

    def extract_lora_delta(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scale = float(self.alpha) / float(self.rank)
        return {
            name: scale * (state[f"{name}.lora_b"].float() @ state[f"{name}.lora_a"].float()).cpu()
            for name in self.qv_modules
        }

    def tower_digest(self, tower: str, kind: str, payload: dict[str, torch.Tensor]) -> str:
        return tower_digest(tower, kind, payload)
