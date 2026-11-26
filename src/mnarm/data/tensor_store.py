"""In-memory fp16 image and tokenized text store for reproducible batching."""

from __future__ import annotations

import concurrent.futures
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from mnarm.core.determinism import make_generator

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MaskedRowsDataset(Dataset):
    """Dataset yielding authorized modality inputs only for a uniform modality mask."""

    def __init__(self, records: list[dict[str, Any]], row_lookup: dict[str, dict[str, Any]], vlm: Any, views: tuple[str, ...]):
        self.records = list(records)
        self.row_lookup = row_lookup
        self.vlm = vlm
        self.views = tuple(views)
        self.tokens: dict[str, torch.Tensor] = {}
        if "text" in self.views and self.records:
            texts = [str(self.row_lookup[r["uid"]]["text"]) for r in self.records]
            self.tokens = self.vlm.tokenize_texts(texts)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        uid = rec["uid"]
        item: dict[str, Any] = {k: v[idx] for k, v in self.tokens.items()}
        if "image" in self.views:
            img_ref = self.row_lookup[uid]["image"]
            if isinstance(img_ref, Image.Image):
                im = img_ref.convert("RGB")
            else:
                with Image.open(img_ref) as raw_im:
                    im = raw_im.convert("RGBA").convert("RGB") if (raw_im.mode == "P" and "transparency" in raw_im.info) else raw_im.convert("RGB")
            item["pixel_values"] = self.vlm.processor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)
        item["label"] = torch.tensor(int(rec["label"]), dtype=torch.long)
        item["sample_id"] = uid
        return item


def _auto_gpu_cache(vlm: Any) -> bool:
    if getattr(vlm, "device", "cpu") != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory >= 36 * (1024 ** 3)


class PreprocessedTensorStore:
    """Caches tokenized text and fp16 images in contiguous RAM/VRAM banks for single-kernel gather."""

    def __init__(
        self,
        row_lookup: dict[str, dict[str, Any]],
        vlm: Any,
        max_cached_images: int = 25000,
        workers: int = 0,
        gpu_cache: bool | None = None,
    ):
        self.row_lookup = row_lookup
        self.vlm = vlm
        self.max_cached_images = int(max_cached_images)
        self.workers = min(6, max(2, (os.cpu_count() or 4) // 2)) if int(workers) <= 0 else max(1, int(workers))
        self.gpu_cache = _auto_gpu_cache(vlm) if gpu_cache is None else bool(gpu_cache and getattr(vlm, "device", "cpu") == "cuda")
        self.cache_device = getattr(vlm, "device", "cpu") if self.gpu_cache else "cpu"
        self._img_slot: dict[str, int] = {}
        self._img_bank: torch.Tensor | None = None
        self._txt_slot: dict[str, int] = {}
        self._txt_bank: dict[str, torch.Tensor] = {}
        self.decode_count = 0

    def _load_pil(self, img_ref: Any) -> Image.Image:
        if isinstance(img_ref, Image.Image):
            return img_ref.convert("RGBA").convert("RGB") if (img_ref.mode == "P" and "transparency" in img_ref.info) else img_ref.convert("RGB")
        with Image.open(Path(img_ref)) as im:
            return im.convert("RGBA").convert("RGB") if (im.mode == "P" and "transparency" in im.info) else im.convert("RGB")

    def _decode_pil_chunk(self, uids: list[str]) -> torch.Tensor:
        if not uids:
            return torch.empty((0, 3, 224, 224), dtype=torch.float16, device=self.cache_device)
        refs = [self.row_lookup[u]["image"] for u in uids]
        if self.workers > 1 and len(refs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(refs))) as pool:
                pils = list(pool.map(self._load_pil, refs))
        else:
            pils = [self._load_pil(rf) for rf in refs]
        parts = []
        proc_bs = 256 if self.gpu_cache else 64
        for off in range(0, len(pils), proc_bs):
            pv = self.vlm.processor(images=pils[off : off + proc_bs], return_tensors="pt")["pixel_values"]
            parts.append(pv.to(device=self.cache_device, dtype=torch.float16))
        self.decode_count += len(uids)
        return torch.cat(parts, dim=0)

    def ensure_text(self, uids: list[str]) -> None:
        missing = list(dict.fromkeys(u for u in uids if u not in self._txt_slot))
        if not missing:
            return
        with torch.inference_mode(False), torch.no_grad():
            tok = self.vlm.tokenize_texts([str(self.row_lookup[u]["text"]) for u in missing])
            cur = len(self._txt_slot)
            need = cur + len(missing)
            for k, val in tok.items():
                t_dev = val.detach().clone().to(self.cache_device)
                if k not in self._txt_bank:
                    cap_t = max(len(self.row_lookup), need, 1)
                    self._txt_bank[k] = torch.empty((cap_t, *t_dev.shape[1:]), dtype=t_dev.dtype, device=self.cache_device)
                elif need > self._txt_bank[k].shape[0]:
                    extra = need - self._txt_bank[k].shape[0]
                    self._txt_bank[k] = torch.cat([self._txt_bank[k], torch.empty((extra, *t_dev.shape[1:]), dtype=t_dev.dtype, device=self.cache_device)], dim=0)
                self._txt_bank[k][cur:need].copy_(t_dev)
            for i, uid in enumerate(missing):
                self._txt_slot[uid] = cur + i

    def ensure_images(self, uids: list[str]) -> None:
        rem = max(0, self.max_cached_images - len(self._img_slot))
        if rem == 0:
            return
        missing = list(dict.fromkeys(u for u in uids if u not in self._img_slot))[:rem]
        if not missing:
            return
        chunk_sz = 256 if self.gpu_cache else 128
        with torch.inference_mode(False), torch.no_grad():
            for off in range(0, len(missing), chunk_sz):
                sub = missing[off : off + chunk_sz]
                tens = self._decode_pil_chunk(sub).detach().clone()
                cur = len(self._img_slot)
                need = cur + len(sub)
                if self._img_bank is None:
                    cap_i = min(max(len(self.row_lookup), need, 1), self.max_cached_images)
                    self._img_bank = torch.empty((cap_i, *tens.shape[1:]), dtype=torch.float16, device=self.cache_device)
                elif need > self._img_bank.shape[0]:
                    extra = need - self._img_bank.shape[0]
                    self._img_bank = torch.cat([self._img_bank, torch.empty((extra, *tens.shape[1:]), dtype=torch.float16, device=self.cache_device)], dim=0)
                self._img_bank[cur:need].copy_(tens)
                for i, uid in enumerate(sub):
                    self._img_slot[uid] = cur + i

    def make_batch_specs(
        self,
        records: list[dict[str, Any]],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 11,
    ) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
        """Group records by modality mask into deterministic batch specifications."""
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        all_t: list[str] = []
        all_i: list[str] = []
        for rec in records:
            views = tuple(sorted(set(rec["mask"])))
            groups[views].append(rec)
            if "text" in views:
                all_t.append(rec["uid"])
            if "image" in views:
                all_i.append(rec["uid"])
        if all_t:
            self.ensure_text(all_t)
        if all_i:
            self.ensure_images(all_i)

        specs: list[tuple[tuple[str, ...], list[dict[str, Any]]]] = []
        rng = random.Random(int(seed))
        bsz = max(1, int(batch_size))
        for views in sorted(groups):
            recs = list(groups[views])
            if shuffle:
                rng.shuffle(recs)
            for off in range(0, len(recs), bsz):
                specs.append((views, recs[off : off + bsz]))
        if shuffle and len(specs) > 1:
            rng_specs = random.Random(int(seed) + 1)
            rng_specs.shuffle(specs)
        return specs

    def collate_chunk(self, views: tuple[str, ...], chunk_recs: list[dict[str, Any]]) -> dict[str, Any]:
        """Materialize a batch tensor dictionary for a uniform modality mask chunk."""
        uids = [r["uid"] for r in chunk_recs]
        item: dict[str, Any] = {
            "label": torch.tensor([int(r["label"]) for r in chunk_recs], dtype=torch.long),
            "sample_id": uids,
        }
        if "text" in views and uids:
            self.ensure_text(uids)
            idx_t = torch.tensor([self._txt_slot[u] for u in uids], dtype=torch.long, device=self.cache_device)
            for k, bank_t in self._txt_bank.items():
                item[k] = bank_t.index_select(0, idx_t)
        if "image" in views and uids:
            uncached = list(dict.fromkeys(u for u in uids if u not in self._img_slot))
            if not uncached and self._img_bank is not None:
                idx_i = torch.tensor([self._img_slot[u] for u in uids], dtype=torch.long, device=self.cache_device)
                item["pixel_values"] = self._img_bank.index_select(0, idx_i)
            else:
                dec_map = {u: t for u, t in zip(uncached, self._decode_pil_chunk(uncached))} if uncached else {}
                item["pixel_values"] = torch.stack(
                    [self._img_bank[self._img_slot[u]] if u in self._img_slot and self._img_bank is not None else dec_map[u] for u in uids],
                    dim=0,
                )
        return item

    def iter_batches(self, records: list[dict[str, Any]], batch_size: int, shuffle: bool = False, seed: int = 11):
        """Yield (views, batch_dict) across all records."""
        for views, chunk in self.make_batch_specs(records, batch_size=batch_size, shuffle=shuffle, seed=seed):
            yield views, self.collate_chunk(views, chunk)


def make_seeded_loader_generator(seed: int) -> torch.Generator:
    """Alias exposing deterministic DataLoader RNG generator."""
    return make_generator(seed)
