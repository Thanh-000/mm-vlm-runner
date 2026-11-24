"""In-memory fp16 image and tokenized text store for reproducible batching."""

from __future__ import annotations

import concurrent.futures
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from mnarm.core.determinism import make_generator


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
                    im = raw_im.convert("RGB")
            item["pixel_values"] = self.vlm.processor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)
        item["label"] = torch.tensor(int(rec["label"]), dtype=torch.long)
        item["sample_id"] = uid
        return item


class PreprocessedTensorStore:
    """Caches tokenized text and fp16 preprocessed images in CPU RAM for fast deterministic reuse."""

    def __init__(self, row_lookup: dict[str, dict[str, Any]], vlm: Any, max_cached_images: int = 25000, workers: int = 4):
        self.row_lookup = row_lookup
        self.vlm = vlm
        self.max_cached_images = int(max_cached_images)
        self.workers = max(1, int(workers))
        self._img_cache: dict[str, torch.Tensor] = {}
        self._txt_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.decode_count = 0

    def _load_pil(self, img_ref: Any) -> Image.Image:
        if isinstance(img_ref, Image.Image):
            return img_ref.convert("RGB")
        with Image.open(Path(img_ref)) as im:
            return im.convert("RGB")

    def _decode_pil_chunk(self, uids: list[str]) -> torch.Tensor:
        if not uids:
            return torch.empty((0, 3, 224, 224), dtype=torch.float16)
        refs = [self.row_lookup[u]["image"] for u in uids]
        if self.workers > 1 and len(refs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(refs))) as pool:
                pils = list(pool.map(self._load_pil, refs))
        else:
            pils = [self._load_pil(rf) for rf in refs]
        parts = []
        for off in range(0, len(pils), 64):
            pv = self.vlm.processor(images=pils[off : off + 64], return_tensors="pt")["pixel_values"]
            parts.append(pv.to(torch.float16).cpu())
        self.decode_count += len(uids)
        return torch.cat(parts, dim=0)

    def ensure_text(self, uids: list[str]) -> None:
        missing = list(dict.fromkeys(u for u in uids if u not in self._txt_cache))
        if not missing:
            return
        texts = [str(self.row_lookup[u]["text"]) for u in missing]
        tok = self.vlm.tokenize_texts(texts)
        keys = list(tok.keys())
        for idx, uid in enumerate(missing):
            self._txt_cache[uid] = {k: tok[k][idx].cpu() for k in keys}

    def ensure_images(self, uids: list[str]) -> None:
        rem = max(0, self.max_cached_images - len(self._img_cache))
        if rem == 0:
            return
        missing = list(dict.fromkeys(u for u in uids if u not in self._img_cache))[:rem]
        for off in range(0, len(missing), 128):
            sub = missing[off : off + 128]
            tens = self._decode_pil_chunk(sub)
            for idx, uid in enumerate(sub):
                self._img_cache[uid] = tens[idx]

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
            t_keys = list(self._txt_cache[uids[0]].keys())
            for k in t_keys:
                item[k] = torch.stack([self._txt_cache[u][k] for u in uids], dim=0)
        if "image" in views and uids:
            uncached = list(dict.fromkeys(u for u in uids if u not in self._img_cache))
            dec_map = {}
            if uncached:
                dec_t = self._decode_pil_chunk(uncached)
                dec_map = {u: dec_t[i] for i, u in enumerate(uncached)}
            item["pixel_values"] = torch.stack(
                [self._img_cache[u] if u in self._img_cache else dec_map[u] for u in uids],
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
