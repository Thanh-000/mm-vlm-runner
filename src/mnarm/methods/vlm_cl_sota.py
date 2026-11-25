"""2025 VLM-CL and Analytic MMCIL SOTA baselines (MG-CLIP ICCV'25, PAL-RLS arXiv:2501.09352)."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mnarm.data.tensor_store import PreprocessedTensorStore
from mnarm.models.vlm import MultimodalVLM, l2_normalize


class PALRLSClassifier:
    """Dual-encoder PAL-RLS adaptation (arXiv:2501.09352 Sec. 3.4.2 Eq. 8-12): random ReLU expansion + RLS."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        lam: float = 1.0,
        expand_dim: int | None = None,
        seed: int = 11,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.lam = float(lam)
        self.expand_dim = int(expand_dim) if expand_dim is not None else min(max(4 * self.embed_dim, 64), 2048)
        gen = torch.Generator().manual_seed(int(seed))
        self.w_up = torch.randn(2 * self.embed_dim, self.expand_dim, dtype=torch.float64, generator=gen) / math.sqrt(float(2 * self.embed_dim))
        self.gram = torch.zeros((self.expand_dim, self.expand_dim), dtype=torch.float64)
        self.rhs = torch.zeros((self.expand_dim, num_classes), dtype=torch.float64)
        self.mod_sum = {"text": torch.zeros(self.embed_dim, dtype=torch.float64), "image": torch.zeros(self.embed_dim, dtype=torch.float64)}
        self.mod_count = {"text": 0, "image": 0}
        self.weights: torch.Tensor | None = None
        self._dirty = False

    def _fill_pair(self, u_t: torch.Tensor | None, u_i: torch.Tensor | None, mask: tuple[str, ...]) -> torch.Tensor:
        n = int(u_t.shape[0]) if u_t is not None else int(u_i.shape[0])  # type: ignore[union-attr]
        if "text" in mask and u_t is not None:
            xt = F.normalize(u_t.float().cpu(), dim=-1)
        else:
            ft = F.normalize(self.mod_sum["text"].float(), dim=-1) if self.mod_count["text"] > 0 else torch.zeros(self.embed_dim)
            xt = ft.unsqueeze(0).expand(n, -1)
        if "image" in mask and u_i is not None:
            xi = F.normalize(u_i.float().cpu(), dim=-1)
        else:
            fi = F.normalize(self.mod_sum["image"].float(), dim=-1) if self.mod_count["image"] > 0 else torch.zeros(self.embed_dim)
            xi = fi.unsqueeze(0).expand(n, -1)
        return torch.cat([xt, xi], dim=-1).to(torch.float64)

    def _expand(self, x_pair: torch.Tensor) -> torch.Tensor:
        return F.relu(x_pair.to(torch.float64) @ self.w_up)

    def update(self, u_t: torch.Tensor | None, u_i: torch.Tensor | None, labels: torch.Tensor, mask: tuple[str, ...]) -> None:
        if "text" in mask and u_t is not None and u_t.shape[0] > 0:
            ut_n = F.normalize(u_t.float().cpu(), dim=-1).to(torch.float64)
            self.mod_sum["text"] += ut_n.sum(dim=0)
            self.mod_count["text"] += int(ut_n.shape[0])
        if "image" in mask and u_i is not None and u_i.shape[0] > 0:
            ui_n = F.normalize(u_i.float().cpu(), dim=-1).to(torch.float64)
            self.mod_sum["image"] += ui_n.sum(dim=0)
            self.mod_count["image"] += int(ui_n.shape[0])
        if (u_t is None or u_t.shape[0] == 0) and (u_i is None or u_i.shape[0] == 0):
            return
        h = self._expand(self._fill_pair(u_t, u_i, tuple(mask)))
        y_oh = F.one_hot(labels.detach().cpu().long(), num_classes=self.num_classes).to(torch.float64)
        self.gram += h.T @ h
        self.rhs += h.T @ y_oh
        self._dirty = True

    def _ensure_solved(self) -> None:
        if self._dirty:
            eye = torch.eye(self.expand_dim, dtype=torch.float64)
            self.weights = torch.linalg.solve(self.gram + self.lam * eye, self.rhs).float()
            self._dirty = False

    def predict_batch(self, u_t: torch.Tensor | None, u_i: torch.Tensor | None, mask: tuple[str, ...], seen_classes: list[int]) -> torch.Tensor:
        self._ensure_solved()
        if self.weights is None:
            n = int(u_t.shape[0]) if u_t is not None else int(u_i.shape[0])  # type: ignore[union-attr]
            return torch.zeros((n, len(seen_classes)), dtype=torch.float32)
        w_seen = self.weights[:, [int(c) for c in seen_classes]]
        return self._expand(self._fill_pair(u_t, u_i, tuple(mask))).float() @ w_seen

    def predict_views(self, u_t: torch.Tensor, u_i: torch.Tensor, seen_classes: list[int]) -> dict[str, torch.Tensor]:
        self._ensure_solved()
        if self.weights is None:
            z = torch.zeros((int(u_t.shape[0]), len(seen_classes)), dtype=torch.float32)
            return {"text": z, "image": z, "paired": z}
        return {
            "text": self.predict_batch(u_t, None, ("text",), seen_classes),
            "image": self.predict_batch(None, u_i, ("image",), seen_classes),
            "paired": self.predict_batch(u_t, u_i, ("text", "image"), seen_classes),
        }


class IntraModalCosineClassifier(nn.Module):
    """Verbatim MG-CLIP VisionClassifier: unscaled normalized linear cosine similarity."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, embed_dim, dtype=torch.float32))
        self.initialized = torch.zeros(num_classes, dtype=torch.bool)

    def forward(self, u: torch.Tensor, classes: list[int]) -> torch.Tensor:
        return F.linear(F.normalize(u.float(), p=2, dim=-1), F.normalize(self.weight[classes], p=2, dim=-1))


def _intra_cls_loss(logits: torch.Tensor, targets_local: torch.Tensor, old_count: int) -> torch.Tensor:
    if old_count <= 0 or old_count >= logits.shape[1]:
        return F.cross_entropy(logits, targets_local)
    new_mask = targets_local >= old_count
    if not bool(new_mask.any()):
        return F.cross_entropy(logits, targets_local)
    return F.cross_entropy(logits[new_mask, old_count:], targets_local[new_mask] - old_count)


def _mean_proto_logits(feats_by_mod: dict[str, torch.Tensor], p_ord: torch.Tensor, scale: float) -> torch.Tensor:
    normed = [l2_normalize(v.float()) for v in feats_by_mod.values()]
    return float(scale) * (torch.stack(normed, dim=0).mean(dim=0) @ p_ord.to(normed[0].device).T)


@torch.inference_mode()
def _measure_mean_neg_by_mask(
    vlm: MultimodalVLM, specs: list[tuple[tuple[str, ...], list[dict[str, Any]]]], tensor_store: PreprocessedTensorStore,
    p_ord: torch.Tensor, cls_map: dict[int, int], n_cls: int, amp_dtype: torch.dtype, amp_on: bool,
) -> dict[tuple[str, ...], float]:
    vlm.clip.eval()
    negs: dict[tuple[str, ...], list[torch.Tensor]] = defaultdict(list)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        for views, chunk in specs:
            batch = tensor_store.collate_chunk(views, chunk)
            dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in batch.items() if k not in ("label", "sample_id")}
            feats = {m: (vlm.encode_text(dev_inp) if m == "text" else vlm.encode_image(dev_inp)) for m in views}
            z = _mean_proto_logits(feats, p_ord, vlm.logit_scale)
            y_loc = torch.tensor([cls_map[int(v)] for v in batch["label"].tolist()], dtype=torch.long, device=vlm.device)
            neg_mask = 1.0 - F.one_hot(y_loc, num_classes=n_cls).float()
            negs[views].append(((z * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1.0)).mean().detach().cpu())
    return {vw: float(torch.stack(lst).mean().item()) for vw, lst in negs.items()}


def train_mg_clip_stage(
    *,
    vlm: MultimodalVLM, prev_state: dict[str, torch.Tensor] | None, mg_heads: dict[str, IntraModalCosineClassifier],
    records: list[dict[str, Any]], protos_full: torch.Tensor, old_classes: list[int], new_classes: list[int],
    stage_idx: int, epochs: int, effective_batch_size: int, micro_batch_size: int, lr: float, weight_decay: float,
    seed: int, use_amp: bool, tensor_store: PreprocessedTensorStore,
    gap_alpha: float = 0.10, mgc_epochs: int = 3, mgc_lr: float = 5e-4, mgc_bs: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, IntraModalCosineClassifier]]:
    """Train MG-CLIP (linlany/MindtheGap@21fb1da0): epoch-level MGP early-stop + MGC cosine heads."""
    vlm.restore_base_weights()
    if stage_idx == 1 or prev_state is None:
        vlm.reset_lora()
    else:
        vlm.load_lora_state(prev_state)
    if not mg_heads:
        c_num = int(protos_full.shape[0])
        mg_heads["text"] = IntraModalCosineClassifier(vlm.embed_dim, c_num).to(vlm.device)
        mg_heads["image"] = IntraModalCosineClassifier(vlm.embed_dim, c_num).to(vlm.device)

    old_ord, new_ord = [int(c) for c in old_classes], [int(c) for c in new_classes]
    cls_ord = old_ord + new_ord
    cls_map = {c: i for i, c in enumerate(cls_ord)}
    p_ord = l2_normalize(protos_full[cls_ord].float())
    eff_bs = max(1, int(effective_batch_size))
    micro_bs = min(eff_bs, max(1, int(micro_batch_size)))
    accum_steps = max(1, eff_bs // micro_bs)
    specs = tensor_store.make_batch_specs(records, micro_bs, shuffle=False, seed=seed + stage_idx * 23)
    amp_on = bool(use_amp and vlm.device == "cuda")
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_on and amp_dtype == torch.float16))

    saved_state = vlm.lora_state()
    vlm.zero_lora()
    neg0 = _measure_mean_neg_by_mask(vlm, specs, tensor_store, p_ord, cls_map, len(cls_ord), amp_dtype, amp_on)
    vlm.load_lora_state(saved_state)

    opt = torch.optim.Adam(vlm.lora_parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(lr) * 0.01)
    rng = random.Random(int(seed) + stage_idx * 23)
    train_specs = list(specs)
    last_valid = copy.deepcopy(saved_state)
    for ep in range(int(epochs)):
        vlm.clip.train()
        rng.shuffle(train_specs)
        for gs in range(0, len(train_specs), accum_steps):
            group = train_specs[gs : gs + accum_steps]
            opt.zero_grad()
            denom = float(len(group))
            for views, chunk in group:
                batch = tensor_store.collate_chunk(views, chunk)
                dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in batch.items() if k not in ("label", "sample_id")}
                y_loc = torch.tensor([cls_map[int(v)] for v in batch["label"].tolist()], dtype=torch.long, device=vlm.device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    feats = {m: (vlm.encode_text(dev_inp) if m == "text" else vlm.encode_image(dev_inp)) for m in views}
                    z = _mean_proto_logits(feats, p_ord, vlm.logit_scale)
                    loss_c = F.cross_entropy(z, y_loc) if (stage_idx == 1 or not old_ord) else _intra_cls_loss(z, y_loc, len(old_ord))
                    loss = loss_c / denom
                scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        neg_ep = _measure_mean_neg_by_mask(vlm, specs, tensor_store, p_ord, cls_map, len(cls_ord), amp_dtype, amp_on)
        rel_drift = max((abs(neg_ep[vw] - neg0[vw]) / max(abs(neg0[vw]), 1e-6) for vw in neg_ep), default=0.0)
        if ep == 0:
            last_valid = vlm.lora_state()
            if rel_drift > float(gap_alpha):
                break
        elif rel_drift > float(gap_alpha):
            vlm.load_lora_state(last_valid)
            break
        else:
            last_valid = vlm.lora_state()

    vlm.clip.eval()
    trained_state = vlm.lora_state()
    feat_bank: dict[str, dict[int, list[torch.Tensor]]] = {"text": defaultdict(list), "image": defaultdict(list)}
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
        for views, chunk in specs:
            batch = tensor_store.collate_chunk(views, chunk)
            dev_inp = {k: v.to(vlm.device, non_blocking=True) for k, v in batch.items() if k not in ("label", "sample_id")}
            ys = batch["label"].tolist()
            for tw in views:
                u = l2_normalize(vlm.encode_text(dev_inp) if tw == "text" else vlm.encode_image(dev_inp)).cpu()
                for row_i, yv in enumerate(ys):
                    feat_bank[tw][int(yv)].append(u[row_i])

    for tw in ("text", "image"):
        clf = mg_heads[tw]
        obs_tw = sorted(c for c in new_ord if feat_bank[tw][c])
        if not obs_tw:
            continue
        with torch.no_grad():
            for c in obs_tw:
                clf.weight.data[c].copy_(torch.stack(feat_bank[tw][c]).mean(dim=0).to(clf.weight.device))
                clf.initialized[c] = True
        x_all = torch.stack([v for c in obs_tw for v in feat_bank[tw][c]], dim=0).to(vlm.device)
        y_all = torch.tensor([cls_map[c] for c in obs_tw for _ in feat_bank[tw][c]], dtype=torch.long, device=vlm.device)
        bs_c = max(1, min(int(mgc_bs), int(x_all.shape[0])))
        n_batches = max(1, math.ceil(int(x_all.shape[0]) / bs_c))
        opt_c = torch.optim.Adam([clf.weight], lr=float(mgc_lr))
        sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=max(1, int(mgc_epochs) * n_batches), eta_min=float(mgc_lr) * 0.01)
        g_c = torch.Generator(device="cpu").manual_seed(int(seed) + stage_idx * 31)
        for _ in range(int(mgc_epochs)):
            perm = torch.randperm(int(x_all.shape[0]), generator=g_c).to(vlm.device)
            for start in range(0, int(x_all.shape[0]), bs_c):
                idx = perm[start : start + bs_c]
                opt_c.zero_grad()
                loss_cl = _intra_cls_loss(clf(x_all[idx], cls_ord), y_all[idx], old_count=len(old_ord))
                loss_cl.backward()
                if old_ord and clf.weight.grad is not None:
                    clf.weight.grad[old_ord] = 0.0
                opt_c.step()
                sched_c.step()
    return trained_state, copy.deepcopy(mg_heads)


@torch.inference_mode()
def mg_clip_eval_logits(
    u_text: torch.Tensor | None, u_image: torch.Tensor | None, cross_scores: dict[str, torch.Tensor],
    mg_heads: dict[str, IntraModalCosineClassifier], seen_classes: list[int], logit_scale: float,
    a_weight: float = 1.0, b_weight: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Verbatim MG-CLIP inference: (a * softmax(cross) + b * softmax(unscaled MGC)) / (a + b)."""
    del logit_scale
    seen = [int(c) for c in seen_classes]
    active_intra: dict[str, torch.Tensor] = {}
    for tw, feat in (("text", u_text), ("image", u_image)):
        if feat is None or feat.shape[0] == 0:
            continue
        clf = mg_heads[tw]
        part = clf(feat.to(clf.weight.device), seen).cpu()
        uninit_mask = (~clf.initialized[seen]).cpu()
        if bool(uninit_mask.any()):
            part[:, uninit_mask] = -1e4
        active_intra[tw] = part

    denom = float(a_weight) + float(b_weight)
    out: dict[str, torch.Tensor] = {}
    for vw in ("text", "image"):
        if vw in cross_scores and vw in active_intra and cross_scores[vw].shape[0] == active_intra[vw].shape[0]:
            p_c = F.softmax(cross_scores[vw].float(), dim=-1)
            p_i = F.softmax(active_intra[vw].float(), dim=-1)
            out[vw] = torch.log(((float(a_weight) * p_c + float(b_weight) * p_i) / denom).clamp_min(1e-12))
    if (
        "paired" in cross_scores and "text" in active_intra and "image" in active_intra
        and active_intra["text"].shape[0] == active_intra["image"].shape[0] == cross_scores["paired"].shape[0]
    ):
        p_cp = F.softmax(cross_scores["paired"].float(), dim=-1)
        p_ip = 0.5 * (F.softmax(active_intra["text"].float(), dim=-1) + F.softmax(active_intra["image"].float(), dim=-1))
        out["paired"] = torch.log(((float(a_weight) * p_cp + float(b_weight) * p_ip) / denom).clamp_min(1e-12))
    return out
