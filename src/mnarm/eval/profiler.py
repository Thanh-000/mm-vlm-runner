"""Wall-time, peak GPU VRAM, and per-sample inference latency profiler."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch


class Profiler:
    """Lightweight CPU/CUDA execution profiler recording wall time and peak VRAM."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    @contextmanager
    def track(self, name: str, n_samples: int | None = None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            else:
                peak_mb = 0.0
            sec = float(time.perf_counter() - t0)
            rec: dict[str, Any] = {"wall_seconds": sec, "peak_vram_mb": peak_mb}
            if n_samples and n_samples > 0:
                rec["n_samples"] = int(n_samples)
                rec["ms_per_sample"] = 1000.0 * sec / float(n_samples)
            self.records[name] = rec
