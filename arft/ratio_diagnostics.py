"""Local, loss-forward diagnostics for step-level policy-ratio experiments.

The writer intentionally stores only per-step sufficient statistics rather than
token log-probabilities.  This keeps the diagnostic independent from SwanLab
and makes it cheap enough to enable during a normal PPO run.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import numpy as np
import torch


class RatioDiagnosticsWriter:
    """Buffer rank-local ``(global_step, L_t, S_t)`` observations and write NPZ shards."""

    def __init__(self, output_dir: str, flush_every: int = 16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        self.pid = os.getpid()
        self.flush_every = flush_every
        self.call_index = 0
        self.chunk_index = 0
        self._records: list[dict[str, np.ndarray]] = []
        atexit.register(self.flush)

    def record(
        self,
        lengths: torch.Tensor,
        log_ratio_sums: torch.Tensor,
        advantages: torch.Tensor,
        global_step: int | np.ndarray | torch.Tensor,
    ) -> None:
        """Record one policy-loss forward, with all tensors already masked/reduced."""
        n = lengths.numel()
        if isinstance(global_step, torch.Tensor):
            global_steps = global_step.detach().to("cpu", dtype=torch.int64).numpy().reshape(-1)
        else:
            global_steps = np.asarray(global_step, dtype=np.int64).reshape(-1)
        if global_steps.size == 1:
            global_steps = np.full(n, global_steps.item(), dtype=np.int64)
        elif global_steps.size != n:
            raise ValueError(f"Expected one global step or {n} per-row values, got {global_steps.size}.")
        self._records.append(
            {
                "call_index": np.full(n, self.call_index, dtype=np.int32),
                "global_step": global_steps,
                "length": lengths.detach().to("cpu", dtype=torch.int32).numpy(),
                "log_ratio_sum": log_ratio_sums.detach().to("cpu", dtype=torch.float32).numpy(),
                "step_advantage": advantages.detach().to("cpu", dtype=torch.float32).numpy(),
            }
        )
        self.call_index += 1
        if len(self._records) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        arrays = {key: np.concatenate([record[key] for record in self._records]) for key in self._records[0]}
        output_path = self.output_dir / f"rank={self.rank:04d}_pid={self.pid}_chunk={self.chunk_index:06d}.npz"
        np.savez_compressed(output_path, **arrays)
        self.chunk_index += 1
        self._records.clear()


_writer: RatioDiagnosticsWriter | None = None


def get_ratio_diagnostics_writer() -> RatioDiagnosticsWriter | None:
    """Return the process-local writer when diagnostics are explicitly enabled."""
    global _writer
    if os.environ.get("ARFT_RATIO_DIAGNOSTICS_ENABLE", "0") != "1":
        return None
    if _writer is None:
        output_dir = os.environ.get("ARFT_RATIO_DIAGNOSTICS_DIR")
        if not output_dir:
            raise RuntimeError(
                "ARFT_RATIO_DIAGNOSTICS_ENABLE=1 requires an absolute ARFT_RATIO_DIAGNOSTICS_DIR."
            )
        _writer = RatioDiagnosticsWriter(output_dir=output_dir)
    return _writer
