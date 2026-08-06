"""ARFT-owned policy-loss extensions registered at worker startup."""

from __future__ import annotations

import numpy as np
import torch

from arft.ratio_diagnostics import get_ratio_diagnostics_writer
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo, register_policy_loss

_global_step: int | np.ndarray | torch.Tensor | None = None


def set_global_step(value: int | np.ndarray | torch.Tensor | None) -> None:
    global _global_step
    _global_step = value


@register_policy_loss("gspo_ratio_diag")
def compute_policy_loss_gspo_ratio_diag(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
    global_step: int | np.ndarray | torch.Tensor | None = None,
):
    """Exact GSPO loss plus local sufficient statistics for length-scaling tests.

    The returned loss is delegated to upstream GSPO unchanged.  Diagnostics use
    the same pre-clamp token log-ratio used by GSPO, with padding/tool tokens
    excluded by ``response_mask``.
    """
    pg_loss, pg_metrics = compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    writer = get_ratio_diagnostics_writer()
    if writer is not None:
        global_step = _global_step if global_step is None else global_step
        if global_step is None:
            raise RuntimeError("gspo_ratio_diag requires the per-row global_step supplied by the ARFT trainer.")
        with torch.no_grad():
            mask = response_mask.to(dtype=log_prob.dtype)
            lengths = mask.sum(dim=-1).clamp(min=1)
            log_ratio_sums = ((log_prob - old_log_prob) * mask).sum(dim=-1)
            # Step GAE broadcasts one step advantage to all valid action tokens.
            step_advantages = (advantages * mask).sum(dim=-1) / lengths
            writer.record(lengths, log_ratio_sums, step_advantages, global_step)

    return pg_loss, pg_metrics
