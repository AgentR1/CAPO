#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-scienceworld_gspo_${RUN_TIMESTAMP}}"
exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0003 \
    "$@"
