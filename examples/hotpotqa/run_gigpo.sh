#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-hotpotqa_gigpo_${RUN_TIMESTAMP}}"

exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=gigpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    ++algorithm.gigpo.step_advantage_w=1.0 \
    ++algorithm.gigpo.mode=mean_std_norm \
    ++algorithm.gigpo.enable_similarity=False \
    ++algorithm.gigpo.similarity_thresh=0.95 \
    "$@"
