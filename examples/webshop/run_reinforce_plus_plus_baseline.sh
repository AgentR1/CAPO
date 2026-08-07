#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-webshop_reinforce_plus_plus_baseline}"
export WEBSHOP_VAL_DUMP_DIR="${WEBSHOP_VAL_DUMP_DIR:-$ROOT_DIR/outputs/webshop_validation/$EXP_NAME}"

exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=mse \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    "$@"
