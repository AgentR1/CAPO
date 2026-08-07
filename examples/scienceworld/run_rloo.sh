#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-scienceworld_rloo}"
exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=rloo \
    "$@"
