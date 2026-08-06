#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="${SCIENCEWORLD_CHECKPOINT_PATH:?Set SCIENCEWORLD_CHECKPOINT_PATH to a saved global_step_* directory.}"
export EXP_NAME="${SCIENCEWORLD_EVAL_EXP_NAME:-scienceworld_test_$(basename "$CHECKPOINT_PATH")}"

exec bash "$SCRIPT_DIR/run_grpo.sh" \
    data.train_files="${SCIENCEWORLD_DATA_ROOT:-$SCRIPT_DIR/../../data/scienceworld}/test.parquet" \
    data.val_files="${SCIENCEWORLD_DATA_ROOT:-$SCRIPT_DIR/../../data/scienceworld}/test.parquet" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CHECKPOINT_PATH" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"
