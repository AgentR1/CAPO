cd ../verl

BASE_DIR=/data/wdy/StepPO/checkpoints/
SRC_ROOT=${BASE_DIR}/HotpotQA_ARFT/hotpotqa_reinforce_plus_plus
DST_ROOT=${BASE_DIR}/Convert_1_7B/hotpotqa_reinforce_plus_plus

for step in $(seq 200 20 200); do
    echo "=== Merging checkpoint at step ${step} ==="

    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir ${SRC_ROOT}/global_step_${step}/actor \
        --target_dir ${DST_ROOT}/actor_${step}
done
