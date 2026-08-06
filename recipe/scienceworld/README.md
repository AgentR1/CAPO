# ScienceWorld in StepPO

This recipe gives the policy official state-conditioned action templates and visible object referents, with a
30-agent-action horizon. The simulator itself uses a 100-step limit, so the 30-step agent-flow cutoff is not treated
as an environment terminal state. The full valid-command list is used only to label whether a proposed command matches
its canonical form; the parser accepts additional valid surface forms such as `go kitchen`. The list is not exposed in
the prompt and does not prevent the command from reaching the environment. The prompt keeps an
all-action ledger plus the most recent two observation/action pairs.

## Environment and data preparation

```bash
/data/wdy/.conda/envs/steppo/bin/python -m pip install -e /data/wdy/ScienceWorld/official-ScienceWorld
conda install -n steppo -c conda-forge openjdk
export SCIENCEWORLD_HOME=/data/wdy/ScienceWorld/official-ScienceWorld
export PATH=/data/wdy/.conda/envs/steppo/bin:$PATH
python recipe/scienceworld/prepare_scienceworld_arft.py --output-dir data/scienceworld
```

The manifests contain 3,592 train, 1,819 test, and a deterministic task-stratified 100-sample
`test_in_train` subset drawn from test. No dev manifest is produced or used. Training monitors only `test_in_train`; full `test` is reserved for a
separate checkpoint evaluation. Task descriptions are resolved again from the simulator at rollout time, so no Java
process is started per manifest row.

Run full-test evaluation only after saving a checkpoint (the Slurm entry uses the selected H100 variant):

```bash
sbatch --export=ALL,SCIENCEWORLD_VARIANT=h100,SCIENCEWORLD_ALGO=eval_test,SCIENCEWORLD_CHECKPOINT_PATH=/absolute/path/global_step_N slurm/scienceworld/submit_scienceworld_eval_test.slurm
```

## Launch

```bash
bash examples/scienceworld/run_ppo.sh
SCIENCEWORLD_VARIANT=h100_1_7B sbatch slurm/scienceworld/submit_scienceworld_grpo.slurm
```

Training uses an EPO-style terminal reward: a terminal episode with official `final_score > 0` receives `+10.0` on
its final action; every other action receives `0.0`. Official score changes (including a hard failure's `-100`) are
retained as diagnostics, and evaluation reports terminal positive-score success separately from official final score.
