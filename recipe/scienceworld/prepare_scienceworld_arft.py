#!/usr/bin/env python3
"""Materialize ScienceWorld's official variation splits as StepPO parquet manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.scienceworld.utils import ScienceWorldToolExecutor, scienceworld_home  # noqa: E402


def _metadata_by_task() -> dict[str, dict]:
    tasks_path = scienceworld_home() / "scienceworld" / "tasks.json"
    return {item["task_name"]: item for item in json.loads(tasks_path.read_text())}


def _row(task_name: str, variation_idx: int, split: str, metadata: dict, task_description: str) -> dict:
    return {
        "data_source": f"scienceworld_{split}_template_actions",
        "prompt": [{"role": "user", "content": task_description}],
        "reward_model": {"ground_truth": {"task_name": task_name, "variation_idx": variation_idx}, "style": "rule"},
        "extra_info": {
            "index": f"{task_name}:{variation_idx}",
            "task_name": task_name,
            "variation_idx": variation_idx,
            "task_id": metadata.get("task_id", ""),
            "topic": metadata.get("topic", ""),
            "task_description": task_description,
            "split": split,
            "action_space": "templates_and_visible_objects",
        },
    }


def _select_test_in_train(test_rows: list[dict], size: int) -> list[dict]:
    """Select a deterministic, task-stratified monitoring subset from official test."""
    by_task: dict[str, list[dict]] = {}
    for row in test_rows:
        by_task.setdefault(str(row["extra_info"]["task_name"]), []).append(row)
    task_names = sorted(by_task)
    if size < len(task_names):
        raise ValueError(f"test-in-train size ({size}) must cover all {len(task_names)} tasks")

    base, remainder = divmod(size, len(task_names))
    selected: list[dict] = []
    for index, task_name in enumerate(task_names):
        target = base + int(index < remainder)
        ranked = sorted(
            by_task[task_name],
            key=lambda row: hashlib.sha256(str(row["extra_info"]["index"]).encode()).hexdigest(),
        )
        selected.extend(ranked[:target])

    if len(selected) != size:
        raise RuntimeError(f"expected {size} test-in-train rows, selected {len(selected)}")
    return [
        {
            **row,
            "data_source": "scienceworld_test_in_train_template_actions",
            "extra_info": {**row["extra_info"], "split": "test_in_train"},
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/scienceworld"))
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--test-in-train-size", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_by_task()
    executor = ScienceWorldToolExecutor(args.max_episode_steps)
    rows: dict[str, list[dict]] = {"train": [], "test": []}
    try:
        for task_name in executor.env.get_task_names():
            executor.env.load(task_name, 0, "")
            split_indices = {
                "train": executor.env.get_variations_train(),
                "test": executor.env.get_variations_test(),
            }
            for split, variations in split_indices.items():
                for variation_idx in variations:
                    rows[split].append(
                        _row(
                            task_name,
                            int(variation_idx),
                            split,
                            metadata[task_name],
                            str(metadata[task_name].get("task", task_name)),
                        )
                    )
    finally:
        executor.close()
    rows["test_in_train"] = _select_test_in_train(rows["test"], args.test_in_train_size)
    summary = {"max_episode_steps": args.max_episode_steps, "splits": {}}
    for split, split_rows in rows.items():
        pd.DataFrame(split_rows).to_parquet(args.output_dir / f"{split}.parquet", index=False)
        summary["splits"][split] = len(split_rows)
    (args.output_dir / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
