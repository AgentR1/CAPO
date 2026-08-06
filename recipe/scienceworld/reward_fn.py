from __future__ import annotations

from typing import Any

from verl.utils.reward_score import default_compute_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict[str, Any]:
    if not str(data_source).startswith("scienceworld"):
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    runtime = (extra_info or {}).get("reward_extra_info", {})
    score = float(runtime.get("step_reward", 0.0))
    final_score = float(runtime.get("final_score", 0.0))
    success = bool(runtime.get("success", False))
    return {
        "score": score,
        "acc": float(success),
        "success": success,
        "final_score": final_score,
        "success_reward": float(runtime.get("success_reward", 0.0)),
        "invalid_action": bool(runtime.get("invalid_action", False)),
        "action_available": bool(runtime.get("action_available", False)),
        "num_steps": int(runtime.get("num_steps", 0)),
        "task_name": str(runtime.get("task_name", "")),
        "variation_idx": int(runtime.get("variation_idx", -1)),
        "topic": str(runtime.get("topic", "")),
        "split": str(runtime.get("split", "")),
    }
