from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recipe.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv
from recipe.alfworld.prompts import ALFWORLD_SYSTEM_PROMPT, ALFWORLD_USER_PROMPT


INVALID_TOOL_CALL_ACTION = "<invalid_tool_call>"


@dataclass
class AlfworldToolExecutor:
    max_episode_steps: int = 50
    _env: AlfworldTextworldEnv = field(init=False)
    _history_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._env = AlfworldTextworldEnv(max_episode_steps=self.max_episode_steps)

    def reset(self, game_relative_path: str, task_id: str | None = None) -> str:
        self._history_actions.clear()
        return self._env.reset(game_relative_path=game_relative_path, task_id=task_id)

    def reset_with_info(self, game_relative_path: str, task_id: str | None = None) -> tuple[str, dict[str, Any]]:
        self._history_actions.clear()
        return self._env.reset_with_info(game_relative_path=game_relative_path, task_id=task_id)

    def step(self, command: str) -> dict[str, Any]:
        self._history_actions.append(command)
        observation, reward, done, info = self._env.step(command)
        return {
            "observation": str(observation),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "history_actions": list(self._history_actions),
        }


def format_recent_interactions(history: list[dict[str, str]], *, limit: int = 2) -> str:
    """Render the fixed observation/action memory window used by the agent."""
    if not history:
        return "None"
    recent = history[-limit:]
    start = len(history) - len(recent) + 1
    return "\n\n".join(
        f"[Observation {start + offset}]\n{record['observation']}\n[Action {start + offset}]\n{record['action']}"
        for offset, record in enumerate(recent)
    )


def format_admissible_commands(commands: list[str] | None) -> str:
    if not isinstance(commands, list) or not commands:
        return "None"
    return "\n".join(f"- {command}" for command in commands if command != "help")


def extract_task_text(observation: str, fallback: str | None = None) -> str:
    marker = "Your task is to:"
    if marker in observation:
        task = observation.split(marker, 1)[1].strip()
        task = task.split("\n", 1)[0].strip()
        return f"{marker} {task}"
    if fallback:
        fallback = str(fallback).strip()
        if fallback.lower().startswith(marker.lower()):
            return fallback
        return f"{marker} {fallback}"
    return f"{marker} Unknown."


def build_alfworld_messages(
    *,
    task_text: str,
    observation: str,
    recent_history: list[dict[str, str]],
    admissible_commands: list[str] | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ALFWORLD_USER_PROMPT.format(
                task_text=task_text,
                observation=observation,
                history_actions=format_recent_interactions(recent_history),
                admissible_commands=format_admissible_commands(admissible_commands),
            ),
        },
    ]


def build_invalid_tool_call_observation(previous_observation: str, reason: str) -> str:
    return (
        "Invalid tool call. You must call the `env_step` tool exactly once with JSON arguments "
        'like {"command": "<one admissible command>"}. '
        f"Reason: {reason}\n\n"
        "The environment state did not change. Current Observation:\n"
        f"{previous_observation}"
    )
