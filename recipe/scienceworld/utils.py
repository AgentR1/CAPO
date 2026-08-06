from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def scienceworld_home() -> Path:
    return Path(os.environ.get("SCIENCEWORLD_HOME", "/data/wdy/ScienceWorld/official-ScienceWorld")).expanduser()


def scienceworld_jar_path() -> Path:
    return Path(os.environ.get("SCIENCEWORLD_JAR_PATH", scienceworld_home() / "scienceworld" / "scienceworld.jar"))


@dataclass
class ScienceWorldToolExecutor:
    max_episode_steps: int

    def __post_init__(self) -> None:
        from scienceworld import ScienceWorldEnv

        jar_path = scienceworld_jar_path()
        if not jar_path.is_file():
            raise FileNotFoundError(f"ScienceWorld jar not found: {jar_path}")
        self.env = ScienceWorldEnv(serverPath=str(jar_path), envStepLimit=self.max_episode_steps)

    def reset(self, task_name: str, variation_idx: int, simplification: str) -> tuple[str, dict[str, Any]]:
        self.env.load(task_name, variation_idx, simplification)
        observation, info = self.env.reset()
        return str(observation), dict(info)

    def step(self, command: str) -> tuple[str, float, bool, dict[str, Any]]:
        observation, reward, done, info = self.env.step(command)
        return str(observation), float(reward), bool(done), dict(info)

    def action_space(self) -> tuple[list[str], list[str]]:
        """Return compact, state-conditioned action templates and visible referents."""
        return (
            [str(action) for action in self.env.get_possible_actions()],
            [str(obj) for obj in self.env.get_possible_objects()],
        )

    def valid_action_object_combinations(self) -> list[str]:
        """Return canonical state-valid commands for diagnostics, never for the prompt."""
        return [str(action) for action in self.env.get_valid_action_object_combinations()]

    def close(self) -> None:
        self.env.close()


class ScienceWorldEnvPool:
    """Actor-local pool. A leased JVM remains exclusive for one whole trajectory."""

    def __init__(self, pool_size: int, max_episode_steps: int):
        self.pool_size = pool_size
        self.max_episode_steps = max_episode_steps
        self._queue: asyncio.Queue[ScienceWorldToolExecutor] | None = None

    async def acquire(self) -> ScienceWorldToolExecutor:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.pool_size)
            for _ in range(self.pool_size):
                self._queue.put_nowait(ScienceWorldToolExecutor(self.max_episode_steps))
        return await self._queue.get()

    def release(self, executor: ScienceWorldToolExecutor) -> None:
        assert self._queue is not None
        self._queue.put_nowait(executor)


_POOLS: dict[tuple[int, int, int], ScienceWorldEnvPool] = {}


def get_actor_local_pool(worker_index: int, pool_size: int, max_episode_steps: int) -> ScienceWorldEnvPool:
    key = (worker_index, pool_size, max_episode_steps)
    if key not in _POOLS:
        _POOLS[key] = ScienceWorldEnvPool(pool_size, max_episode_steps)
    return _POOLS[key]


def format_action_templates(action_templates: list[str]) -> str:
    return "\n".join(f"- {action}" for action in action_templates) if action_templates else "None"


def format_visible_objects(visible_objects: list[str]) -> str:
    return "\n".join(f"- {obj}" for obj in visible_objects) if visible_objects else "None"


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


def format_action_ledger(history: list[dict[str, str]], *, recent_limit: int = 2) -> str:
    """Keep all older actions without repeatedly injecting their observations."""
    earlier_actions = history[:-recent_limit] if recent_limit > 0 else history
    if not earlier_actions:
        return "None"
    return "\n".join(
        f"[Action {index}] {record['action']}" for index, record in enumerate(earlier_actions, start=1)
    )
