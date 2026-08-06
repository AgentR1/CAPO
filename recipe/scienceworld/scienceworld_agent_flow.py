from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from arft.reward_loop import ARFTRewardLoopWorker as RewardLoopWorker
from recipe.scienceworld.prompts import SCIENCEWORLD_SYSTEM_PROMPT, SCIENCEWORLD_TOOL_SCHEMAS, SCIENCEWORLD_USER_PROMPT
from recipe.scienceworld.utils import (
    format_action_templates,
    format_action_ledger,
    format_recent_interactions,
    format_visible_objects,
    get_actor_local_pool,
)
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, DictConfigWrap
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _recover_tool_calls(text: str) -> list[FunctionCall]:
    calls: list[FunctionCall] = []
    for raw in _TOOL_CALL_BLOCK.findall(text):
        try:
            payload = json.loads(raw.strip())
            name, arguments = payload["name"], payload["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            if isinstance(name, str) and isinstance(arguments, str):
                calls.append(FunctionCall(name=name, arguments=arguments))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return calls


def _parse_env_step_command(tool_calls: list[FunctionCall]) -> tuple[str, str | None]:
    """Return a command or an invalid-action reason without raising on model output."""
    if not tool_calls:
        return "", "missing env_step tool call"
    if len(tool_calls) != 1:
        return "", f"expected exactly one env_step tool call, got {len(tool_calls)}"

    tool_call = tool_calls[0]
    if tool_call.name != "env_step":
        return "", f"expected env_step tool, got {tool_call.name!r}"

    try:
        tool_args = json.loads(tool_call.arguments)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse env_step arguments: %r", exc)
        return "", f"failed to parse env_step arguments: {exc}"

    if not isinstance(tool_args, dict):
        return "", f"env_step arguments must decode to an object, got {type(tool_args).__name__}"
    command = tool_args.get("command")
    if not isinstance(command, str):
        return "", "command argument must be a string"
    command = command.strip()
    if not command:
        return "", "missing command argument"
    return command, None


@register("scienceworld_agent")
class ScienceWorldAgentFlow(AgentFlowBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        reward_loop_worker: RewardLoopWorker,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, reward_loop_worker, tokenizer, processor, **kwargs)
        self.max_steps = int(kwargs.get("max_steps", 50))
        self.max_episode_steps = int(kwargs.get("max_episode_steps", 50))
        self.env_pool_size = int(kwargs.get("env_pool_size", 1))
        self.simplification = str(kwargs.get("simplification", ""))
        self.success_reward = float(kwargs.get("success_reward", 10.0))
        self.worker_index = int(kwargs.get("agent_flow_worker_index", 0) or 0)
        self.tool_parser = ToolParser.get_tool_parser(
            self.config.actor_rollout_ref.rollout.multi_turn.format, tokenizer
        )
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        extra = kwargs.get("extra_info") or {}
        task_name = str(extra["task_name"])
        variation_idx = int(extra["variation_idx"])
        split = str(extra.get("split", "train"))
        topic = str(extra.get("topic", ""))
        pool = get_actor_local_pool(self.worker_index, self.env_pool_size, self.max_episode_steps)
        executor = await pool.acquire()
        try:
            observation, info = executor.reset(task_name, variation_idx, self.simplification)
            task_description = str(info.get("taskDesc", extra.get("task_description", "")))
            recent_history: list[dict[str, str]] = []
            steps: list[AgentFlowStep] = []
            metrics: dict[str, Any] = {}
            done = False
            num_steps = 0
            final_score = float(info.get("score", 0.0))

            while num_steps < self.max_steps and not done:
                num_steps += 1
                anchor_obs = observation
                # This exact state-valid list is a diagnostic label only. It is
                # never exposed in the prompt and never gates env.step.
                valid_commands = executor.valid_action_object_combinations()
                action_templates, visible_objects = executor.action_space()
                messages = [
                    {"role": "system", "content": SCIENCEWORLD_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": SCIENCEWORLD_USER_PROMPT.format(
                            task_description=task_description,
                            observation=anchor_obs,
                            action_ledger=format_action_ledger(recent_history),
                            history=format_recent_interactions(recent_history),
                            action_templates=format_action_templates(action_templates),
                            visible_objects=format_visible_objects(visible_objects),
                        ),
                    },
                ]
                prompt_ids = await self.apply_chat_template(messages, tools=SCIENCEWORLD_TOOL_SCHEMAS)
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                    )
                response_ids = output.token_ids[: self.response_length]
                _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                if not tool_calls:
                    tool_calls = _recover_tool_calls(self.tokenizer.decode(response_ids, skip_special_tokens=True))

                invalid_action = False
                invalid_reason: str | None = None
                action_available = False
                env_delta = 0.0
                success = False
                command, invalid_reason = _parse_env_step_command(tool_calls)
                if invalid_reason is not None:
                    invalid_action = True
                    # A malformed tool call has no command to execute. Keep the
                    # raw state observation unchanged; the failure appears once
                    # in history rather than recursively inside the observation.
                    recent_history.append(
                        {"observation": anchor_obs, "action": f"<invalid_tool_call>: {invalid_reason}"}
                    )
                else:
                    action_available = command in valid_commands

                    # EPO-style behavior: parsed commands always reach
                    # ScienceWorld. The valid-command API uses canonical
                    # strings (for example, ``go to kitchen``), while the
                    # parser also accepts synonyms such as ``go kitchen``;
                    # canonical membership is therefore diagnostic only.
                    observation, env_delta, done, info = executor.step(command)
                    final_score = float(info.get("score", final_score))
                    recent_history.append({"observation": anchor_obs, "action": command})
                    if done and final_score > 0.0:
                        success = True

                # EPO-style terminal reward: positive terminal score earns +10;
                # non-terminal official score deltas remain diagnostics.
                step_reward = self.success_reward if success else 0.0
                reward_extra_info = {
                    "step_env_reward": env_delta,
                    "normalized_delta_score": env_delta / 100.0,
                    "step_reward": step_reward,
                    "success_reward": self.success_reward if success else 0.0,
                    "final_score": final_score,
                    "success": success,
                    "invalid_action": invalid_action,
                    "action_available": action_available,
                    # Keep diagnostic text in trajectory dumps, but never emit None:
                    # validation metric aggregation only skips strings and otherwise expects numeric values.
                    "invalid_reason": invalid_reason or "",
                    "num_steps": num_steps,
                    "task_name": task_name,
                    "variation_idx": variation_idx,
                    "topic": topic,
                    "split": split,
                    "action_templates_count": len(action_templates),
                    "visible_objects_count": len(visible_objects),
                    "valid_commands_count": len(valid_commands),
                }
                step = AgentFlowStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                    reward_score=step_reward,
                    extra_fields={"anchor_obs": anchor_obs, "reward_extra_info": reward_extra_info},
                )
                steps.append(await self._postprocess(step, **kwargs))
            return AgentFlowOutput(steps=steps, metrics=metrics)
        finally:
            pool.release(executor)
