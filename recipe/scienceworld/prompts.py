SCIENCEWORLD_SYSTEM_PROMPT = """You are an agent operating ScienceWorld.
Choose exactly one command formed from a supplied action template and visible object referents, then execute it with env_step.
Use the official observation and task description as the source of truth. Do not answer in prose."""

SCIENCEWORLD_USER_PROMPT = """### Task
{task_description}

### Earlier Actions
{action_ledger}

### Recent Interaction History
{history}

### Current Observation
{observation}

### Action Templates
Use `OBJ` placeholders only with exact names from Visible Object Referents.
{action_templates}

### Visible Object Referents
{visible_objects}

### Output Format
<think>Briefly decide the next experiment action.</think>
<tool_call>
{{"name": "env_step", "arguments": {{"command": "one command using the templates and visible referents"}}}}
</tool_call>
"""

SCIENCEWORLD_TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "env_step",
        "description": "Execute one exact ScienceWorld candidate command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]
