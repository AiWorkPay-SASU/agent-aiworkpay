"""Planner agent — decomposes an objective into an ordered list of Tasks."""

import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from ..config import Settings
from ..models import Task, TaskType

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a Planner agent for a multi-agent AI system.

Given a high-level objective, decompose it into a precise, ordered list of
atomic tasks that together will achieve the objective. Each task should be
independently executable by a specialist agent.

Guidelines:
- Prefer fewer, well-scoped tasks over many fine-grained ones (aim for 3–7).
- Assign a task type that best matches the work: research, analysis, synthesis,
  verification, code, or custom.
- Set priority 1 (highest) to 10 (lowest) based on execution order and
  dependency weight.
- The final task should always be a synthesis or verification task.

Output ONLY a valid JSON array with no markdown fences, no prose, and no keys
beyond those shown:
[
  {"type": "research|analysis|synthesis|verification|code|custom",
   "content": "<concise task description>",
   "priority": <integer 1-10>}
]
"""


class PlannerAgent:
    """Decomposes a high-level objective into an ordered list of Tasks."""

    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def plan(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> list[Task]:
        """Return a decomposed task list for the given objective."""
        user_content = f"Objective: {objective}"
        if context:
            user_content += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        logger.info("Planner: decomposing objective (%d chars)", len(objective))

        response = await self._client.messages.create(
            model=self._settings.claude_model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        raw_json = self._extract_text(response)
        tasks = self._parse_tasks(raw_json, context or {})
        logger.info("Planner: produced %d tasks", len(tasks))
        return tasks

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text
        raise ValueError("Planner response contained no text block")

    @staticmethod
    def _parse_tasks(raw: str, context: dict[str, Any]) -> list[Task]:
        # Strip optional markdown fences the model may still emit
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE)

        items: list[dict[str, Any]] = json.loads(raw)
        tasks: list[Task] = []
        for item in items:
            task_type = TaskType(item["type"])
            tasks.append(
                Task(
                    type=task_type,
                    content=item["content"],
                    priority=int(item.get("priority", 5)),
                    context=context,
                )
            )
        return tasks
