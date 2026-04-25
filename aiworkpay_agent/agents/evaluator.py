"""Evaluator agent — synthesises task results and judges goal satisfaction."""

import json
import logging
import time
from typing import Any

from anthropic import AsyncAnthropic

from ..config import Settings
from ..models import FinalResult, TaskResult, TaskSource

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an Evaluator agent for a multi-agent AI system.

You receive an objective and a set of completed task results. Your job is to:
1. Synthesise all results into a single, coherent, high-quality final answer.
2. Assess whether the objective has been fully satisfied.

Reply with ONLY a valid JSON object — no markdown fences, no prose outside JSON:
{
  "final_result": "<comprehensive answer to the objective>",
  "goal_reached": true | false,
  "reasoning": "<one sentence explaining goal_reached decision>"
}
"""


class EvaluatorAgent:
    """
    Synthesises multiple TaskResults into a FinalResult and decides whether
    the overall objective has been satisfied.
    """

    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def evaluate(
        self,
        objective: str,
        results: list[TaskResult],
        loop_iterations: int = 1,
    ) -> tuple[FinalResult, bool]:
        """
        Return (FinalResult, goal_reached).

        goal_reached=True means the orchestrator should stop looping.
        """
        start_ms = int(time.monotonic() * 1000)
        logger.info(
            "Evaluator: synthesising %d task results for objective (%d chars)",
            len(results),
            len(objective),
        )

        user_content = self._build_user_message(objective, results)

        response = await self._client.messages.create(
            model=self._settings.claude_model,
            max_tokens=4096,
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

        duration_ms = int(time.monotonic() * 1000) - start_ms
        parsed = self._parse_response(response)

        total_cost = sum(r.cost for r in results)
        total_duration = sum(r.duration_ms for r in results) + duration_ms
        human_tasks = sum(1 for r in results if r.source == TaskSource.HUMAN)
        ai_tasks = sum(1 for r in results if r.source == TaskSource.AI)

        final = FinalResult(
            objective=objective,
            final_result=parsed["final_result"],
            sources=results,
            total_cost=total_cost,
            total_duration_ms=total_duration,
            human_tasks=human_tasks,
            ai_tasks=ai_tasks,
            loop_iterations=loop_iterations,
        )

        goal_reached: bool = parsed.get("goal_reached", True)
        logger.info(
            "Evaluator: goal_reached=%s — %s",
            goal_reached,
            parsed.get("reasoning", ""),
        )
        return final, goal_reached

    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(
        objective: str, results: list[TaskResult]
    ) -> str:
        parts = [f"Objective: {objective}", "\nCompleted task results:"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"\n[Task {i}]"
                f"\nSource: {r.source.value}"
                f"\nConfidence: {r.confidence:.2f}"
                f"\nResult:\n{r.result or '(no result)'}"
            )
            if r.error:
                parts.append(f"Error: {r.error}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        import re

        for block in response.content:
            if block.type == "text":
                raw = block.text.strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                return json.loads(raw)
        raise ValueError("Evaluator response contained no text block")
