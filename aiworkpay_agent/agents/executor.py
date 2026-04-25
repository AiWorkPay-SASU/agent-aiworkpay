"""Executor agent — runs a single Task using the Claude API with streaming."""

import json
import logging
import time
from typing import Any

from anthropic import AsyncAnthropic

from ..config import Settings
from ..models import Task, TaskResult, TaskSource

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an Executor agent for a multi-agent AI system.

Your sole responsibility is to complete the task you are given as thoroughly
and accurately as possible. Think carefully before answering.

Guidelines:
- Be concise but complete. Do not pad responses.
- Cite reasoning steps when helpful.
- If you are uncertain, say so explicitly — do not fabricate information.
- End your response with a self-assessment line in this exact format:
  CONFIDENCE: <0.00–1.00>
  Example: CONFIDENCE: 0.85
"""


class ExecutorAgent:
    """
    Runs a single Task against Claude and returns a TaskResult.

    Confidence is extracted from the model's own self-assessment line,
    falling back to a heuristic when the line is absent.
    """

    def __init__(self, client: AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def execute(self, task: Task) -> TaskResult:
        """Stream-execute the task and return a TaskResult."""
        start_ms = int(time.monotonic() * 1000)

        user_content = self._build_user_message(task)
        logger.info(
            "Executor: running task %s (type=%s)", task.id, task.type.value
        )

        async with self._client.messages.stream(
            model=self._settings.claude_model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            final_message = await stream.get_final_message()

        duration_ms = int(time.monotonic() * 1000) - start_ms
        text, confidence, has_thinking = self._parse_response(final_message)

        usage = final_message.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        # Rough cost estimate using Opus 4.7 pricing ($5/1M in, $25/1M out)
        cost = (input_tokens / 1_000_000 * 5.0) + (output_tokens / 1_000_000 * 25.0)

        logger.info(
            "Executor: task %s done — confidence=%.2f duration=%dms cost=$%.4f",
            task.id,
            confidence,
            duration_ms,
            cost,
        )

        return TaskResult(
            task_id=task.id,
            result=text,
            confidence=confidence,
            source=TaskSource.AI,
            cost=cost,
            duration_ms=duration_ms,
            metadata={
                "has_thinking": has_thinking,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": self._settings.claude_model,
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(task: Task) -> str:
        parts = [f"Task type: {task.type.value}", f"Task: {task.content}"]
        if task.context:
            parts.append(f"Context:\n{json.dumps(task.context, indent=2)}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(message: Any) -> tuple[str, float, bool]:
        """Return (text, confidence, has_thinking_block)."""
        text_parts: list[str] = []
        has_thinking = False

        for block in message.content:
            if block.type == "thinking":
                has_thinking = True
            elif block.type == "text":
                text_parts.append(block.text)

        full_text = "\n".join(text_parts).strip()

        # Extract self-reported confidence from the last CONFIDENCE: line
        confidence = _extract_confidence(full_text)
        if confidence is None:
            confidence = _heuristic_confidence(full_text, has_thinking)

        # Strip the confidence line from the visible result
        lines = full_text.splitlines()
        cleaned = [l for l in lines if not l.strip().upper().startswith("CONFIDENCE:")]
        result_text = "\n".join(cleaned).strip()

        return result_text, confidence, has_thinking


def _extract_confidence(text: str) -> float | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip().upper()
        if stripped.startswith("CONFIDENCE:"):
            try:
                value = float(stripped.split(":", 1)[1].strip())
                return max(0.0, min(1.0, value))
            except ValueError:
                pass
    return None


def _heuristic_confidence(text: str, has_thinking: bool) -> float:
    base = 0.5
    if has_thinking:
        base += 0.2
    length = len(text)
    if length > 500:
        base += 0.1
    if length > 1500:
        base += 0.1
    uncertainty_phrases = (
        "i'm not sure",
        "i am not sure",
        "i don't know",
        "uncertain",
        "unclear",
        "cannot confirm",
        "may be incorrect",
    )
    lower = text.lower()
    if any(p in lower for p in uncertainty_phrases):
        base -= 0.2
    return round(max(0.0, min(1.0, base)), 2)
