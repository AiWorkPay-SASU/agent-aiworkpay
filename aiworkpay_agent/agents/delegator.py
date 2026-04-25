"""Delegator agent — routes low-confidence tasks to the AIWorkPay workforce."""

import logging
import time

from ..client import AsyncAIWorkPayClient, MockAIWorkPayClient
from ..config import Settings
from ..models import Task, TaskResult, TaskSource

logger = logging.getLogger(__name__)

AIWorkPayClient = AsyncAIWorkPayClient | MockAIWorkPayClient


class DelegatorAgent:
    """
    Submits a Task to the AIWorkPay human workforce, waits for the result,
    then approves payment and returns a TaskResult.
    """

    def __init__(
        self,
        client: AIWorkPayClient,
        settings: Settings,
    ) -> None:
        self._client = client
        self._settings = settings

    async def delegate(self, task: Task) -> TaskResult:
        """Delegate task to a human worker and return the completed TaskResult."""
        start_ms = int(time.monotonic() * 1000)

        payload = {
            "type": task.type.value,
            "content": task.content,
            "context": task.context,
            "budget": task.budget or self._settings.default_task_budget,
            "priority": task.priority,
        }

        logger.info(
            "Delegator: submitting task %s to human workforce (budget=$%.2f)",
            task.id,
            payload["budget"],
        )

        created = await self._client.create_task(payload)
        remote_id: str = created["id"]

        try:
            completed = await self._client.wait_for_result(remote_id)
        except Exception as exc:
            duration_ms = int(time.monotonic() * 1000) - start_ms
            logger.error(
                "Delegator: task %s failed while waiting: %s", task.id, exc
            )
            return TaskResult(
                task_id=task.id,
                result=None,
                confidence=0.0,
                source=TaskSource.HUMAN,
                cost=0.0,
                duration_ms=duration_ms,
                metadata={"remote_id": remote_id},
                error=str(exc),
            )

        await self._client.approve_and_pay(remote_id)
        duration_ms = int(time.monotonic() * 1000) - start_ms

        result_text: str = completed.get("result", "")
        worker: str = completed.get("worker", "unknown")
        cost: float = float(completed.get("cost", payload["budget"]))

        logger.info(
            "Delegator: task %s completed by worker=%s in %dms",
            task.id,
            worker,
            duration_ms,
        )

        return TaskResult(
            task_id=task.id,
            result=result_text,
            confidence=1.0,  # human results are treated as authoritative
            source=TaskSource.HUMAN,
            cost=cost,
            duration_ms=duration_ms,
            metadata={"remote_id": remote_id, "worker": worker},
        )
