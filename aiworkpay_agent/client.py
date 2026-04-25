"""
Async AIWorkPay client.

Provides AsyncAIWorkPayClient for production use and MockAIWorkPayClient for
local development / CI when no API key is configured.
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AIWorkPayError(Exception):
    """Base error for AIWorkPay operations."""


class TaskNotFoundError(AIWorkPayError):
    """Remote task ID does not exist."""


class TaskTimeoutError(AIWorkPayError):
    """Timed out waiting for human completion."""


class AsyncAIWorkPayClient:
    """
    Async HTTP client for the AIWorkPay REST API.

    Handles task creation, polling, and payment approval with
    exponential-backoff retries on transient errors.
    """

    _POLL_INTERVAL_S: float = 5.0
    _MAX_WAIT_S: float = 300.0
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

    async def __aenter__(self) -> "AsyncAIWorkPayClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a new task to the human workforce."""
        return await self._request("POST", "/tasks", json=payload)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Fetch the current state of a task."""
        try:
            return await self._request("GET", f"/tasks/{task_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise TaskNotFoundError(task_id) from exc
            raise

    async def wait_for_result(
        self,
        task_id: str,
        timeout_s: float = _MAX_WAIT_S,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ) -> dict[str, Any]:
        """Poll until a human completes the task or timeout is reached."""
        elapsed = 0.0
        while elapsed < timeout_s:
            task = await self.get_task(task_id)
            status = task.get("status", "")

            if status == "completed":
                return task
            if status in {"failed", "cancelled"}:
                raise AIWorkPayError(
                    f"Task {task_id} ended with status={status!r}"
                )

            logger.debug(
                "Task %s status=%s — polling again in %.1fs",
                task_id,
                status,
                poll_interval_s,
            )
            await asyncio.sleep(poll_interval_s)
            elapsed += poll_interval_s

        raise TaskTimeoutError(f"Task {task_id} timed out after {timeout_s}s")

    async def approve_and_pay(self, task_id: str) -> dict[str, Any]:
        """Approve a completed task and release worker payment."""
        return await self._request(
            "POST", f"/tasks/{task_id}/approve", json={"approved": True}
        )

    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        backoff = 2.0
        last_exc: Exception | None = None

        for attempt in range(retries + 1):
            try:
                resp = await self._http.request(method, url, json=json)
                if resp.status_code in self._RETRY_STATUS and attempt < retries:
                    logger.warning(
                        "HTTP %s on %s %s — retry %d in %.1fs",
                        resp.status_code,
                        method,
                        path,
                        attempt + 1,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < retries:
                    logger.warning(
                        "Network error on %s %s — retry %d in %.1fs: %s",
                        method,
                        path,
                        attempt + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2

        raise AIWorkPayError(
            f"Request {method} {path} failed after {retries} retries"
        ) from last_exc


class MockAIWorkPayClient:
    """
    Local mock that simulates human delegation without a real API key.

    Completes tasks automatically after a short delay so the rest of the
    pipeline can be exercised in development and CI.
    """

    async def __aenter__(self) -> "MockAIWorkPayClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def close(self) -> None:
        pass

    async def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = f"mock-{id(payload):x}"
        logger.info(
            "[MOCK] Created human task %s (type=%s)",
            task_id,
            payload.get("type"),
        )
        return {"id": task_id, "status": "pending", **payload}

    async def wait_for_result(
        self, task_id: str, **_: Any
    ) -> dict[str, Any]:
        logger.info(
            "[MOCK] Simulating human work on task %s (2s delay)", task_id
        )
        await asyncio.sleep(2.0)
        return {
            "id": task_id,
            "status": "completed",
            "result": (
                "[MOCK HUMAN RESULT] Auto-completed by the mock AIWorkPay "
                "client. Set AIWORKPAY_API_KEY to connect a real workforce."
            ),
            "worker": "mock-worker-001",
        }

    async def approve_and_pay(self, task_id: str) -> dict[str, Any]:
        logger.info("[MOCK] Approved and paid task %s", task_id)
        return {"id": task_id, "payment_status": "released"}


def build_client(
    api_key: str, base_url: str
) -> AsyncAIWorkPayClient | MockAIWorkPayClient:
    """Return a real or mock AIWorkPay client based on configuration."""
    if api_key:
        return AsyncAIWorkPayClient(api_key=api_key, base_url=base_url)
    logger.warning(
        "AIWORKPAY_API_KEY not configured — using local mock. "
        "Human delegation will be simulated automatically."
    )
    return MockAIWorkPayClient()
