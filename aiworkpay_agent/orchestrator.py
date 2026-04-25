"""
Central Orchestrator.

Coordinates Planner → parallel Executor(s) + Delegator fallback →
Evaluator in a loop until the goal is satisfied or max iterations reached.
"""

import asyncio
import logging
from typing import Any

from anthropic import AsyncAnthropic

from .agents import DelegatorAgent, EvaluatorAgent, ExecutorAgent, PlannerAgent
from .client import build_client
from .config import Settings, get_settings
from .models import FinalResult, Task, TaskResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level coordinator for the AIWorkPay multi-agent system.

    Usage
    -----
    Single-pass (plan → execute → evaluate):

        result = await Orchestrator().run("Summarise the risks of AGI")

    Autonomous loop (iterates until goal satisfied or max loops reached):

        result = await Orchestrator().run_autonomous("Write a market analysis")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._anthropic = AsyncAnthropic(
            api_key=self._settings.anthropic_api_key
        )
        self._aiworkpay = build_client(
            api_key=self._settings.aiworkpay_api_key,
            base_url=self._settings.aiworkpay_base_url,
        )
        self._planner = PlannerAgent(self._anthropic, self._settings)
        self._executor = ExecutorAgent(self._anthropic, self._settings)
        self._evaluator = EvaluatorAgent(self._anthropic, self._settings)
        self._delegator = DelegatorAgent(self._aiworkpay, self._settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> FinalResult:
        """Single planning → execution → evaluation pass."""
        logger.info("Orchestrator.run: objective=%r", objective[:80])

        tasks = await self._planner.plan(objective, context)
        results = await self._execute_tasks(tasks)
        final, _ = await self._evaluator.evaluate(objective, results)
        return final

    async def run_autonomous(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> FinalResult:
        """
        Autonomous loop: plan → execute → evaluate → repeat until the
        Evaluator reports goal_reached or max_autonomous_loops is hit.

        Each iteration passes the previous FinalResult's synthesis as
        additional context so the Planner can refine its approach.
        """
        max_loops = self._settings.max_autonomous_loops
        accumulated_context: dict[str, Any] = dict(context or {})
        all_results: list[TaskResult] = []
        final: FinalResult | None = None

        logger.info(
            "Orchestrator.run_autonomous: objective=%r max_loops=%d",
            objective[:80],
            max_loops,
        )

        for iteration in range(1, max_loops + 1):
            logger.info("Autonomous loop — iteration %d/%d", iteration, max_loops)

            tasks = await self._planner.plan(objective, accumulated_context)
            iteration_results = await self._execute_tasks(tasks)
            all_results.extend(iteration_results)

            final, goal_reached = await self._evaluator.evaluate(
                objective, all_results, loop_iterations=iteration
            )

            if goal_reached:
                logger.info(
                    "Orchestrator: goal reached at iteration %d", iteration
                )
                break

            # Feed synthesised answer back so next iteration builds on it
            accumulated_context["previous_synthesis"] = final.final_result
            accumulated_context["iteration"] = iteration
            logger.info(
                "Orchestrator: goal not yet reached — continuing (iteration %d)",
                iteration,
            )

        if final is None:
            raise RuntimeError("Autonomous loop produced no result")

        return final

    async def close(self) -> None:
        """Release underlying HTTP clients."""
        await self._anthropic.close()
        await self._aiworkpay.close()

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_tasks(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute all tasks in parallel.

        Each task is attempted by the Executor first. If the confidence
        score falls below the threshold, it is re-routed to the Delegator
        (human workforce).
        """
        logger.info(
            "Orchestrator: executing %d tasks in parallel (threshold=%.2f)",
            len(tasks),
            self._settings.confidence_threshold,
        )

        coros = [self._run_single_task(t) for t in tasks]
        results: list[TaskResult] = await asyncio.gather(*coros)
        return list(results)

    async def _run_single_task(self, task: Task) -> TaskResult:
        """Run one task: try AI first, delegate to human if confidence is low."""
        ai_result = await self._executor.execute(task)

        if ai_result.confidence >= self._settings.confidence_threshold:
            return ai_result

        logger.info(
            "Task %s confidence=%.2f < threshold=%.2f — delegating to human",
            task.id,
            ai_result.confidence,
            self._settings.confidence_threshold,
        )
        human_result = await self._delegator.delegate(task)
        return human_result
