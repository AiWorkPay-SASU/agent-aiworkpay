"""
AIWorkPay Multi-Agent Orchestrator — entry point.

Usage
-----
Single-pass mode (default):
    python main.py "Analyse the competitive landscape of LLM providers"

Autonomous loop mode:
    python main.py --autonomous "Write a comprehensive market analysis"

Development (mock human workforce, no AIWORKPAY_API_KEY needed):
    python main.py "Summarise the risks of AGI"

Environment
-----------
Copy .env.example to .env and fill in at least ANTHROPIC_API_KEY.
"""

import argparse
import asyncio
import json
import logging
import sys

from aiworkpay_agent import Orchestrator
from aiworkpay_agent.config import get_settings


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AIWorkPay Multi-Agent Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("objective", help="High-level objective for the agents")
    p.add_argument(
        "--autonomous",
        action="store_true",
        help="Run in autonomous loop mode (iterates until goal is satisfied)",
    )
    p.add_argument(
        "--context",
        metavar="JSON",
        default="{}",
        help="Optional JSON object with extra context passed to all agents",
    )
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    try:
        context: dict = json.loads(args.context)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --context is not valid JSON: {exc}", file=sys.stderr)
        return 1

    settings = get_settings()

    async with Orchestrator(settings) as orch:
        if args.autonomous:
            result = await orch.run_autonomous(args.objective, context)
        else:
            result = await orch.run(args.objective, context)

    if args.output == "json":
        print(result.model_dump_json(indent=2))
    else:
        _print_text(result)

    return 0


def _print_text(result) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"OBJECTIVE : {result.objective}")
    print(sep)
    print(result.final_result)
    print(sep)
    print(
        f"Tasks: {result.ai_tasks} AI + {result.human_tasks} human  |  "
        f"Iterations: {result.loop_iterations}  |  "
        f"Duration: {result.total_duration_ms / 1000:.1f}s  |  "
        f"Cost: ${result.total_cost:.4f}"
    )
    print(sep)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    try:
        exit_code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        logging.getLogger(__name__).exception("Unhandled error: %s", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
