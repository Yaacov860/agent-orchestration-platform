"""End-to-end against a running API with a real Gemini key: POST /tasks, wait, GET /result.

    docker compose up --build            # with GEMINI_API_KEY in your shell or .env
    uv run python scripts/gemini_e2e.py  # or: --base-url http://localhost:8000 --out e2e-results

Submits three tasks (research and summary; analysis and writing; code), polls each until it
finishes, fetches its steps and result, saves everything as JSON under --out, and prints the
task ids and statuses. It never reads or prints an API key: the key lives in the server.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

TASKS: dict[str, dict[str, object]] = {
    "research_and_summary": {
        "goal": "Research what the Python Global Interpreter Lock is and summarize the current "
        "state of efforts to remove it.",
        "constraints": {"max_words": 300},
        "output_format": "markdown",
    },
    "analysis_and_writing": {
        "goal": "Compare PostgreSQL and SQLite for a small internal web app with about 20 users, "
        "then write a one-page recommendation for the team.",
        "constraints": {"tone": "plain", "max_words": 500},
        "output_format": "markdown",
    },
    "code": {
        "goal": "Write a Python function that validates an email address with a regular "
        "expression, with a short explanation and examples of inputs it accepts and rejects.",
        "constraints": {"include_code_examples": True},
        "output_format": "markdown",
    },
}
FINISHED = {"completed", "failed", "cancelled"}


async def get_json(client: httpx.AsyncClient, path: str) -> tuple[int, Any]:
    """GET with a few retries on network errors: one dropped poll must not end the run."""
    for attempt in range(5):
        try:
            response = await client.get(path)
            return response.status_code, response.json()
        except httpx.TransportError:
            if attempt == 4:
                raise
            await asyncio.sleep(2)
    raise AssertionError("unreachable")


async def run_one(
    client: httpx.AsyncClient, name: str, body: dict[str, object], max_wait: float
) -> dict[str, Any]:
    submitted = await client.post("/tasks", json=body)
    submitted.raise_for_status()
    task_id = submitted.json()["task_id"]
    print(f"submitted {name}: {task_id}", flush=True)
    started = time.monotonic()
    status: dict[str, Any] = {}
    while time.monotonic() - started < max_wait:
        _, status = await get_json(client, f"/tasks/{task_id}")
        if status["status"] in FINISHED:
            break
        await asyncio.sleep(3)
    _, steps = await get_json(client, f"/tasks/{task_id}/steps")
    _, trace = await get_json(client, f"/tasks/{task_id}/trace")
    result_status, result = await get_json(client, f"/tasks/{task_id}/result")
    return {
        "name": name,
        "task_id": task_id,
        "status": status.get("status"),
        "failure_stage": status.get("failure_stage"),
        "error": status.get("error"),
        "seconds": round(time.monotonic() - started, 1),
        "agents_used": sorted({s["agent"] for s in steps["steps"]}),
        "tokens_total": status.get("usage", {}).get("total", {}).get("total"),
        "task": status,
        "steps": steps,
        "trace": trace,
        "result_http_status": result_status,
        "result": result,
    }


def save(out: Path, run: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run['name']}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--out", default="e2e-results", type=Path)
    parser.add_argument("--timeout", default=600.0, type=float, help="seconds per task")
    parser.add_argument("--only", choices=sorted(TASKS), action="append", help="run just these")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="submit all tasks at once (on a free-tier key this mostly hits the per-minute limit)",
    )
    args = parser.parse_args()

    tasks = {n: b for n, b in TASKS.items() if not args.only or n in args.only}
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        try:
            (await client.get("/health")).raise_for_status()
        except httpx.HTTPError as exc:
            print(f"API not reachable at {args.base_url}: {exc}", file=sys.stderr)
            return 2
        ready = await client.get("/health/ready")
        print(f"ready: {ready.status_code} {json.dumps(ready.json()['checks'])}")
        agents = (await client.get("/agents")).json()["agents"]
        print(f"agents: {[a['name'] for a in agents]}")
        if args.parallel:
            runs = list(
                await asyncio.gather(
                    *(run_one(client, n, b, args.timeout) for n, b in tasks.items())
                )
            )
        else:
            runs = []
            for name, body in tasks.items():
                run = await run_one(client, name, body, args.timeout)
                save(args.out, run)  # at once: a later failure must not lose this result
                runs.append(run)

    for run in runs:
        save(args.out, run)
    keys = (
        "name",
        "task_id",
        "status",
        "failure_stage",
        "seconds",
        "agents_used",
        "tokens_total",
        "error",
    )
    summary = [{k: run[k] for k in keys} for run in runs]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'task':<24} {'task_id':<38} {'status':<10} {'stage':<10} seconds")
    for row in summary:
        cells = (row["name"], row["task_id"], row["status"], row["failure_stage"], row["seconds"])
        print("{:<24} {:<38} {!s:<10} {!s:<10} {}".format(*cells))
        print(f"    agents: {row['agents_used']}  tokens: {row['tokens_total']}")
    print(f"\nsaved to {args.out}/")
    return 0 if all(r["status"] == "completed" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
