"""Investigation loop: model calls tools until it submits a verdict that survives validation.

Usage:
    export DATAVERDICT_MODEL=<model id>      # see your provider's model list
    python agent.py "Revenue for January 2018 looks about 100x too high. Investigate the root cause."
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from llm import LLMTurn, call_llm, make_client
from tools import TOOL_SPECS, Toolbox

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT / "fixtures" / "currency_case"
DEFAULT_TRACE_DIR = ROOT / "traces"

SYSTEM_PROMPT = """\
You are a data pipeline investigator. A user reports a data problem in a warehouse that was built by \
a pipeline kept in a git repository. Find out whether there is a real problem, where it originates, \
and which change caused it.

How to work:
- Gather evidence yourself with the tools. You decide where to look and in what order. Do not assume \
the problem is where the symptom appears.
- The warehouse has bronze (raw), silver (staged) and gold (reporting) schemas. Use list_tables and \
describe_table to discover them.
- Do not assume a difference between tables is a bug. Work out whether it is intended, using the \
pipeline code and what the data shows.
- To blame a commit, read its diff and check that it explains the numbers you observed. Only commits \
in the deployed history can have produced this warehouse.
- Keep queries small and targeted. Results are capped.
- You have a limited number of steps (at most {max_steps}), so avoid redundant calls.

Finishing:
- Call submit_verdict exactly once when you are done.
- Evidence: each item cites the tool_call_id of the result it came from, plus a one-sentence finding. \
When you cite a number, add an observation (row index, column name, value) copied exactly from that \
run_query result.
- attribution_level: 'supported' only if you read the commit's diff and it explains the observed \
discrepancy; 'suspected' if the relevant file changed but you did not confirm the link; null if you \
name no commit.
- If the evidence is insufficient or contradictory, use status 'inconclusive'. If nothing is wrong, \
use 'no_issue_found'. Never invent a cause.
"""

LLMFn = Callable[..., LLMTurn]


# --------------------------------------------------------------------------- verdict checks
def _values_match(actual: Any, claimed: Any) -> bool:
    try:
        return math.isclose(float(actual), float(claimed), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(actual) == str(claimed)


def _seen_commits(calls: dict[str, dict]) -> set[str]:
    seen: set[str] = set()
    for call in calls.values():
        result = call["result"]
        if call["name"] == "git_log":
            seen.update(c["commit"] for c in result.get("commits", []))
        elif call["name"] == "git_diff" and "commit" in result:
            seen.add(result["commit"])
    return seen


def check_verdict_against_trace(verdict: dict[str, Any], calls: dict[str, dict]) -> list[str]:
    """Deterministic checks of the verdict against what the tools actually returned.

    This catches invented IDs, misquoted numbers and unverified attribution. It cannot judge
    whether the agent *interpreted* the evidence correctly; the ground-truth scorer does that.
    """
    problems: list[str] = []

    for evidence in verdict.get("evidence", []):
        call_id = evidence["tool_call_id"]
        call = calls.get(call_id)
        if call is None:
            problems.append(f"Evidence cites unknown tool_call_id '{call_id}'.")
            continue
        result = call["result"]
        for obs in evidence.get("observations", []):
            columns, rows = result.get("columns"), result.get("rows")
            if columns is None or rows is None:
                problems.append(f"{call_id} is not a run_query result; observations can only cite query results.")
            elif obs["column"] not in columns or not 0 <= obs["row"] < len(rows):
                problems.append(f"{call_id} has no column '{obs['column']}' at row {obs['row']}.")
            else:
                actual = rows[obs["row"]][columns.index(obs["column"])]
                if not _values_match(actual, obs["value"]):
                    problems.append(
                        f"Observation mismatch in {call_id}: column '{obs['column']}' row {obs['row']} "
                        f"is {actual!r}, not {obs['value']!r}."
                    )

    commit = verdict.get("suspect_commit")
    if commit:
        if len(commit) < 7 or not any(sha.startswith(commit) for sha in _seen_commits(calls)):
            problems.append("suspect_commit was never returned by git_log or git_diff in this investigation.")
        elif verdict.get("attribution_level") == "supported" and not any(
            c["name"] == "git_diff" and c["result"].get("commit", "").startswith(commit) for c in calls.values()
        ):
            problems.append("attribution_level 'supported' requires reading that commit with git_diff.")
    return problems


def _budget_exhausted_verdict() -> dict[str, Any]:
    return {
        "status": "budget_exhausted",
        "affected_table": None,
        "origin_layer": None,
        "origin_table": None,
        "suspect_commit": None,
        "attribution_level": None,
        "root_cause": None,
        "evidence": [],
        "suggested_fix": None,
        "confidence": "low",
    }


# --------------------------------------------------------------------------- terminal output
def _summarize(name: str, result: dict[str, Any]) -> str:
    if "error" in result:
        detail = result.get("problems") or result["error"]
        return f"ERROR: {detail}"[:140]
    if name == "run_query":
        first = result["rows"][0] if result["rows"] else None
        return f"{result['row_count']} row(s), first: {first}"
    if name == "list_tables":
        return f"{len(result['tables'])} tables"
    if name == "describe_table":
        return f"{len(result['columns'])} columns"
    if name == "git_log":
        return f"{len(result['commits'])} commits"
    if name == "git_diff":
        return f"{result['commit'][:7]} {result['subject']}"
    if name == "get_deployment_info":
        return f"deployed {str(result.get('deployed_commit'))[:7]}"
    if name == "submit_verdict":
        return "accepted" if result.get("accepted") else "rejected"
    return "ok"


def _print_final(outcome: dict[str, Any]) -> None:
    v = outcome["verdict"]
    print("\n" + v["status"].replace("_", " ").upper())
    if v.get("origin_table"):
        print(f"Origin: {v['origin_table']} ({v.get('origin_layer')})")
    if v.get("root_cause"):
        print(f"Cause: {v['root_cause']}")
    if v.get("suspect_commit"):
        print(f"Commit: {v['suspect_commit'][:7]} ({v.get('attribution_level')})")
    if v.get("suggested_fix"):
        print(f"Fix: {v['suggested_fix']}")
    print(f"Confidence: {v['confidence']}")
    print(
        f"Steps: {outcome['steps']} | tool calls: {outcome['tool_calls']} | "
        f"tokens in/out: {outcome['input_tokens']}/{outcome['output_tokens']} | {outcome['seconds']}s"
    )
    print(f"Trace: {outcome['trace_path']}")


# --------------------------------------------------------------------------- the loop
def investigate(
    symptom: str,
    fixture_dir: str | Path = DEFAULT_FIXTURE,
    max_steps: int = 12,
    max_tool_calls: int = 30,
    model: str | None = None,
    llm: LLMFn = call_llm,
    toolbox: Toolbox | None = None,
    client: Any = None,
    trace_dir: str | Path = DEFAULT_TRACE_DIR,
    quiet: bool = False,
) -> dict[str, Any]:
    model = model or os.environ.get("DATAVERDICT_MODEL")
    if not model:
        raise SystemExit("Set DATAVERDICT_MODEL to a model id from your provider's current model list.")
    tb = toolbox or Toolbox(fixture_dir)
    if client is None and llm is call_llm:
        client = make_client()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    scenario = Path(fixture_dir).name
    trace_path = trace_dir / f"{scenario}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    started = time.time()

    def log(event: dict[str, Any]) -> None:
        with trace_path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    log({"type": "run_start", "symptom": symptom, "model": model, "max_steps": max_steps, "scenario": scenario})

    instructions = SYSTEM_PROMPT.format(max_steps=max_steps)
    input_items: list[Any] = [{"role": "user", "content": symptom}]
    calls: dict[str, dict] = {}
    tokens_in = tokens_out = steps = 0
    verdict: dict[str, Any] | None = None

    if not quiet:
        print(f"DATAVERDICT-LITE | {scenario} | model={model}\n{symptom}\n")

    while steps < max_steps and len(calls) < max_tool_calls and verdict is None:
        steps += 1
        turn = llm(client, model, instructions, input_items, TOOL_SPECS)
        tokens_in += turn.input_tokens
        tokens_out += turn.output_tokens
        log({
            "type": "model_turn", "step": steps, "text": turn.text,
            "tool_calls": [c.name for c in turn.tool_calls],
            "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
        })
        input_items += turn.output_items

        if not turn.tool_calls:
            input_items.append({
                "role": "user",
                "content": "Continue by calling a tool, or call submit_verdict if you are done.",
            })
            continue

        for tc in turn.tool_calls:
            t0 = time.time()
            if tc.error:
                result: dict[str, Any] = {"error": tc.error}
            else:
                result = tb.execute(tc.name, tc.arguments)
            if tc.name == "submit_verdict" and result.get("accepted"):
                problems = check_verdict_against_trace(result["verdict"], calls)
                if problems:
                    result = {"error": "Verdict rejected by validator. Fix these problems and resubmit.", "problems": problems}
                else:
                    verdict = result["verdict"]

            calls[tc.id] = {"name": tc.name, "arguments": tc.arguments, "result": result}
            log({
                "type": "tool_call", "step": steps, "tool_call_id": tc.id, "name": tc.name,
                "arguments": tc.arguments, "result": result, "ms": round((time.time() - t0) * 1000),
            })
            if not quiet:
                args_text = json.dumps(tc.arguments)
                args_text = args_text if len(args_text) <= 80 else args_text[:77] + "..."
                print(f"[{len(calls):02d}] {tc.name}({args_text}) -> {_summarize(tc.name, result)}")

            input_items.append({
                "type": "function_call_output",
                "call_id": tc.id,
                "output": json.dumps({"tool_call_id": tc.id, "result": result}, default=str),
            })
            if verdict is not None:
                break

    if verdict is None:
        verdict = _budget_exhausted_verdict()

    outcome = {
        "verdict": verdict,
        "steps": steps,
        "tool_calls": len(calls),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "seconds": round(time.time() - started, 1),
        "model": model,
        "trace_path": str(trace_path),
    }
    log({"type": "final", **outcome})
    if not quiet:
        _print_final(outcome)
    return outcome


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Investigate a data issue in a scenario fixture.")
    parser.add_argument("symptom", help="Free-text description of the data issue.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    investigate(args.symptom, fixture_dir=args.fixture, max_steps=args.max_steps, model=args.model)


if __name__ == "__main__":
    main()
