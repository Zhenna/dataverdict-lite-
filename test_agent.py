"""Tests the loop, trace logging and verdict validation with a scripted fake LLM (no API calls).

Run `python setup_demo.py` first, then `pytest -q`.
"""

import json
from pathlib import Path

import pytest

from agent import investigate
from llm import LLMTurn, ToolCall
from tools import Toolbox

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "currency_case"
TRUTH = ROOT / "evals" / "ground_truth" / "currency_case.json"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="run python setup_demo.py first")


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(TRUTH.read_text())


@pytest.fixture(scope="module")
def tb() -> Toolbox:
    return Toolbox(FIXTURE)


class ScriptedLLM:
    """Replays a fixed list of turns; optionally repeats the last one forever."""

    def __init__(self, turns, repeat_last=False):
        self.turns, self.repeat_last, self.i = list(turns), repeat_last, 0

    def __call__(self, client, model, instructions, input_items, tool_specs):
        if self.i < len(self.turns):
            turn = self.turns[self.i]
        elif self.repeat_last:
            turn = self.turns[-1]
        else:
            raise AssertionError("script ran out of turns")
        self.i += 1
        return turn


def turn(*calls, text=""):
    return LLMTurn(
        output_items=[],
        text=text,
        tool_calls=[ToolCall(id=i, name=n, arguments=a) for i, n, a in calls],
        input_tokens=10,
        output_tokens=5,
    )


def verdict_args(truth: dict, **overrides) -> dict:
    base = {
        "status": "root_cause_found",
        "affected_table": "gold.monthly_revenue",
        "origin_layer": "silver",
        "origin_table": "silver.stg_payments",
        "suspect_commit": truth["faulty_commit"][:7],
        "attribution_level": "supported",
        "root_cause": "Cents-to-dollars division removed in staging",
        "evidence": [
            {
                "tool_call_id": "c3",
                "finding": "Gold total equals the Bronze cents total",
                "observations": [{"row": 0, "column": "total", "value": truth["reported_total_usd"]}],
            },
            {"tool_call_id": "c7", "finding": "Commit removed the division by 100", "observations": []},
        ],
        "suggested_fix": "Restore amount / 100.0",
        "confidence": "high",
    }
    base.update(overrides)
    return base


def investigation_turns(truth: dict):
    """Turns 1-4 gather evidence; c3 is the gold total, c7 is the diff."""
    return [
        turn(("c1", "list_tables", {}), ("c2", "get_deployment_info", {})),
        turn(
            ("c3", "run_query", {"sql": "SELECT SUM(revenue_usd) AS total FROM gold.monthly_revenue"}),
            ("c4", "run_query", {"sql": "SELECT SUM(amount) AS total FROM bronze.raw_payments"}),
            ("c5", "run_query", {"sql": "SELECT SUM(amount_usd) AS total FROM silver.stg_payments"}),
        ),
        turn(("c6", "git_log", {"path": "silver.sql"})),
        turn(("c7", "git_diff", {"commit": truth["faulty_commit"][:7]})),
    ]


def run(tb, tmp_path, llm, **kwargs):
    return investigate(
        "Revenue looks about 100x too high.",
        fixture_dir=FIXTURE,
        model="test-model",
        llm=llm,
        toolbox=tb,
        client=object(),
        trace_dir=tmp_path,
        quiet=True,
        **kwargs,
    )


def test_happy_path(tb, truth, tmp_path):
    llm = ScriptedLLM(investigation_turns(truth) + [turn(("c8", "submit_verdict", verdict_args(truth)))])
    out = run(tb, tmp_path, llm)
    v = out["verdict"]
    assert v["status"] == "root_cause_found"
    assert v["origin_table"] == truth["origin_table"]
    assert truth["faulty_commit"].startswith(v["suspect_commit"])
    assert v["attribution_level"] == "supported"
    assert out["steps"] == 5 and out["tool_calls"] == 8
    assert out["input_tokens"] == 50 and out["output_tokens"] == 25


def test_trace_file_is_written(tb, truth, tmp_path):
    llm = ScriptedLLM(investigation_turns(truth) + [turn(("c8", "submit_verdict", verdict_args(truth)))])
    out = run(tb, tmp_path, llm)
    events = [json.loads(line) for line in Path(out["trace_path"]).read_text().splitlines()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "final"
    assert kinds.count("tool_call") == 8 and events[0]["model"] == "test-model"


def test_misquoted_number_is_rejected_then_corrected(tb, truth, tmp_path):
    bad = verdict_args(truth)
    bad["evidence"][0]["observations"][0]["value"] = 999.0
    llm = ScriptedLLM(
        investigation_turns(truth)
        + [turn(("c8", "submit_verdict", bad)), turn(("c9", "submit_verdict", verdict_args(truth)))]
    )
    out = run(tb, tmp_path, llm)
    assert out["verdict"]["status"] == "root_cause_found"
    assert out["steps"] == 6
    assert "mismatch" in str(Path(out["trace_path"]).read_text())


def test_unknown_tool_call_id_is_rejected(tb, truth, tmp_path):
    bad = verdict_args(truth)
    bad["evidence"][0]["tool_call_id"] = "c99"
    llm = ScriptedLLM(investigation_turns(truth) + [turn(("c8", "submit_verdict", bad))], repeat_last=True)
    out = run(tb, tmp_path, llm, max_steps=6)
    assert out["verdict"]["status"] == "budget_exhausted"


def test_supported_requires_reading_the_diff(tb, truth, tmp_path):
    turns = investigation_turns(truth)
    no_diff = turns[:3]  # skips git_diff
    bad = verdict_args(truth)
    bad["evidence"] = [bad["evidence"][0]]
    ok = verdict_args(truth, attribution_level="suspected")
    ok["evidence"] = bad["evidence"]
    llm = ScriptedLLM(no_diff + [turn(("c7", "submit_verdict", bad)), turn(("c8", "submit_verdict", ok))])
    out = run(tb, tmp_path, llm)
    assert out["verdict"]["attribution_level"] == "suspected"
    assert out["steps"] == 5


def test_unseen_commit_is_rejected(tb, truth, tmp_path):
    bad = verdict_args(truth, suspect_commit="deadbeef")
    llm = ScriptedLLM(investigation_turns(truth) + [turn(("c8", "submit_verdict", bad))], repeat_last=True)
    out = run(tb, tmp_path, llm, max_steps=6)
    assert out["verdict"]["status"] == "budget_exhausted"


def test_budget_exhausted_when_agent_never_submits(tb, truth, tmp_path):
    llm = ScriptedLLM([turn(("c1", "list_tables", {}))], repeat_last=True)
    out = run(tb, tmp_path, llm, max_steps=3)
    assert out["verdict"]["status"] == "budget_exhausted"
    assert out["steps"] == 3


def test_text_only_turn_is_nudged_and_loop_continues(tb, truth, tmp_path):
    turns = [turn(text="Let me think.")] + investigation_turns(truth)
    # tool-call ids shift nothing: ids are chosen by the script, not by position
    llm = ScriptedLLM(turns + [turn(("c8", "submit_verdict", verdict_args(truth)))])
    out = run(tb, tmp_path, llm)
    assert out["verdict"]["status"] == "root_cause_found" and out["steps"] == 6


def test_inconclusive_verdict_needs_no_commit(tb, truth, tmp_path):
    v = {"status": "inconclusive", "confidence": "low", "root_cause": None}
    llm = ScriptedLLM([turn(("c1", "list_tables", {})), turn(("c2", "submit_verdict", v))])
    out = run(tb, tmp_path, llm)
    assert out["verdict"]["status"] == "inconclusive"
    assert out["verdict"]["attribution_level"] is None
