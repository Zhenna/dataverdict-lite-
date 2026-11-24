"""Guardrail and behaviour tests for tools.py.

Run `python setup_demo.py` first, then `pytest -q`.
"""

import json
from pathlib import Path

import pytest

from tools import Toolbox

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "currency_case"
TRUTH = ROOT / "evals" / "ground_truth" / "currency_case.json"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="run python setup_demo.py first")


@pytest.fixture(scope="module")
def tb() -> Toolbox:
    return Toolbox(FIXTURE)


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(TRUTH.read_text())


# ---------------------------------------------------------------- warehouse
def test_list_tables(tb):
    tables = tb.list_tables()["tables"]
    assert "silver.stg_payments" in tables and "gold.monthly_revenue" in tables


def test_describe_table(tb):
    cols = {c["name"] for c in tb.describe_table("silver.stg_payments")["columns"]}
    assert {"payment_id", "order_id", "amount_usd"} <= cols


@pytest.mark.parametrize("bad", ["payments", "silver.x; drop table y", "main.foo", "../x", ""])
def test_describe_table_rejects_bad_names(tb, bad):
    assert "error" in tb.describe_table(bad)


def test_run_query_returns_rows(tb):
    result = tb.run_query("SELECT SUM(amount_usd) AS total FROM silver.stg_payments;")
    assert result["columns"] == ["total"] and result["rows"][0][0] > 0


def test_run_query_truncates(tb):
    result = tb.run_query("SELECT * FROM bronze.raw_payments")
    assert result["row_count"] == 50 and result["truncated"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "DROP TABLE silver.stg_payments",
        "CREATE TABLE silver.x AS SELECT 1",
        "INSERT INTO silver.stg_payments SELECT * FROM silver.stg_payments",
        "COPY silver.stg_payments TO 'out.csv'",
        "ATTACH 'other.duckdb' AS o",
        "SELECT * FROM read_csv_auto('seeds/raw_payments.csv')",
        "SELECT * FROM glob('*')",
        "PRAGMA database_list",
        "SET enable_external_access = true",
    ],
)
def test_run_query_blocks_unsafe_sql(tb, sql):
    assert "error" in tb.run_query(sql)


def test_connection_is_read_only_even_if_tool_checks_are_bypassed(tb):
    with pytest.raises(Exception):
        tb.con.execute("CREATE TABLE silver.sneaky (i INTEGER)")


def test_external_access_is_disabled_even_if_tool_checks_are_bypassed(tb):
    with pytest.raises(Exception):
        tb.con.execute("SELECT * FROM read_csv_auto('seeds/raw_payments.csv')").fetchall()


# ---------------------------------------------------------------- git
def test_git_log_marks_deployed_history(tb, truth):
    commits = tb.git_log("silver.sql")["commits"]
    assert [c["commit"] for c in commits][0] == truth["faulty_commit"]
    assert all(c["in_deployed_history"] for c in commits)


def test_git_diff_shows_the_fault(tb, truth):
    result = tb.git_diff(truth["faulty_commit"][:7])
    assert "-    amount / 100.0 AS amount_usd" in result["diff"]
    assert "+    amount AS amount_usd" in result["diff"]


@pytest.mark.parametrize("bad", ["--output=x", "HEAD", "HEAD~1", "zzzzzzz", "abc", "0000000", ""])
def test_git_diff_rejects_bad_commits(tb, bad):
    assert "error" in tb.git_diff(bad)


@pytest.mark.parametrize(
    "path",
    [
        "../../evals/ground_truth/currency_case.json",
        "../warehouse.duckdb",
        "/etc/passwd",
        "-p",
        ".git/config",
    ],
)
def test_git_log_rejects_paths_outside_repo(tb, path):
    assert "error" in tb.git_log(path)


# ---------------------------------------------------------------- verdict
def _good_verdict(truth: dict) -> dict:
    return {
        "status": "root_cause_found",
        "affected_table": "gold.monthly_revenue",
        "origin_layer": "silver",
        "origin_table": "silver.stg_payments",
        "suspect_commit": truth["faulty_commit"],
        "attribution_level": "supported",
        "root_cause": "Cents-to-dollars division removed",
        "evidence": [{"tool_call_id": "call_1", "finding": "Silver equals Bronze cents"}],
        "suggested_fix": "Restore division by 100",
        "confidence": "high",
    }


def test_valid_verdict_accepted(tb, truth):
    assert tb.submit_verdict(**_good_verdict(truth))["accepted"] is True


def test_verdict_requires_attribution_level_with_commit(tb, truth):
    v = _good_verdict(truth)
    v["attribution_level"] = None
    assert "error" in tb.submit_verdict(**v)


def test_verdict_requires_evidence_for_root_cause(tb, truth):
    v = _good_verdict(truth)
    v["evidence"] = []
    assert "error" in tb.submit_verdict(**v)


def test_agent_cannot_submit_budget_exhausted(tb):
    assert "error" in tb.submit_verdict(status="budget_exhausted", confidence="low")


def test_unknown_tool_returns_error(tb):
    assert "error" in tb.execute("rm_rf", {})
