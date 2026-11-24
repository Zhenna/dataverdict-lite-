"""Tests for scenario building: determinism, deployment boundary, and leakage of the answer.

Needs seeds/*.csv. Builds into a temporary directory, so it does not touch your fixtures.
Run with `pytest -q`.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import setup_demo
from evals.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(
    not (ROOT / "seeds" / "raw_payments.csv").exists(), reason="seeds/ not populated"
)

LEAKY_WORDS = ("bug", "fault", "inject", "wrong", "broken", "error", "fix")


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(ROOT / "seeds", root / "seeds")
    return root


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "--no-pager", *args], capture_output=True, text=True)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_build_is_deterministic(project, name):
    first = setup_demo.build(SCENARIOS[name], root=project)
    second = setup_demo.build(SCENARIOS[name], root=project)
    assert first == second


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_faulty_commit_is_deployed_and_a_later_commit_is_not(project, name):
    truth = setup_demo.build(SCENARIOS[name], root=project)
    repo = project / "fixtures" / name / "pipeline_repo"
    assert truth["deployed_commit"] != git(repo, "rev-parse", "HEAD").stdout.strip()
    assert len(truth["undeployed_commits"]) == 1
    assert git(repo, "merge-base", "--is-ancestor", truth["faulty_commit"], truth["deployed_commit"]).returncode == 0
    for sha in truth["undeployed_commits"]:
        assert git(repo, "merge-base", "--is-ancestor", sha, truth["deployed_commit"]).returncode != 0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_faulty_commit_touches_only_the_origin_file(project, name):
    truth = setup_demo.build(SCENARIOS[name], root=project)
    repo = project / "fixtures" / name / "pipeline_repo"
    changed = git(repo, "show", "--name-only", "--format=", truth["faulty_commit"]).stdout.split()
    assert changed == ["silver.sql"]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_answer_key_is_not_inside_the_fixture(project, name):
    truth = setup_demo.build(SCENARIOS[name], root=project)
    fixture = project / "fixtures" / name
    assert (project / "evals" / "ground_truth" / f"{name}.json").exists()
    assert not any("ground_truth" in str(p) for p in fixture.rglob("*"))
    assert truth["faulty_commit"] not in (fixture / "deployment.yaml").read_text()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_commit_messages_do_not_leak_the_answer(project, name):
    truth = setup_demo.build(SCENARIOS[name], root=project)
    for commit in truth["commits"]:
        assert not any(word in commit["message"].lower() for word in LEAKY_WORDS), commit["message"]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_symptoms_do_not_state_the_magnitude_or_cause(project, name):
    truth = setup_demo.build(SCENARIOS[name], root=project)
    for symptom in truth["symptoms"]:
        assert "{month}" not in symptom and "100" not in symptom
        assert not any(w in symptom.lower() for w in ("cents", "dollar", "fan", "duplicate", "join"))


def test_fanout_inflates_counts_but_not_revenue(project):
    truth = setup_demo.build(SCENARIOS["fanout_case"], root=project)
    assert truth["reported_order_count"] > truth["expected_order_count"]
    assert truth["reported_avg_order_value_usd"] < truth["expected_avg_order_value_usd"]
    assert truth["revenue_unaffected"] is True


def test_currency_inflation_is_exactly_100x(project):
    truth = setup_demo.build(SCENARIOS["currency_case"], root=project)
    assert truth["inflation_factor"] == 100.0
