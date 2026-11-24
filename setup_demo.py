"""Build scenario fixtures: a git repo, a DuckDB warehouse and a deployment manifest per scenario.

Run from the project root:
    python setup_demo.py                 # build every scenario
    python setup_demo.py fanout_case     # build one

Inputs : seeds/raw_customers.csv, seeds/raw_orders.csv, seeds/raw_payments.csv
Outputs: fixtures/<scenario>/            (git-ignored; safe to delete and rebuild)
         evals/ground_truth/<scenario>.json   (answer key, kept OUTSIDE the fixture)
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import duckdb
import yaml

from evals.scenarios import BUILD_ORDER, SCENARIOS, Scenario

ROOT = Path(__file__).resolve().parent

# Columns this script relies on. If your CSVs differ, the check below fails loudly.
EXPECTED_COLUMNS = {
    "raw_orders": {"id", "user_id", "order_date", "status"},
    "raw_payments": {"id", "order_id", "payment_method", "amount"},
}

# Fixed identities and dates so commit hashes are identical on every run.
GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Pipeline Dev",
    "GIT_AUTHOR_EMAIL": "dev@example.com",
    "GIT_COMMITTER_NAME": "Pipeline Dev",
    "GIT_COMMITTER_EMAIL": "dev@example.com",
}


def git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, **GIT_IDENTITY}
    if date:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def make_history(scenario: Scenario, repo: Path) -> tuple[list[str], dict[str, str]]:
    """Create the repo and commits. Returns (commit hashes, files as of the deployed commit)."""
    repo.mkdir(parents=True)
    git(repo, "init", "-b", "main")
    hashes: list[str] = []
    for commit in scenario.commits:
        for name, content in commit.files.items():
            (repo / name).write_text(content)
        git(repo, "add", "-A")
        git(repo, "commit", "-m", commit.message, date=commit.date)
        hashes.append(git(repo, "rev-parse", "HEAD"))

    deployed_files: dict[str, str] = {}
    for commit in scenario.commits[: scenario.deployed_index + 1]:
        deployed_files.update(commit.files)
    return hashes, deployed_files


def load_bronze(con: duckdb.DuckDBPyConnection, seeds: Path) -> None:
    con.execute("CREATE SCHEMA bronze")
    for name in ("raw_customers", "raw_orders", "raw_payments"):
        path = (seeds / f"{name}.csv").as_posix()
        con.execute(f"CREATE TABLE bronze.{name} AS SELECT * FROM read_csv_auto('{path}')")
    for table, wanted in EXPECTED_COLUMNS.items():
        got = {row[0] for row in con.execute(f"DESCRIBE bronze.{table}").fetchall()}
        if not wanted <= got:
            raise SystemExit(f"bronze.{table} is missing columns {wanted - got}; found {sorted(got)}")


def build(scenario: Scenario, root: Path = ROOT) -> dict:
    seeds = root / "seeds"
    missing = [n for n in ("raw_customers", "raw_orders", "raw_payments") if not (seeds / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing seed files in {seeds}: {missing}")

    fixture = root / "fixtures" / scenario.name
    repo = fixture / "pipeline_repo"
    warehouse = fixture / "warehouse.duckdb"
    if fixture.exists():
        shutil.rmtree(fixture)

    hashes, deployed_files = make_history(scenario, repo)
    deployed = hashes[scenario.deployed_index]

    con = duckdb.connect(str(warehouse))
    try:
        load_bronze(con, seeds)
        con.execute("CREATE SCHEMA silver")
        con.execute("CREATE SCHEMA gold")
        for name in BUILD_ORDER:
            if name in deployed_files:
                con.execute(deployed_files[name])
        measures = scenario.measure(con)
        first_month = con.execute(f"SELECT MIN(month) FROM {scenario.affected_table}").fetchone()[0]
    finally:
        con.close()

    manifest = {
        "pipeline_run_id": "run_001",
        "deployed_commit": deployed,
        "warehouse_snapshot": warehouse.name,
    }
    (fixture / "deployment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    month = f"{first_month:%B %Y}"
    truth = {
        "scenario": scenario.name,
        "tier": scenario.tier,
        "origin_layer": scenario.origin_layer,
        "origin_table": scenario.origin_table,
        "affected_table": scenario.affected_table,
        "faulty_commit": hashes[scenario.faulty_index],
        "last_good_commit": hashes[scenario.faulty_index - 1],
        "deployed_commit": deployed,
        "undeployed_commits": hashes[scenario.deployed_index + 1 :],
        "commits": [{"hash": h, "message": c.message} for h, c in zip(hashes, scenario.commits)],
        **measures,
        "symptoms": [s.format(month=month) for s in scenario.symptoms],
    }
    truth_path = root / "evals" / "ground_truth" / f"{scenario.name}.json"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario fixtures.")
    parser.add_argument("scenario", nargs="?", default="all", choices=["all", *SCENARIOS])
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        truth = build(SCENARIOS[name])
        print(f"\n[{name}]  tier {truth['tier']}  fixture: fixtures/{name}")
        print(f"  faulty commit  : {truth['faulty_commit'][:7]}   deployed: {truth['deployed_commit'][:7]}")
        print(f"  origin / affected: {truth['origin_table']} -> {truth['affected_table']}")
        skip = {"scenario", "tier", "origin_layer", "origin_table", "affected_table", "faulty_commit",
                "last_good_commit", "deployed_commit", "undeployed_commits", "commits", "symptoms"}
        for key, value in truth.items():
            if key not in skip:
                print(f"  {key}: {value}")
        print(f"  suggested symptom: {truth['symptoms'][0]}")


if __name__ == "__main__":
    main()
