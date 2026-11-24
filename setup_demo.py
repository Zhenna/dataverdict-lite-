"""Build the currency_case fixture: a git repo, a DuckDB warehouse and a deployment manifest.

Run from the project root:  python setup_demo.py

Inputs : seeds/raw_customers.csv, seeds/raw_orders.csv, seeds/raw_payments.csv
Outputs: fixtures/currency_case/   (git-ignored; safe to delete and rebuild)
         evals/ground_truth/currency_case.json   (answer key, kept OUTSIDE the fixture)
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent
SEEDS = ROOT / "seeds"
SCENARIO = "currency_case"
FIXTURE = ROOT / "fixtures" / SCENARIO
REPO = FIXTURE / "pipeline_repo"
WAREHOUSE = FIXTURE / "warehouse.duckdb"
GROUND_TRUTH = ROOT / "evals" / "ground_truth" / f"{SCENARIO}.json"

# Columns this script relies on. If your CSVs differ, the check below fails loudly.
EXPECTED_COLUMNS = {
    "raw_orders": {"id", "user_id", "order_date", "status"},
    "raw_payments": {"id", "order_id", "payment_method", "amount"},
}

SILVER_GOOD = """\
CREATE OR REPLACE TABLE silver.stg_orders AS
SELECT
    id AS order_id,
    user_id AS customer_id,
    CAST(order_date AS DATE) AS order_date,
    status
FROM bronze.raw_orders;

CREATE OR REPLACE TABLE silver.stg_payments AS
SELECT
    id AS payment_id,
    order_id,
    payment_method,
    amount / 100.0 AS amount_usd
FROM bronze.raw_payments;
"""

SILVER_BAD = SILVER_GOOD.replace("amount / 100.0 AS amount_usd", "amount AS amount_usd")

GOLD = """\
CREATE OR REPLACE TABLE gold.monthly_revenue AS
SELECT
    date_trunc('month', o.order_date) AS month,
    SUM(p.amount_usd) AS revenue_usd
FROM silver.stg_payments AS p
JOIN silver.stg_orders AS o ON p.order_id = o.order_id
GROUP BY 1
ORDER BY 1;
"""

# Fixed identities and dates so commit hashes are identical on every run.
GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Pipeline Dev",
    "GIT_AUTHOR_EMAIL": "dev@example.com",
    "GIT_COMMITTER_NAME": "Pipeline Dev",
    "GIT_COMMITTER_EMAIL": "dev@example.com",
}


def git(*args: str, date: str | None = None) -> str:
    env = {**os.environ, **GIT_IDENTITY}
    if date:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    result = subprocess.run(
        ["git", "-C", str(REPO), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def commit(message: str, date: str) -> str:
    git("add", "-A")
    git("commit", "-m", message, date=date)
    return git("rev-parse", "HEAD")


def check_seeds() -> None:
    missing = [n for n in ("raw_customers", "raw_orders", "raw_payments") if not (SEEDS / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing seed files in {SEEDS}: {missing}")


def load_bronze(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA bronze")
    for name in ("raw_customers", "raw_orders", "raw_payments"):
        path = (SEEDS / f"{name}.csv").as_posix()
        con.execute(f"CREATE TABLE bronze.{name} AS SELECT * FROM read_csv_auto('{path}')")
    for table, wanted in EXPECTED_COLUMNS.items():
        got = {row[0] for row in con.execute(f"DESCRIBE bronze.{table}").fetchall()}
        if not wanted <= got:
            raise SystemExit(f"bronze.{table} is missing columns {wanted - got}; found {sorted(got)}")


def run_sql_files(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    for name in ("silver.sql", "gold.sql"):
        con.execute((REPO / name).read_text())


def main() -> None:
    check_seeds()
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    REPO.mkdir(parents=True)

    # 1. Git history: a correct commit, then a faulty one with a neutral message.
    git("init", "-b", "main")
    (REPO / "silver.sql").write_text(SILVER_GOOD)
    (REPO / "gold.sql").write_text(GOLD)
    good = commit("Initialize payment transformations", "2026-01-05T10:00:00+00:00")

    (REPO / "silver.sql").write_text(SILVER_BAD)
    bad = commit("Standardize payment fields", "2026-01-12T10:00:00+00:00")

    # 2. Warehouse built from the deployed (faulty) version of the pipeline.
    con = duckdb.connect(str(WAREHOUSE))
    load_bronze(con)
    run_sql_files(con)

    # 3. Deployment manifest: which commit produced this warehouse.
    manifest = {
        "pipeline_run_id": "run_001",
        "deployed_commit": bad,
        "warehouse_snapshot": WAREHOUSE.name,
    }
    (FIXTURE / "deployment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    # 4. Independent ground truth, computed with plain SQL (no agent involved).
    bronze_cents = con.execute("SELECT SUM(amount) FROM bronze.raw_payments").fetchone()[0]
    silver_total = con.execute("SELECT SUM(amount_usd) FROM silver.stg_payments").fetchone()[0]
    gold_total = con.execute("SELECT SUM(revenue_usd) FROM gold.monthly_revenue").fetchone()[0]
    expected_total = con.execute(
        "SELECT SUM(p.amount) / 100.0 FROM bronze.raw_payments p JOIN bronze.raw_orders o ON p.order_id = o.id"
    ).fetchone()[0]
    first_month = con.execute("SELECT MIN(month) FROM gold.monthly_revenue").fetchone()[0]
    con.close()

    truth = {
        "scenario": SCENARIO,
        "origin_layer": "silver",
        "origin_table": "silver.stg_payments",
        "faulty_commit": bad,
        "last_good_commit": good,
        "affected_table": "gold.monthly_revenue",
        "expected_total_usd": float(expected_total),
        "reported_total_usd": float(gold_total),
        "inflation_factor": round(float(gold_total) / float(expected_total), 2),
    }
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps(truth, indent=2) + "\n")

    print("Fixture built:", FIXTURE)
    print(f"  commits         : good={good[:7]}  faulty={bad[:7]}")
    print(f"  bronze payments : {bronze_cents:,.0f} cents")
    print(f"  silver payments : {silver_total:,.2f} 'USD'")
    print(f"  gold revenue    : {gold_total:,.2f}")
    print(f"  expected revenue: {expected_total:,.2f}")
    print(f"  inflation factor: {truth['inflation_factor']}x")
    print(f"Suggested symptom : Revenue for {first_month:%B %Y} looks about 100x too high. Investigate the root cause.")


if __name__ == "__main__":
    main()
