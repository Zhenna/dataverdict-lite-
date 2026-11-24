"""Scenario definitions: commit history, deployed state and answer key for each seeded fault.

This module is never exposed to the agent. Fixtures are built from it by setup_demo.py.
Each scenario's `measure` hook computes the numbers for the answer key and raises AssertionError
if the seeded fault did not actually show up in the warehouse.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

# Files run (in this order, when present) to build the warehouse from the deployed commit.
BUILD_ORDER = ("silver.sql", "silver_customers.sql", "gold.sql")


@dataclass(frozen=True)
class Commit:
    message: str
    date: str
    files: dict[str, str]  # file name -> full content after this commit


@dataclass(frozen=True)
class Scenario:
    name: str
    tier: int
    commits: tuple[Commit, ...]
    faulty_index: int
    deployed_index: int  # commits after this one exist in git but did not produce the warehouse
    origin_layer: str
    origin_table: str
    affected_table: str
    symptoms: tuple[str, ...]  # {month} is filled in from the data; none states the magnitude
    measure: Callable[[Any], dict[str, Any]]


# ----------------------------------------------------------------------------- shared SQL
STG_ORDERS = """\
CREATE OR REPLACE TABLE silver.stg_orders AS
SELECT
    id AS order_id,
    user_id AS customer_id,
    CAST(order_date AS DATE) AS order_date,
    status
FROM bronze.raw_orders;
"""

STG_PAYMENTS = """\
CREATE OR REPLACE TABLE silver.stg_payments AS
SELECT
    id AS payment_id,
    order_id,
    payment_method,
    amount / 100.0 AS amount_usd
FROM bronze.raw_payments;
"""

CUSTOMERS = """\
CREATE OR REPLACE TABLE silver.stg_customers AS
SELECT * FROM bronze.raw_customers;
"""


def _sorted_desc(sql: str) -> str:
    return sql.replace("ORDER BY 1;", "ORDER BY 1 DESC;")


# ----------------------------------------------------------------------------- scenario: currency
CUR_SILVER_GOOD = STG_ORDERS + "\n" + STG_PAYMENTS
CUR_SILVER_BAD = CUR_SILVER_GOOD.replace("amount / 100.0 AS amount_usd", "amount AS amount_usd")

CUR_GOLD = """\
CREATE OR REPLACE TABLE gold.monthly_revenue AS
SELECT
    date_trunc('month', o.order_date) AS month,
    SUM(p.amount_usd) AS revenue_usd
FROM silver.stg_payments AS p
JOIN silver.stg_orders AS o ON p.order_id = o.order_id
GROUP BY 1
ORDER BY 1;
"""
CUR_GOLD_DOC = "-- Monthly revenue from payments, dated by order.\n" + CUR_GOLD
CUR_GOLD_SORTED = _sorted_desc(CUR_GOLD_DOC)


def measure_currency(con: Any) -> dict[str, Any]:
    expected = float(con.execute(
        "SELECT SUM(p.amount) / 100.0 FROM bronze.raw_payments p "
        "JOIN bronze.raw_orders o ON p.order_id = o.id"
    ).fetchone()[0])
    reported = float(con.execute("SELECT SUM(revenue_usd) FROM gold.monthly_revenue").fetchone()[0])
    factor = reported / expected
    if not math.isclose(factor, 100.0, rel_tol=1e-6):
        raise AssertionError(f"currency fault did not manifest: factor={factor}")
    return {
        "expected_total_usd": expected,
        "reported_total_usd": reported,
        "inflation_factor": round(factor, 2),
    }


CURRENCY = Scenario(
    name="currency_case",
    tier=2,
    commits=(
        Commit("Initialize payment transformations", "2026-01-05T10:00:00+00:00",
               {"silver.sql": CUR_SILVER_GOOD, "gold.sql": CUR_GOLD}),
        Commit("Add customer staging table", "2026-01-07T10:00:00+00:00",
               {"silver_customers.sql": CUSTOMERS}),
        Commit("Standardize payment fields", "2026-01-12T10:00:00+00:00",
               {"silver.sql": CUR_SILVER_BAD}),
        Commit("Document gold model", "2026-01-14T10:00:00+00:00",
               {"gold.sql": CUR_GOLD_DOC}),
        Commit("Sort monthly output newest first", "2026-01-20T10:00:00+00:00",
               {"gold.sql": CUR_GOLD_SORTED}),
    ),
    faulty_index=2,
    deployed_index=3,
    origin_layer="silver",
    origin_table="silver.stg_payments",
    affected_table="gold.monthly_revenue",
    symptoms=(
        "{month} revenue looks much higher than expected. Investigate the root cause.",
        "Finance says {month} revenue should be far lower than the revenue report shows. "
        "Investigate the root cause.",
    ),
    measure=measure_currency,
)

# ----------------------------------------------------------------------------- scenario: fan-out
FAN_TOTALS_GOOD = """\
CREATE OR REPLACE TABLE silver.order_totals AS
SELECT
    order_id,
    SUM(amount_usd) AS total_usd
FROM silver.stg_payments
GROUP BY order_id;
"""
# The grain changes from one row per order to one row per payment.
FAN_TOTALS_BAD = """\
CREATE OR REPLACE TABLE silver.order_totals AS
SELECT
    order_id,
    payment_id,
    amount_usd AS total_usd
FROM silver.stg_payments;
"""
FAN_SILVER_GOOD = STG_ORDERS + "\n" + STG_PAYMENTS + "\n" + FAN_TOTALS_GOOD
FAN_SILVER_BAD = STG_ORDERS + "\n" + STG_PAYMENTS + "\n" + FAN_TOTALS_BAD

FAN_GOLD = """\
CREATE OR REPLACE TABLE gold.monthly_orders AS
SELECT
    date_trunc('month', o.order_date) AS month,
    COUNT(*) AS order_count,
    SUM(t.total_usd) AS revenue_usd,
    SUM(t.total_usd) / COUNT(*) AS avg_order_value_usd
FROM silver.stg_orders AS o
JOIN silver.order_totals AS t ON t.order_id = o.order_id
GROUP BY 1
ORDER BY 1;
"""
FAN_GOLD_DOC = "-- Monthly order counts, revenue and average order value.\n" + FAN_GOLD
FAN_GOLD_SORTED = _sorted_desc(FAN_GOLD_DOC)


def measure_fanout(con: Any) -> dict[str, Any]:
    expected_orders = int(con.execute(
        "SELECT COUNT(DISTINCT p.order_id) FROM bronze.raw_payments p "
        "JOIN bronze.raw_orders o ON p.order_id = o.id"
    ).fetchone()[0])
    expected_revenue = float(con.execute(
        "SELECT SUM(p.amount) / 100.0 FROM bronze.raw_payments p "
        "JOIN bronze.raw_orders o ON p.order_id = o.id"
    ).fetchone()[0])
    reported_orders = int(con.execute("SELECT SUM(order_count) FROM gold.monthly_orders").fetchone()[0])
    reported_revenue = float(con.execute("SELECT SUM(revenue_usd) FROM gold.monthly_orders").fetchone()[0])
    if reported_orders <= expected_orders:
        raise AssertionError("fan-out did not manifest: no order has more than one payment")
    if not math.isclose(reported_revenue, expected_revenue, rel_tol=1e-6):
        raise AssertionError("revenue should be unaffected by the fan-out")
    return {
        "expected_order_count": expected_orders,
        "reported_order_count": reported_orders,
        "expected_avg_order_value_usd": round(expected_revenue / expected_orders, 2),
        "reported_avg_order_value_usd": round(reported_revenue / reported_orders, 2),
        "revenue_unaffected": True,
    }


FANOUT = Scenario(
    name="fanout_case",
    tier=1,
    commits=(
        Commit("Initialize order transformations", "2026-01-05T10:00:00+00:00",
               {"silver.sql": FAN_SILVER_GOOD, "gold.sql": FAN_GOLD}),
        Commit("Add customer staging table", "2026-01-07T10:00:00+00:00",
               {"silver_customers.sql": CUSTOMERS}),
        Commit("Keep payment detail on order totals", "2026-01-12T10:00:00+00:00",
               {"silver.sql": FAN_SILVER_BAD}),
        Commit("Document gold model", "2026-01-14T10:00:00+00:00",
               {"gold.sql": FAN_GOLD_DOC}),
        Commit("Sort monthly output newest first", "2026-01-20T10:00:00+00:00",
               {"gold.sql": FAN_GOLD_SORTED}),
    ),
    faulty_index=2,
    deployed_index=3,
    origin_layer="silver",
    origin_table="silver.order_totals",
    affected_table="gold.monthly_orders",
    symptoms=(
        "Average order value for {month} looks much lower than expected. Investigate the root cause.",
        "The order count for {month} in the monthly orders report looks too high. "
        "Investigate the root cause.",
    ),
    measure=measure_fanout,
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (CURRENCY, FANOUT)}
