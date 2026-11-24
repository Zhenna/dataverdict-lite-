"""Agent tools with built-in guardrails.

Every tool returns a JSON-serializable dict. Failures are returned as {"error": "..."} instead of
raised, so the model can read the message and recover.

Guardrail layers for SQL (none of them is trusted alone):
  1. DuckDB opens the warehouse read-only with external access disabled and config locked.
  2. The query must parse as exactly one SELECT statement (DuckDB's own parser).
  3. A deny-list rejects file/network-style table functions.
  4. The query is wrapped in a row limit and interrupted after a timeout.

Guardrails for git: arguments are passed as a list (no shell), commits must match a hex pattern
and exist, and paths must resolve inside the scenario's pipeline_repo.
"""

import datetime as dt
import re
import subprocess
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import yaml
from pydantic import ValidationError

from schemas import Verdict

MAX_ROWS = 50
QUERY_TIMEOUT_S = 10
MAX_DIFF_CHARS = 6000
MAX_LOG_ENTRIES = 20

TABLE_RE = re.compile(r"^(bronze|silver|gold)\.[a-z_][a-z0-9_]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
DENY_RE = re.compile(
    r"\b(read_\w+|\w+_scan|glob|sniff_csv|query|query_table)\s*\(",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


class Toolbox:
    def __init__(self, fixture_dir: str | Path):
        self.fixture = Path(fixture_dir).resolve()
        self.repo = (self.fixture / "pipeline_repo").resolve()
        self.warehouse = self.fixture / "warehouse.duckdb"
        self._con: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_tables": self.list_tables,
            "describe_table": self.describe_table,
            "run_query": self.run_query,
            "get_deployment_info": self.get_deployment_info,
            "git_log": self.git_log,
            "git_diff": self.git_diff,
            "submit_verdict": self.submit_verdict,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool '{name}'. Available: {sorted(handlers)}"}
        try:
            return handler(**(args or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # last-resort guard; the message goes back to the model
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ warehouse
    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(
                str(self.warehouse),
                read_only=True,
                config={
                    "enable_external_access": False,
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                },
            )
            self._con.execute("SET lock_configuration = true")
        return self._con

    def list_tables(self) -> dict[str, Any]:
        rows = self.con.execute(
            "SELECT table_schema || '.' || table_name FROM information_schema.tables "
            "WHERE table_schema IN ('bronze', 'silver', 'gold') ORDER BY 1"
        ).fetchall()
        return {"tables": [r[0] for r in rows]}

    def describe_table(self, table: str) -> dict[str, Any]:
        if not isinstance(table, str) or not TABLE_RE.match(table):
            return {"error": "table must look like 'silver.stg_payments' (bronze, silver or gold schema)."}
        schema, name = table.split(".")
        rows = self.con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [schema, name],
        ).fetchall()
        if not rows:
            return {"error": f"Unknown table '{table}'. Call list_tables."}
        return {"table": table, "columns": [{"name": r[0], "type": r[1]} for r in rows]}

    def run_query(self, sql: str) -> dict[str, Any]:
        if not isinstance(sql, str) or not sql.strip():
            return {"error": "sql must be a non-empty string."}
        cleaned = sql.strip().rstrip(";").strip()

        try:
            statements = duckdb.extract_statements(cleaned)
        except Exception as exc:
            return {"error": f"Could not parse SQL: {exc}"}
        if len(statements) != 1:
            return {"error": "Exactly one statement is allowed."}
        if statements[0].type != duckdb.StatementType.SELECT:
            return {"error": "Only SELECT statements are allowed."}
        if DENY_RE.search(cleaned):
            return {"error": "File, network and table-function access is not allowed."}

        timer = threading.Timer(QUERY_TIMEOUT_S, self.con.interrupt)
        timer.start()
        try:
            cursor = self.con.execute(f"SELECT * FROM ({cleaned}) AS q LIMIT {MAX_ROWS + 1}")
            columns = [d[0] for d in cursor.description]
            fetched = cursor.fetchall()
        except duckdb.InterruptException:
            return {"error": f"Query exceeded {QUERY_TIMEOUT_S}s and was cancelled."}
        except Exception as exc:
            return {"error": f"Query failed: {exc}"}
        finally:
            timer.cancel()

        truncated = len(fetched) > MAX_ROWS
        rows = [[_jsonable(v) for v in row] for row in fetched[:MAX_ROWS]]
        return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}

    # ------------------------------------------------------------------ deployment and git
    def get_deployment_info(self) -> dict[str, Any]:
        return yaml.safe_load((self.fixture / "deployment.yaml").read_text())

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo), "--no-pager", *args],
            capture_output=True,
            text=True,
        )

    def _safe_repo_path(self, path: str) -> str:
        if not isinstance(path, str) or not path or path.startswith("-") or "\x00" in path:
            raise ValueError("path must be a non-empty relative file path.")
        resolved = (self.repo / path).resolve()
        try:
            relative = resolved.relative_to(self.repo)
        except ValueError:
            raise ValueError("path must stay inside the pipeline repository.")
        if ".git" in relative.parts:
            raise ValueError("path must not point into .git.")
        return relative.as_posix()

    def _deployed_commit(self) -> str | None:
        try:
            return self.get_deployment_info().get("deployed_commit")
        except Exception:
            return None

    def git_log(self, path: str | None = None) -> dict[str, Any]:
        args = ["log", f"--max-count={MAX_LOG_ENTRIES}", "--date=short", "--format=%H%x1f%ad%x1f%s"]
        if path:
            try:
                args += ["--", self._safe_repo_path(path)]
            except ValueError as exc:
                return {"error": str(exc)}
        result = self._git(*args)
        if result.returncode != 0:
            return {"error": result.stderr.strip() or "git log failed"}

        deployed = self._deployed_commit()
        commits = []
        for line in result.stdout.splitlines():
            sha, date, subject = line.split("\x1f", 2)
            in_deployed = None
            if deployed:
                in_deployed = self._git("merge-base", "--is-ancestor", sha, deployed).returncode == 0
            commits.append({"commit": sha, "date": date, "subject": subject, "in_deployed_history": in_deployed})
        return {"path": path, "commits": commits}

    def git_diff(self, commit: str) -> dict[str, Any]:
        if not isinstance(commit, str) or not COMMIT_RE.match(commit):
            return {"error": "commit must be a 7-40 character lowercase hex hash from git_log."}
        verified = self._git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
        if verified.returncode != 0:
            return {"error": f"Unknown commit '{commit}'."}
        full = verified.stdout.strip()

        shown = self._git(
            "show", "--no-ext-diff", "--no-textconv", "--no-color",
            "--date=short", "--format=%H%n%ad%n%s", full, "--",
        )
        if shown.returncode != 0:
            return {"error": shown.stderr.strip() or "git show failed"}
        sha, date, subject, diff = (shown.stdout.split("\n", 3) + ["", "", "", ""])[:4]
        truncated = len(diff) > MAX_DIFF_CHARS
        return {
            "commit": sha,
            "date": date,
            "subject": subject,
            "diff": diff[:MAX_DIFF_CHARS],
            "truncated": truncated,
        }

    # ------------------------------------------------------------------ verdict
    def submit_verdict(self, **fields: Any) -> dict[str, Any]:
        if fields.get("status") == "budget_exhausted":
            return {"error": "budget_exhausted is set by the system, not by the agent."}
        try:
            verdict = Verdict(**fields)
        except ValidationError as exc:
            return {"error": "Invalid verdict", "details": exc.errors(include_url=False, include_context=False)}
        return {"accepted": True, "verdict": verdict.model_dump()}


# ---------------------------------------------------------------------- provider-neutral specs
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_tables",
        "description": "List the tables available in the warehouse as schema.table names.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_table",
        "description": "Return column names and types for one table, e.g. 'silver.stg_payments'.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}},
            "required": ["table"],
        },
    },
    {
        "name": "run_query",
        "description": (
            f"Run one read-only SELECT against the warehouse. At most {MAX_ROWS} rows are returned. "
            "Use it for counts, sums, null rates and tracing specific rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "get_deployment_info",
        "description": "Return which pipeline commit produced the current warehouse snapshot.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "git_log",
        "description": (
            "List recent commits in the pipeline repository, optionally for one file path "
            "(relative to the repo root). Each commit says whether it is in the deployed history."
        ),
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "git_diff",
        "description": "Show the change introduced by one commit (hash taken from git_log).",
        "parameters": {
            "type": "object",
            "properties": {"commit": {"type": "string"}},
            "required": ["commit"],
        },
    },
    {
        "name": "submit_verdict",
        "description": (
            "Submit the final diagnosis and end the investigation. Cite evidence by tool call ID. "
            "Use status 'inconclusive' if the evidence is insufficient."
        ),
        "parameters": Verdict.model_json_schema(),
    },
]
