"""Structured verdict returned by the agent through the submit_verdict tool."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Observation(BaseModel):
    """One concrete value read from a tool result, so a validator can check it mechanically."""

    row: int = Field(description="Row index in the cited tool result (0-based).")
    column: str = Field(description="Column name in the cited tool result.")
    value: int | float | str = Field(description="The value exactly as returned by the tool.")


class Evidence(BaseModel):
    tool_call_id: str = Field(description="ID of the tool call that produced this evidence.")
    finding: str = Field(description="One-sentence plain-language finding.")
    observations: list[Observation] = Field(default_factory=list)


class Verdict(BaseModel):
    status: Literal["root_cause_found", "no_issue_found", "inconclusive", "budget_exhausted"]
    affected_table: str | None = None
    origin_layer: Literal["bronze", "silver", "gold"] | None = None
    origin_table: str | None = None
    suspect_commit: str | None = Field(default=None, description="Commit hash from git_log or git_diff.")
    attribution_level: Literal["suspected", "supported"] | None = Field(
        default=None,
        description=(
            "suspected: the relevant file changed. "
            "supported: the diff explains the observed discrepancy. "
            "Null when there is no suspect commit."
        ),
    )
    root_cause: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_fix: str | None = None
    confidence: Literal["low", "medium", "high"] = Field(
        description="Self-assessed confidence. This is not a calibrated probability."
    )

    @model_validator(mode="after")
    def check_consistency(self) -> "Verdict":
        if self.status == "root_cause_found":
            missing = [
                name
                for name in ("origin_layer", "origin_table", "root_cause")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"root_cause_found requires: {', '.join(missing)}")
            if not self.evidence:
                raise ValueError("root_cause_found requires at least one evidence item")
        if self.suspect_commit and self.attribution_level is None:
            raise ValueError("attribution_level is required when suspect_commit is set")
        if not self.suspect_commit and self.attribution_level is not None:
            raise ValueError("attribution_level must be null when there is no suspect_commit")
        return self
