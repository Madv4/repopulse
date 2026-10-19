"""Local JSONL diagnostics; full run tracing belongs to Phase 7."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


class DegradationEvent(BaseModel):
    """A safe reason code with no prompt, source text, or credentials."""

    timestamp: datetime
    event: str = "degradation"
    reason: str


class ValidationEvent(BaseModel):
    """A failed model response, kept only in the ignored local trace directory."""

    timestamp: datetime
    event: str = "validation_failure"
    stage: str
    raw_output: str
    error: str


class TraceWriter:
    """Append safe degradation events to one local run trace."""

    def __init__(self, directory: Path) -> None:
        self.run_id = uuid4().hex
        self.path = directory / f"{self.run_id}.jsonl"

    def degradation(self, reason: str) -> None:
        """Record why the run used a fallback without storing untrusted content."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = DegradationEvent(timestamp=datetime.now(UTC), reason=reason)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(event.model_dump_json() + "\n")

    def validation_failure(self, stage: str, raw_output: str, error: str) -> None:
        """Preserve failed structured output for local diagnosis."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = ValidationEvent(
            timestamp=datetime.now(UTC), stage=stage, raw_output=raw_output, error=error
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(event.model_dump_json() + "\n")
