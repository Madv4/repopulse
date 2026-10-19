"""Local JSONL diagnostics; full run tracing belongs to Phase 7."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from repopulse.guardrails import redact


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


class QuarantineEvent(BaseModel):
    """Metadata about excluded source text, without copying its payload."""

    timestamp: datetime
    event: str = "quarantine"
    evidence_id: str
    reasons: list[str]


class ClaimRejectionEvent(BaseModel):
    """Local evidence for why a generated claim failed quote grounding."""

    timestamp: datetime
    event: str = "claim_rejection"
    claim: str
    quote: str
    evidence_id: str
    evidence_text: str
    reason: str


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
            timestamp=datetime.now(UTC),
            stage=stage,
            raw_output=redact(raw_output),
            error=redact(error),
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(event.model_dump_json() + "\n")

    def quarantine(self, evidence_id: str, reasons: list[str]) -> None:
        """Record exclusion of a suspicious retrieved passage."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = QuarantineEvent(
            timestamp=datetime.now(UTC), evidence_id=evidence_id, reasons=reasons
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(event.model_dump_json() + "\n")

    def claim_rejection(
        self, claim: str, quote: str, evidence_id: str, evidence_text: str, reason: str
    ) -> None:
        """Record a failed claim with cited text, redacting secrets and contact details."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = ClaimRejectionEvent(
            timestamp=datetime.now(UTC),
            claim=redact(claim),
            quote=redact(quote),
            evidence_id=evidence_id,
            evidence_text=redact(evidence_text),
            reason=reason,
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(event.model_dump_json() + "\n")
