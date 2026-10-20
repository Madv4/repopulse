"""Local, per-run JSONL event log with best-effort secret redaction."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from repopulse.guardrails import redact

RUN_ID = re.compile(r"[0-9a-f]{32}")


def _safe(value: Any) -> Any:
    """Redact strings recursively before any event reaches disk."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


class TraceWriter:
    """Append events for one run; traces are local and ignored by Git."""

    def __init__(self, directory: Path) -> None:
        self.run_id = uuid4().hex
        self.path = directory / f"{self.run_id}.jsonl"

    def record(self, event: str, **fields: Any) -> None:
        """Write one timestamped event after recursively redacting string fields."""
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            **_safe(fields),
            "run_id": self.run_id,
            "event": event,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def degradation(self, reason: str) -> None:
        """Record why the run used a fallback."""
        self.record("degradation", reason=reason)

    def validation_failure(self, stage: str, raw_output: str, error: str) -> None:
        """Preserve a failed model response for local diagnosis."""
        self.record("validation_failure", stage=stage, raw_output=raw_output, error=error)

    def quarantine(self, evidence_id: str, reasons: list[str]) -> None:
        """Record exclusion of a suspicious retrieved passage."""
        self.record("quarantine", evidence_id=evidence_id, reasons=reasons)

    def claim_rejection(
        self, claim: str, quote: str, evidence_id: str, evidence_text: str, reason: str
    ) -> None:
        """Record a failed claim with the cited text, redacting common secrets."""
        self.record(
            "claim_rejection",
            claim=claim,
            quote=quote,
            evidence_id=evidence_id,
            evidence_text=evidence_text,
            reason=reason,
        )


def load_trace(directory: Path, run_id: str) -> list[dict[str, Any]]:
    """Read one trace after strict run-ID validation prevents path traversal."""
    if RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be a 32-character lowercase hexadecimal ID")
    path = directory / f"{run_id}.jsonl"
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("trace path resolves outside the trace directory")
    if not path.is_file():
        raise FileNotFoundError(f"Trace {run_id} was not found in {directory}")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Trace contains a non-object event")
            if event.get("run_id", run_id) != run_id:
                raise ValueError("Trace contains an event from another run")
            events.append(event)
    return events
