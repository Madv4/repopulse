"""Configurable boundaries around repository questions and untrusted evidence."""

import re

from pydantic import BaseModel, Field

from repopulse.mcp_server import Evidence


class Guardrails(BaseModel):
    """Individual switches for Phase 6 ablations; all defenses default on."""

    input_validation: bool = True
    injection_detection: bool = True
    untrusted_wrapping: bool = True
    output_redaction: bool = True
    structured_validation: bool = True
    citation_verification: bool = True
    action_gate: bool = True


class ActionPlan(BaseModel):
    """A write request is represented as an inert dry-run proposal."""

    requested_action: str
    repo: str
    dry_run: bool = True
    requires_confirmation: bool = True
    status: str = "confirmation_required"


class Detection(BaseModel):
    suspicious: bool
    reasons: list[str] = Field(default_factory=list)


_PATTERNS: dict[str, str] = {
    "instruction_override": (
        r"(?i)(ignore|disregard|override|forget)\s+(all\s+)?"
        r"(previous|prior|system|developer|user)\s+(instructions?|prompts?|rules?)"
    ),
    "fake_role": (
        r"(?im)(\[/?(system|developer|assistant)\]|<\|im_start\|>|"
        r"<\|system\|>|^\s*(system|developer)\s*:)"
    ),
    "exfiltration": (
        r"(?i)(send|post|upload|exfiltrate|leak)\b.{0,100}"
        r"(token|secret|key|credential|\.env)\b"
    ),
    "tool_hijack": (
        r"(?i)(call|invoke|run|execute)\s+(the\s+)?"
        r"(tool|function|shell|command|terminal)\b"
    ),
    "markdown_link": r"(?i)\[.{0,100}\]\(https?://[^)]+(?:token|secret|key|\.env)[^)]*\)",
    "directive": (
        r"(?i)(you are now|new instructions:|answer with exactly|"
        r"must obey this|do not tell the user)"
    ),
}


def validate_input(question: str, repo: str) -> None:
    """Keep questions bounded and repository scope unambiguous."""
    if not 1 <= len(question.strip()) <= 1000:
        raise ValueError("question length must be between 1 and 1000 characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("repo must be one public GitHub owner/name")
    if re.search(r"(?i)\b(private repo|local files?|filesystem|other repositories)\b", question):
        raise ValueError("question exceeds the allowed public repository scope")
    mentioned = re.findall(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", question)
    if any(name.casefold() != repo.casefold() for name in mentioned):
        raise ValueError("question references a GitHub repository outside the allowed scope")


def write_action(question: str, repo: str) -> ActionPlan | None:
    """Detect direct requests to mutate repository state, without executing them."""
    match = re.search(
        r"(?i)\b(create|open|close|reopen|delete|merge|push|edit|update|"
        r"modify|post|write|publish|approve|label|assign|comment on)\b"
        r"\s+(?:a |an |the )?"
        r"(issue|pull request|pr|commit|release|branch|file|comment)\b",
        question,
    )
    return ActionPlan(requested_action=match.group(0), repo=repo) if match else None


def detect_injection(evidence: Evidence) -> Detection:
    """Flag instruction-like text in a retrieved issue, PR, or comment passage."""
    content = f"{evidence.title}\n{evidence.excerpt}"
    reasons = [name for name, pattern in _PATTERNS.items() if re.search(pattern, content)]
    if len(re.findall(r"(?i)\b(prompt|instruction|system|assistant|tool)\b", content)) >= 5:
        reasons.append("instruction_density")
    return Detection(suspicious=bool(reasons), reasons=reasons)


def untrusted_block(evidence: Evidence) -> str:
    """Encode source text so delimiters inside a source cannot close its block."""

    def escaped(value: str) -> str:
        return value.replace("<", "\\u003c")

    return (
        f'<untrusted_data evidence_id="{evidence.evidence_id}">\n'
        f"Evidence ID: {evidence.evidence_id}\n"
        f"URL: {escaped(evidence.url)}\n"
        f"Kind: {escaped(evidence.kind)}\n"
        f"Title:\n{escaped(evidence.title)}\n"
        f"Excerpt:\n{escaped(evidence.excerpt)}\n"
        "</untrusted_data>"
    )


_SECRETS = re.compile(
    r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|gsk_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)")


def redact(text: str) -> str:
    """Best-effort redaction of common credentials and personal contact data."""
    text = _SECRETS.sub("[REDACTED_SECRET]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return _PHONE.sub("[REDACTED_PHONE]", text)
