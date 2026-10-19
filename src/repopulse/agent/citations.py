"""Verify a short verbatim quote against one retrieved source passage."""

import re
import unicodedata

from repopulse.mcp_server import Evidence


def supports_text(text: str, evidence: Evidence) -> bool:
    """Conservative exact-text check retained for long-term memory promotion."""

    def normalized(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()

    wanted = normalized(text)
    return bool(wanted) and any(
        wanted in normalized(source) for source in (evidence.title, evidence.excerpt)
    )


def quote_in_evidence(quote: str, evidence: Evidence) -> bool:
    """Require a case-sensitive verbatim span of at most 30 words."""
    if not quote.strip() or len(quote.split()) > 30:
        return False
    return any(quote in source for source in (evidence.title, evidence.excerpt))
