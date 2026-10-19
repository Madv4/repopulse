"""Conservative, source-local support check for extractive claims."""

import re
import unicodedata

from repopulse.mcp_server import Evidence


def supports_text(text: str, evidence: Evidence) -> bool:
    """Accept a claim only when its full text occurs in the cited source."""

    def normalized(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()

    wanted = normalized(text)
    return bool(wanted) and any(
        wanted in normalized(source) for source in (evidence.title, evidence.excerpt)
    )
