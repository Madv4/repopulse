"""Private local JSON cache for GitHub GET responses."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from repopulse.github.models import CacheEntry


class DiskCache:
    """Store each URL and credential scope in a separate atomic JSON file."""

    def __init__(self, directory: Path, token: str | None = None) -> None:
        self.directory = directory
        self._scope = hashlib.sha256(token.encode()).hexdigest() if token else "public"

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(f"{self._scope}\n{url}".encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def read(self, url: str) -> CacheEntry | None:
        """Return a valid cached entry or miss on absent/corrupt JSON."""
        try:
            return CacheEntry.model_validate_json(self._path(url).read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            return None

    def write(self, url: str, entry: CacheEntry) -> None:
        """Atomically replace one cache entry without storing the token."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(url)
        temporary = self.directory / f".{target.stem}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(entry.model_dump_json(), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
