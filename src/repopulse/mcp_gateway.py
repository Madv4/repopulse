"""MCP session adapter used by the production agent CLI."""

import asyncio
import json
from typing import Any

from mcp import ClientSession

from repopulse.mcp_server import ToolEvidence


class MCPGateway:
    """Call the local MCP server and validate each structured evidence result."""

    def __init__(self, session: ClientSession, *, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive")
        self.session = session
        self.timeout_seconds = timeout_seconds

    async def call(self, name: str, arguments: dict[str, str | int]) -> ToolEvidence:
        """Return one validated MCP tool result or raise its safe error."""
        result = await asyncio.wait_for(
            self.session.call_tool(name, arguments=arguments),
            timeout=self.timeout_seconds,
        )
        if result.isError:
            raise RuntimeError(f"MCP tool {name} returned an error")
        structured: Any = result.structuredContent
        if isinstance(structured, dict):
            if "result" in structured and isinstance(structured["result"], dict):
                structured = structured["result"]
            return ToolEvidence.model_validate(structured)
        for item in result.content:
            if item.type == "text":
                return ToolEvidence.model_validate(json.loads(item.text))
        raise ValueError(f"MCP tool {name} returned no structured evidence")
