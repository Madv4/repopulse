"""Recorded, network-free MCP stdio server for the integration test."""

from test_mcp_server import EmptySearcher, RecordedReader

from repopulse.mcp_server import ToolService, create_server

if __name__ == "__main__":
    create_server(ToolService(RecordedReader(), EmptySearcher)).run(transport="stdio")
