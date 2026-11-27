"""External plugin interfaces (MCP stdio server) so other agents can mount SceneFactory cleanly."""

from .mcp_server import MCPServer, run_stdio_server

__all__ = ["MCPServer", "run_stdio_server"]
