#!/usr/bin/env python3
"""Test if MCP server can run without Azure packages."""

from mcp.server.fastmcp import FastMCP

# Create a minimal MCP server
mcp = FastMCP("TestServer")

@mcp.tool()
def test_tool(message: str) -> str:
    """Test tool that just echoes a message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    print("MCP is installed correctly!")
    print("Starting test server...")
    mcp.run(transport="stdio")