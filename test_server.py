#!/usr/bin/env python3
"""
Test script for Azure Deep Research MCP Server

This script demonstrates how to connect to the server and call the deep_research tool.
"""

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    """Test the deep research server with a sample query."""
    # Define server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=["deep_research_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            print("Connected to Azure Deep Research MCP Server")
            
            # List available tools
            tools = await session.list_tools()
            print(f"\nAvailable tools: {[tool.name for tool in tools]}")
            
            # List available prompts
            prompts = await session.list_prompts()
            print(f"Available prompts: {[prompt.name for prompt in prompts]}")
            
            # Call the deep_research tool
            print("\nPerforming deep research query...")
            query = "Latest breakthroughs in quantum computing error correction techniques in 2024"
            
            try:
                result = await session.call_tool(
                    "deep_research",
                    arguments={"query": query}
                )
                print(f"\nResearch complete! Summary saved to: {result}")
            except Exception as e:
                print(f"Error during research: {e}")


if __name__ == "__main__":
    print("Azure Deep Research MCP Server Test")
    print("=" * 50)
    print("Note: Ensure you have set up your .env file with Azure credentials")
    print("=" * 50)
    
    asyncio.run(main())