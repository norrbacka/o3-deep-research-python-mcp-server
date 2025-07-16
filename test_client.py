#!/usr/bin/env python3
"""
Test client for Azure Deep Research MCP Server
"""

import asyncio
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    """Test the deep research server"""
    try:
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",
            args=["deep_research_server.py"],
            env=None,  # Will inherit current environment
        )
        
        print("Connecting to server...", file=sys.stderr)
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                print("Initializing session...", file=sys.stderr)
                await session.initialize()
                print("Session initialized!", file=sys.stderr)
                
                # List available tools
                print("\nListing available tools...", file=sys.stderr)
                tools = await session.list_tools()
                print(f"Tools response type: {type(tools)}", file=sys.stderr)
                if tools:
                    for tool in tools:
                        print(f"  Tool type: {type(tool)}", file=sys.stderr)
                        if hasattr(tool, 'name'):
                            print(f"  - {tool.name}: {tool.description}", file=sys.stderr)
                        else:
                            print(f"  - Tool: {tool}", file=sys.stderr)
                
                # List available prompts
                print("\nListing available prompts...", file=sys.stderr)
                prompts = await session.list_prompts()
                if prompts:
                    for prompt in prompts:
                        if hasattr(prompt, 'name'):
                            print(f"  - {prompt.name}: {prompt.description}", file=sys.stderr)
                        else:
                            print(f"  - Prompt: {prompt}", file=sys.stderr)
                
                # Test the deep research tool
                print("\nTesting deep research tool...", file=sys.stderr)
                query = "Provide a comprehensive overview of the latest advances in quantum computing in 2024, including key breakthroughs, applications, and leading research institutions."
                print(f"Query: '{query}'", file=sys.stderr)
                
                result = await session.call_tool(
                    "deep_research", 
                    arguments={"query": query}
                )
                
                print(f"\nResult: {result}", file=sys.stderr)
                
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)