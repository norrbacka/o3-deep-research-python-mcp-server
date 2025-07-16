#!/usr/bin/env python3
"""Simple test to verify MCP server starts and responds."""

import asyncio
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def test_connection():
    """Test basic connection to the MCP server."""
    server_params = StdioServerParameters(
        command="python",
        args=["deep_research_server.py"],
        env={
            "PROJECT_ENDPOINT": os.environ.get("PROJECT_ENDPOINT"),
            "BING_RESOURCE_NAME": os.environ.get("BING_RESOURCE_NAME"),
            "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME": os.environ.get("DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"),
            "MODEL_DEPLOYMENT_NAME": os.environ.get("MODEL_DEPLOYMENT_NAME"),
        }
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                print("✅ Successfully connected to Azure Deep Research MCP Server")
                
                # List available tools
                tools = await session.list_tools()
                print(f"\n📋 Available tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # List available prompts
                prompts = await session.list_prompts()
                if prompts:
                    print(f"\n💡 Available prompts:")
                    for prompt in prompts:
                        print(f"  - {prompt.name}: {prompt.description}")
                
                print("\n✅ Server is ready to use!")
                return True
                
    except Exception as e:
        print(f"❌ Error connecting to server: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check environment variables
    required_vars = [
        "PROJECT_ENDPOINT",
        "BING_RESOURCE_NAME",
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME",
        "MODEL_DEPLOYMENT_NAME"
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        sys.exit(1)
    
    print("Azure Deep Research MCP Server Connection Test")
    print("=" * 50)
    
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)