#!/usr/bin/env python3
"""
Runner script for Azure Deep Research MCP Server with dependency checking.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    # Check MCP
    try:
        import mcp
        print("✓ MCP package installed")
    except ImportError:
        missing.append("mcp")
    
    # Check Azure packages
    try:
        import azure.ai.projects
        print("✓ Azure AI Projects package installed")
    except ImportError:
        missing.append("azure-ai-projects")
    
    try:
        import azure.identity
        print("✓ Azure Identity package installed")
    except ImportError:
        missing.append("azure-identity")
    
    try:
        import aiohttp
        print("✓ aiohttp package installed")
    except ImportError:
        missing.append("aiohttp")
    
    return missing

def check_env_vars():
    """Check if required environment variables are set."""
    required_vars = [
        "PROJECT_ENDPOINT",
        "BING_RESOURCE_NAME",
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME",
        "MODEL_DEPLOYMENT_NAME"
    ]
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file found")
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ .env file loaded")
        except ImportError:
            print("! python-dotenv not installed, .env file not loaded")
            print("  Run: pip install python-dotenv")
    else:
        print("! .env file not found. Copy .env.example to .env and fill in your values.")
    
    missing_vars = []
    for var in required_vars:
        if os.environ.get(var):
            print(f"✓ {var} is set")
        else:
            missing_vars.append(var)
    
    return missing_vars

def main():
    """Main runner function."""
    print("Azure Deep Research MCP Server - Dependency Check")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\nTo install missing dependencies:")
        print("  Option 1: pip install " + " ".join(missing_deps))
        print("  Option 2: Run ./install.sh")
        print("\nNote: If you get cryptography build errors with Python 3.13t,")
        print("consider using Python 3.11 or 3.12 instead.")
        return 1
    
    print("\n" + "=" * 50)
    print("Environment Variable Check")
    print("=" * 50)
    
    # Check environment variables
    missing_vars = check_env_vars()
    
    if missing_vars:
        print("\n❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file.")
        return 1
    
    print("\n" + "=" * 50)
    print("✅ All checks passed! Starting server...")
    print("=" * 50 + "\n")
    
    # Run the actual server
    try:
        subprocess.run([sys.executable, "deep_research_server.py"])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"\nError running server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())