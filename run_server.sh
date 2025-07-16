#!/bin/bash
# Wrapper script for running the server from WSL

# Source asdf to ensure Python is available
source /home/norrbacka/.asdf/asdf.sh

# Change to the project directory
cd /home/norrbacka/repos/o3-deep-research-python-mcp-server

# Run the server (redirect stderr to a log file for debugging)
exec python deep_research_server.py 2>/tmp/azure_deep_research_mcp.log