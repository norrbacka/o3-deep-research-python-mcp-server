#!/bin/bash
# Monitor the Azure Deep Research MCP server log

echo "Monitoring Azure Deep Research MCP Server..."
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

# Watch the log file
tail -f /tmp/azure_deep_research_mcp.log | grep -E "Deep research|Still polling|seconds elapsed|Research summary|Error|progress"