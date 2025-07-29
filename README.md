# Azure Deep Research MCP Server

An MCP (Model Context Protocol) server that integrates with Azure AI Foundry's O3 Deep Research to provide in-depth research capabilities for coding agents like Claude and Gemini.

## Overview

This MCP server enables AI coding assistants to perform comprehensive research queries using Azure's Deep Research capabilities. The server handles long-running research tasks (up to 30 minutes) asynchronously, returning results as markdown files.

### Key Features

- **Deep Research Integration**: Leverages Azure's O3 Deep Research model for comprehensive analysis
- **Asynchronous Processing**: Non-blocking research that runs in the background
- **MCP Protocol**: Compatible with Claude Desktop and other MCP clients
- **File-based Results**: Research summaries saved as markdown files for easy access
- **Progress Tracking**: Real-time status updates during research

## Prerequisites

- Python 3.9+ (recommended: 3.11 or 3.12)
- Azure subscription with AI Foundry access
- Azure AI Project with Deep Research model deployed
- Bing resource connection configured in your project

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd o3-deep-research-python-mcp-server

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure credentials

# Run the server
python deep_research_server.py
```


### Python 3.13t Users

Python 3.13t has compatibility issues with the cryptography package. Options:

1. **Use Python 3.11/3.12** (Recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Manual Installation**:
   ```bash
   pip install mcp aiohttp python-dotenv
   pip install --no-deps azure-ai-projects azure-ai-agents
   conda install cryptography  # If available
   pip install azure-identity
   ```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
BING_RESOURCE_NAME=your-bing-resource
DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME=o3-deep-research
MODEL_DEPLOYMENT_NAME=gpt-4o
```

### Azure Authentication

Ensure you're logged into Azure:
```bash
az login
```

Your account needs appropriate role assignments in the Azure AI Project.

## Usage

### Running the Server

```bash
# Standard mode
python deep_research_server.py

# With MCP Inspector (for debugging)
mcp dev deep_research_server.py

# Install for Claude Desktop
mcp install deep_research_server.py
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "azure-deep-research": {
    "command": "python",
    "args": ["/path/to/deep_research_server.py"],
    "env": {
      "PROJECT_ENDPOINT": "your-endpoint",
      "BING_RESOURCE_NAME": "your-bing-resource",
      "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME": "o3-deep-research",
      "MODEL_DEPLOYMENT_NAME": "gpt-4o"
    }
  }
}
```

See `claude_desktop_config_example.json` for a complete example.

### Available Tools

1. **deep_research(query: str)** - Performs deep research on a topic
   - Returns a file URI to the research summary
   - Research runs asynchronously in the background
   - Can take up to 30 minutes to complete

2. **check_research_status()** - Check status of ongoing research tasks
   - Returns status of all recent research tasks
   - Shows completion progress and file locations

3. **generate_research_query(topic: str)** - Generate optimized research queries
   - Helps format queries for better research results

## Testing

```bash
# Test MCP server connectivity
python test_mcp_connection.py

# Test deep research functionality
python test_client.py
```

## How It Works

1. **Client Request**: MCP client sends a research query
2. **Immediate Response**: Server returns a status file URI immediately
3. **Background Processing**: Research continues asynchronously
   - Azure agent asks for clarification
   - Server automatically responds with "go ahead"
   - Deep research begins (10-30 minutes)
4. **Result Update**: Status file is updated with final research
5. **Citations Included**: Results include referenced sources

## Troubleshooting

### Common Issues

**Issue**: cryptography build fails  
**Solution**: Use Python 3.11/3.12 or install cryptography from conda

**Issue**: "No module named 'azure.ai.projects'"  
**Solution**: Install with `pip install azure-ai-projects --pre`

**Issue**: Authentication errors  
**Solution**: Run `az login` and check Azure role assignments

**Issue**: Research times out  
**Solution**: Deep research can take up to 30 minutes - this is normal

### Debug Mode

Check server logs:
```bash
# Logs are written to stderr during development
python deep_research_server.py 2>server.log
```

## Project Structure

```
o3-deep-research-python-mcp-server/
├── deep_research_server.py    # Main MCP server implementation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
├── install.sh                # Installation helper script
├── test_client.py            # Integration test
├── test_mcp_connection.py    # Connection test
├── claude_desktop_config_example.json  # Example configuration
└── research_summaries/       # Output directory for research results
```

## Dependencies

- `mcp[cli]` - MCP protocol implementation
- `azure-ai-projects>=1.0.0b12` - Azure AI integration
- `azure-identity` - Azure authentication
- `aiohttp` - Async HTTP client

## Security Considerations

This server is designed for local development use. For production deployment:

1. **Input Validation**: The server validates query length (10-1000 chars) to prevent abuse
2. **File Security**: All output files are created within the `research_summaries` directory
3. **No Sensitive Logging**: Set `DEBUG=1` environment variable to enable detailed logging
4. **Resource Limits**: Each research can take up to 30 minutes - consider implementing rate limiting
5. **Authentication**: The server relies on Azure's DefaultAzureCredential - ensure proper RBAC setup

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please ensure any PRs maintain security best practices.

## Future Enhancements

- Docker support for easier deployment
- Streaming research updates
- Multiple research agents support
- Custom research templates
- Integration with more MCP clients