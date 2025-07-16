# Azure Deep Research MCP Server

An MCP (Model Context Protocol) server that integrates with Azure AI Foundry's O3 Deep Research to perform in-depth research queries. The server saves research summaries as markdown files and returns file URIs to MCP clients.

## Features

- **Deep Research Integration**: Uses Azure AI Foundry's O3 Deep Research model for comprehensive research
- **Async Architecture**: Fully asynchronous implementation for handling concurrent requests
- **File-based Output**: Saves large research summaries to markdown files with unique naming
- **Progress Updates**: Provides real-time progress updates during long-running research operations
- **Citation Support**: Includes URL citations and references in research summaries
- **Error Handling**: Comprehensive error handling with detailed logging

## Prerequisites

- Python 3.9+ (Python 3.11 or 3.12 recommended)
- Azure subscription with access to Azure AI Foundry
- Azure AI Project with Deep Research model deployed
- Bing connection resource configured in your project

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/o3-deep-research-python-mcp-server.git
cd o3-deep-research-python-mcp-server
```

2. Install dependencies:

**Option A: Standard installation (recommended for Python 3.11/3.12)**
```bash
pip install -r requirements.txt
```

**Option B: If you encounter cryptography build errors with Python 3.13t**
```bash
# Install minimal dependencies first
pip install -r requirements-minimal.txt

# Then try installing Azure packages separately
pip install azure-ai-projects azure-identity

# Or use the install script
./install.sh
```

**Option C: Use a virtual environment with Python 3.11/3.12**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

## Configuration

Set the following environment variables in your `.env` file:

- `PROJECT_ENDPOINT`: Your Azure AI Project endpoint (e.g., `https://your-project.services.ai.azure.com/api/projects/your-project`)
- `BING_RESOURCE_NAME`: Name of your Bing connection resource
- `DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME`: Deployment name for Deep Research model (e.g., `o3-deep-research`)
- `MODEL_DEPLOYMENT_NAME`: Deployment name for base LLM (e.g., `gpt-4o`)

## Usage

### Running the Server

**Option 1: Use the run script (recommended)**
```bash
python run_server.py
```
This will check dependencies and environment variables before starting.

**Option 2: Direct execution**
```bash
python deep_research_server.py
```

**Option 3: With MCP CLI**
```bash
# Development mode with inspector
mcp dev deep_research_server.py

# Install for Claude Desktop
mcp install deep_research_server.py --name "Azure Deep Research"
```

### Testing the Server

Run the test client:
```bash
python test_server.py
```

### Using the Deep Research Tool

The server exposes a `deep_research` tool that accepts a query string:

```python
# Example query
result = await deep_research(context, "Latest research on quantum computing advances in 2024")
# Returns: file:///path/to/research_summaries/20240716_123456_quantum_computing_abcd1234.md
```

### Output Format

Research summaries are saved in the `research_summaries/` directory with:
- Timestamp-based naming
- Sanitized query in filename
- Full markdown formatting
- URL citations and references

## Troubleshooting

### Dependency Installation Issues

If you encounter issues with cryptography package on Python 3.13t:

1. **Use Python 3.11 or 3.12**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install from conda**:
   ```bash
   conda install cryptography
   pip install -r requirements.txt
   ```

3. **Use minimal requirements**:
   ```bash
   pip install -r requirements-minimal.txt
   ```

### Environment Variable Issues

Run the diagnostic script:
```bash
python run_server.py
```

This will show which environment variables are missing.

### Azure Authentication Issues

1. Ensure you're logged in: `az login`
2. Check your role assignments in Azure AI Project
3. Verify the PROJECT_ENDPOINT format is correct

## Architecture

- **FastMCP Framework**: High-level MCP server implementation
- **Async Lifespan Management**: Initializes Azure resources once on startup
- **Reusable Agent**: Creates a single agent instance to avoid per-query overhead
- **Non-blocking Polling**: Uses `asyncio.sleep()` for responsive concurrent handling
- **File URI Return**: Returns standard `file://` URIs for local file access

## Error Handling

The server includes comprehensive error handling:
- Azure authentication failures
- API rate limits
- Research run failures
- File system errors
- Timeout protection (configurable)

## Security Considerations

- Uses `DefaultAzureCredential` for secure authentication
- No hardcoded credentials
- Sanitizes filenames to prevent directory traversal
- Validates all inputs

## Limitations

- Deep Research queries can take several minutes to complete
- Each query incurs Azure AI costs
- File storage grows over time (manual cleanup required)
- Requires local file system access for output

## Development

To contribute:
1. Fork the repository
2. Create a feature branch
3. Follow the async patterns established in the codebase
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details