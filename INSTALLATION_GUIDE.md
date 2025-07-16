# Installation Guide for Azure Deep Research MCP Server

This guide helps you install the Azure Deep Research MCP Server, especially if you encounter issues with Python 3.13t and the cryptography package.

## Quick Start (If you have Python 3.11 or 3.12)

```bash
# Clone the repository
git clone <repository-url>
cd o3-deep-research-python-mcp-server

# Install all dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure credentials

# Run the server
python deep_research_server.py
```

## For Python 3.13t Users

Python 3.13t (free-threaded build) has compatibility issues with the cryptography package required by Azure Identity. Here are your options:

### Option 1: Use a Different Python Version (Recommended)

```bash
# Create a virtual environment with Python 3.11 or 3.12
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Manual Azure Package Installation

Due to the cryptography build issues, you might need to install Azure packages through conda or system packages:

```bash
# Install what we can with pip
pip install mcp aiohttp python-dotenv

# Try installing Azure packages
pip install --no-deps azure-ai-projects azure-ai-agents azure-core

# If that fails, try with conda
conda install cryptography
pip install azure-identity azure-ai-projects
```

### Option 3: Use Docker (Future Enhancement)

A Docker image will be provided in future updates to avoid dependency issues entirely.

## Verify Installation

Run the diagnostic script to check your installation:

```bash
python run_server.py
```

This will show:
- ✓ Which packages are installed
- ❌ Which packages are missing
- ❌ Which environment variables need to be set

## Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Azure credentials:
   ```
   PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
   BING_RESOURCE_NAME=your-bing-resource
   DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME=o3-deep-research
   MODEL_DEPLOYMENT_NAME=gpt-4o
   ```

3. Ensure you're logged into Azure:
   ```bash
   az login
   ```

## Common Issues and Solutions

### Issue: cryptography build fails
**Solution**: Use Python 3.11/3.12 or install cryptography from conda

### Issue: "No module named 'azure.ai.projects'"
**Solution**: Install with `pip install azure-ai-projects --pre`

### Issue: Authentication errors
**Solution**: Run `az login` and check your Azure role assignments

### Issue: Missing environment variables
**Solution**: Run `python run_server.py` to see which variables are missing

## Testing the Installation

1. Test MCP is working:
   ```bash
   python test_mcp_only.py
   ```

2. Test the full server (requires Azure packages):
   ```bash
   python test_server.py
   ```

## Next Steps

Once installed:
1. Run the server: `python deep_research_server.py`
2. Or use with MCP Inspector: `mcp dev deep_research_server.py`
3. Or install for Claude Desktop: `mcp install deep_research_server.py`

For more details, see README_FULL.md