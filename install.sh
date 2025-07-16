#!/bin/bash

echo "Installing Azure Deep Research MCP Server dependencies..."
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Try to install dependencies with different strategies
echo -e "\nAttempting installation..."

# Method 1: Try with no-binary for cryptography
echo "Method 1: Installing with pre-built wheels where possible..."
pip install --no-cache-dir mcp[cli] aiohttp azure-core azure-ai-agents 2>/dev/null

# Method 2: Install azure packages separately without cryptography first
echo -e "\nMethod 2: Installing Azure packages..."
pip install --no-deps azure-ai-projects azure-ai-agents 2>/dev/null
pip install --no-deps azure-identity 2>/dev/null

# Method 3: Try minimal installation
echo -e "\nMethod 3: Installing core MCP package..."
pip install mcp 2>/dev/null

# Check what's installed
echo -e "\nChecking installed packages:"
pip list | grep -E "(mcp|azure|aiohttp)" || echo "No relevant packages found"

echo -e "\n=============================================="
echo "Installation attempt complete."
echo ""
echo "If you encounter cryptography build errors with Python 3.13t,"
echo "you may need to:"
echo "1. Use Python 3.11 or 3.12 instead"
echo "2. Or install cryptography from conda: conda install cryptography"
echo "3. Or use a virtual environment with a different Python version"
echo ""
echo "To create a venv with Python 3.11:"
echo "  python3.11 -m venv venv"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"