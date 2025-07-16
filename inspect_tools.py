#!/usr/bin/env python3
"""Inspect Azure SDK tool structures"""

import inspect
from azure.ai.agents.models import BingGroundingTool
import json

# Check BingGroundingTool
print('BingGroundingTool parameters:')
sig = inspect.signature(BingGroundingTool.__init__)
for param, details in sig.parameters.items():
    if param != 'self':
        print(f'  {param}: {details}')

# Try to create one
try:
    tool = BingGroundingTool(connection_id='test')
    print(f'\nTool type: {tool}')
    
    # Try to serialize it
    if hasattr(tool, 'as_dict'):
        print(f'Tool as_dict: {json.dumps(tool.as_dict(), indent=2)}')
    
    # Check if it's a dict-like object
    if hasattr(tool, '__dict__'):
        print(f'Tool __dict__: {tool.__dict__}')
        
    # Check the type attribute
    if hasattr(tool, 'type'):
        print(f'Tool type attr: {tool.type}')
        
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()