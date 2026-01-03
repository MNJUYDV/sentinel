#!/bin/bash
# Script to upgrade OpenAI library to fix compatibility issues

echo "Upgrading OpenAI library..."
source venv/bin/activate
pip install --upgrade "openai>=1.12.0"

echo ""
echo "✅ OpenAI library upgraded!"
echo "Restart your server with: uvicorn app:app --reload"
echo ""

