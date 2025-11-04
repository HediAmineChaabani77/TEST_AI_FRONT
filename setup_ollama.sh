#!/bin/bash
# Quick setup script for Ollama with qwen2:0.5b model

echo "=================================="
echo "Ollama Setup for AI Invoice Generator"
echo "=================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found!"
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama already installed"
fi

# Pull qwen2:0.5b model (lightweight, fast)
echo ""
echo "📥 Downloading qwen2:0.5b model (lightweight ~350MB)..."
ollama pull qwen2:0.5b

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To run the app:"
echo "  python3 app.py"
echo ""
echo "The app will use:"
echo "  - Regex method (fast, always available)"
echo "  - AI method (qwen2:0.5b, smart extraction)"
echo ""

