#!/usr/bin/env bash
# ==============================================================================
# PRIME Local Author Studio Launcher
# Double-click or run this script to launch the interactive local AI author
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "======================================================================"
echo "👑 Launching PRIME Interactive Local Author Studio..."
echo "💻 Target: AMD Radeon Graphics (ROCm 7.2) | Model: Meta-Llama-3-8B"
echo "🌐 Web Dashboard & Reader: http://localhost:7860/reader"
echo "======================================================================"

exec "$DIR/../venv/bin/python" "$DIR/chat_local_author.py"
