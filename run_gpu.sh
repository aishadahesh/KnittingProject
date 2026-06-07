#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH=$(find "$PROJECT_DIR/.venv" -name "*.so*" 2>/dev/null | xargs -n 1 dirname | sort -u | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/imgui_app.py" "$@"
