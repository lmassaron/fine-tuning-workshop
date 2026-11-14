#!/usr/bin/env sh
# ==============================================================================
# Helper script to clean up all virtual environments (.venv) across the workshop
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cleaning workshop virtual environments (.venv) ==="

VENVS=$(find . -maxdepth 3 -type d -name ".venv" -prune 2>/dev/null || true)

if [ -z "$VENVS" ]; then
    echo "No .venv directories found."
    exit 0
fi

FOUND=0
for venv_dir in $VENVS; do
    if [ -d "$venv_dir" ]; then
        echo ">>> Removing: $venv_dir"
        rm -rf "$venv_dir"
        FOUND=$((FOUND + 1))
    fi
done

echo ""
echo "✅ Successfully removed $FOUND virtual environment(s)."
