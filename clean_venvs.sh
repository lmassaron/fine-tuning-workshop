#!/usr/bin/env bash
# ==============================================================================
# Helper script to clean up all virtual environments (.venv) across the workshop
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cleaning workshop virtual environments (.venv) ==="

FOUND=0
while IFS= read -r venv_dir; do
    if [ -d "$venv_dir" ]; then
        echo ">>> Removing: $venv_dir"
        rm -rf "$venv_dir"
        FOUND=$((FOUND + 1))
    fi
done < <(find . -maxdepth 3 -type d -name ".venv" -prune)

if [ "$FOUND" -eq 0 ]; then
    echo "No .venv directories found."
else
    echo ""
    echo "✅ Successfully removed $FOUND virtual environment(s)."
fi
