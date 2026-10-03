#!/usr/bin/env bash
# ==============================================================================
# Master Installation Dispatcher for Fine-Tuning Workshop
# Purpose: Allows setting up environments across any or all example tracks
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRACKS=(
    "knowledge-injection-sherlock"
    "reasoned-financial-sentiment"
    "tunix-med-jax"
    "medical-expert-cardiology"
    "vision-finetuning-latex"
    "alignment-dpo"
    "alignment-grpo"
    "gemma3-function-calling"
    "nemotron-embed-finetuning"
    "code-multiagent"
    "code-multiagent-trl"
)

usage() {
    echo "Usage: ./install.sh [track_name | all]"
    echo ""
    echo "Available tracks:"
    for track in "${TRACKS[@]}"; do
        echo "  - $track"
    done
    echo "  - all (installs all environments sequentially)"
    exit 1
}

TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    echo "=========================================================="
    echo "Fine-Tuning Workshop: Multi-Track Environment Setup"
    echo "=========================================================="
    echo "Please specify a track to install, or pass 'all':"
    select track in "${TRACKS[@]}" "all" "exit"; do
        case "$track" in
            "all")
                TARGET="all"
                break
                ;;
            "exit")
                echo "Exiting."
                exit 0
                ;;
            *)
                if [ -n "$track" ]; then
                    TARGET="$track"
                    break
                else
                    echo "Invalid selection."
                fi
                ;;
        esac
    done
fi

install_track() {
    local dir="$1"
    if [ -d "$dir" ] && [ -f "$dir/install.sh" ]; then
        echo ""
        echo ">>> Installing environment for: $dir..."
        (cd "$dir" && chmod +x install.sh && ./install.sh)
    else
        echo "Error: Directory '$dir' or '$dir/install.sh' not found." >&2
        exit 1
    fi
}

if [ "$TARGET" = "all" ]; then
    echo ">>> Installing all workshop track environments..."
    for track in "${TRACKS[@]}"; do
        install_track "$track"
    done
    echo ""
    echo "✅ All track environments installed successfully!"
else
    install_track "$TARGET"
fi
