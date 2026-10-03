#!/usr/bin/env bash
# ==============================================================================
# Installation script for Reasoned Financial Sentiment (CoT)
# Purpose: Sets up a dedicated uv virtual environment (.venv) for GPU training
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME=".venv"
PYTHON_VERSION="3.11"

echo "=== Setting up environment for Reasoned Financial Sentiment ==="

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "Error: 'uv' is required but not installed."
    echo "Install it via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Detect OS and GPU
OS_TYPE=$(uname -s)
ARCH=$(uname -m)
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    HAS_NVIDIA_GPU=true
fi

echo ">>> Detected OS: $OS_TYPE ($ARCH)"
echo ">>> NVIDIA GPU Available: $HAS_NVIDIA_GPU"

# Create virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo ">>> Creating virtual environment '$VENV_NAME' with Python $PYTHON_VERSION..."
    uv venv "$VENV_NAME" --python "$PYTHON_VERSION"
else
    echo ">>> Existing virtual environment '$VENV_NAME' found."
fi

VENV_PYTHON="$SCRIPT_DIR/$VENV_NAME/bin/python"

# Determine PyTorch Index for GPU
if [ "$HAS_NVIDIA_GPU" = true ]; then
    CUDA_VERSION_STRING=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9\.]*\).*/\1/p')
    echo ">>> Detected CUDA Version: $CUDA_VERSION_STRING"
    CUDA_MAJOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f2)

    if [ "$ARCH" = "aarch64" ]; then
        PT_CU_VERSION="cu130"
    else
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            PT_CU_VERSION="cu130"
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            if [ "$CUDA_MINOR" -ge 8 ]; then
                PT_CU_VERSION="cu128"
            elif [ "$CUDA_MINOR" -ge 4 ]; then
                PT_CU_VERSION="cu124"
            elif [ "$CUDA_MINOR" -ge 1 ]; then
                PT_CU_VERSION="cu121"
            else
                PT_CU_VERSION="cu118"
            fi
        else
            PT_CU_VERSION="cu118"
        fi
    fi

    echo ">>> Installing PyTorch with CUDA support ($PT_CU_VERSION)..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        --extra-index-url "https://download.pytorch.org/whl/$PT_CU_VERSION" \
        "torch" "torchvision"
else
    echo ">>> Installing CPU PyTorch..."
    uv pip install -U --python "$VENV_PYTHON" "torch" "torchvision"
fi

# Install dependencies
echo ">>> Installing project dependencies..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "transformers" \
    "trl" \
    "peft" \
    "accelerate" \
    "bitsandbytes" \
    "datasets" \
    "evaluate" \
    "sentencepiece" \
    "pandas" \
    "numpy" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm" \
    "tenacity" \
    "jupyter" \
    "ipykernel"

# Register Jupyter kernel
echo ">>> Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install --user \
    --name "reasoned-financial-sentiment" \
    --display-name "Python (Reasoned Financial Sentiment)"

echo ""
echo "✅ Environment setup complete in $SCRIPT_DIR/$VENV_NAME"
echo "To activate:"
echo "  source $SCRIPT_DIR/$VENV_NAME/bin/activate"
