#!/usr/bin/env bash
# ==============================================================================
# Installation script for Tunix-Med JAX SFT
# Purpose: Sets up a dedicated uv virtual environment (.venv) for JAX/Tunix GPU
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME=".venv"
PYTHON_VERSION="3.11"

echo "=== Setting up environment for Tunix-Med (JAX & Tunix) ==="

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

# Install JAX and CUDA dependencies
if [ "$HAS_NVIDIA_GPU" = true ]; then
    echo ">>> Installing JAX with CUDA acceleration..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "jax[cuda12]"
else
    echo ">>> Installing CPU JAX..."
    uv pip install -U --python "$VENV_PYTHON" "jax"
fi

# Install Tunix, Flax, Optax, Transformers and data science stack
echo ">>> Installing Tunix, Flax, Transformers and tools..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "tunix" \
    "flax" \
    "optax" \
    "transformers" \
    "datasets" \
    "huggingface_hub" \
    "sentence-transformers" \
    "pandas" \
    "numpy" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm" \
    "pyyaml" \
    "jupyter" \
    "ipykernel"

# Register Jupyter kernel
echo ">>> Registering Jupyter kernel..."
"$VENV_PYTHON" -m ipykernel install --user \
    --name "tunix-med-jax" \
    --display-name "Python (Tunix-Med JAX)"

echo ""
echo "✅ Environment setup complete in $SCRIPT_DIR/$VENV_NAME"
echo "To activate:"
echo "  source $SCRIPT_DIR/$VENV_NAME/bin/activate"
