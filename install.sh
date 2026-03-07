#!/bin/bash
# 
# Installation script to set up the Python environment for the fine-tuning workshop
# using uv (https://astral.sh/uv).
#
# This script detects the OS (macOS vs. Linux) and GPU availability to optimize 
# the installation of PyTorch and related libraries.

set -e # Exit on error

VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# 1. Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first (e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh')"
    exit 1
fi

# Detect OS and GPU
OS_TYPE=$(uname -s)
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    HAS_NVIDIA_GPU=true
fi

echo ">>> Detected OS: $OS_TYPE"
echo ">>> NVIDIA GPU Found: $HAS_NVIDIA_GPU"

echo ">>> Creating virtual environment '$VENV_NAME' with Python $PYTHON_VERSION..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION"

echo ">>> Activating virtual environment and installing dependencies..."
VENV_PYTHON="./$VENV_NAME/bin/python"

# 2. Install core ML libraries based on hardware
if [ "$OS_TYPE" == "Darwin" ]; then
    echo ">>> Installing macOS-optimized (MPS/CPU) stack..."
    # On macOS, we install standard torch (which includes MPS support)
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "torch" \
        "torchvision" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate" \
    
    echo "⚠️  Note: 'xformers', 'unsloth', and 'vLLM' are primarily for Linux/CUDA and will be skipped on macOS."
elif [ "$HAS_NVIDIA_GPU" = true ]; then
    echo ">>> Installing Linux/CUDA-optimized (NVIDIA GPU) stack..."
    # Install torch with CUDA 12.4 support
    uv pip install -U \
        --python "$VENV_PYTHON" \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        "torch" \
        "torchvision" \
        "xformers" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate"
    
    echo ">>> Installing Linux/GPU workshop tools (unsloth, vllm)..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "unsloth" \
        "vllm"
else
    echo ">>> Installing standard Linux (CPU only) stack..."
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "torch" \
        "torchvision" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate"
    
    echo "⚠️  Note: 'xformers', 'unsloth', and 'vLLM' require an NVIDIA GPU and will be skipped."
fi

# 3. Install data science and utility libraries (Common to all)
echo ">>> Installing data science and utility libraries..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "pandas" \
    "numpy" \
    "scikit-learn" \
    "matplotlib" \
    "tqdm" \
    "tenacity" \
    "sentencepiece" \
    "sentence-transformers" \
    "huggingface_hub" \
    "wikipedia-api" \
    "synthetic-data-kit"

# 4. Install Jupyter for notebook support
echo ">>> Installing Jupyter..."
uv pip install -U \
    --python "$VENV_PYTHON" \
    "jupyter" \
    "ipykernel"

# Register the kernel
"$VENV_PYTHON" -m ipykernel install --user --name "fine-tuning-workshop" --display-name "Python (Fine-Tuning Workshop)"

echo ""
echo "✅ Environment setup complete!"
echo "To activate the environment, run:"
echo "source $VENV_NAME/bin/activate"
echo ""
echo "You can then select the 'Python (Fine-Tuning Workshop)' kernel in your notebooks."
