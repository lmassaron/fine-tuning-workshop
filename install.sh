#!/bin/bash
#
# Installation script to set up the Python environment for the
# Gemma fine-tuning and hyperparameter optimization script using uv.
#
# v3: Includes special handling for the build dependencies of flash-attn.

set -e # Exit immediately if a command exits with a non-zero status.

VENV_NAME=".venv"
PYTHON_VERSION="3.12" # Specify your desired Python version

# --- Function to check for uv installation (from previous version) ---
check_uv_installation() {
    if command -v uv &> /dev/null; then
        echo ">>> Found uv version: $(uv --version)"
        return 0
    fi
    DETECTED_PATH=""
    if [ -f "$HOME/.cargo/bin/uv" ]; then DETECTED_PATH="$HOME/.cargo/bin"; fi
    if [ -f "$HOME/.local/bin/uv" ]; then DETECTED_PATH="$HOME/.local/bin"; fi
    if [ -n "$DETECTED_PATH" ]; then
        echo "Error: 'uv' command not found. It seems to be installed in '$DETECTED_PATH'." >&2
        echo "Please run this script again using:" >&2
        echo "PATH=\"$DETECTED_PATH:\$PATH\" ./setup_env.sh" >&2
    else
        echo "Error: uv is not installed. Please install it from https://astral.sh/uv" >&2
    fi
    exit 1
}

# --- Main Script Logic ---

# 1. Check if uv is available
check_uv_installation

echo ">>> Creating virtual environment '$VENV_NAME' with Python >= $PYTHON_VERSION..."

# 2. Create the virtual environment
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --seed

echo ">>> Virtual environment created."
echo ">>> Installing main dependencies..."

# 3. Install the main set of packages
# Note: Adjust CUDA version (cu121, cu118, etc.) to match your system.
uv pip install -U \
    --python "$VENV_NAME/bin/python" \
    'torch' --extra-index-url https://download.pytorch.org/whl/cu124 \
    'transformers[torch]' \
    'trl[peft]' \
    'datasets' \
    'accelerate' \
    'bitsandbytes' \
    'evaluate' \
    'trl'

uv pip install -U \
    'protobuf' \
    'sentencepiece' \
    'optuna' \
    'scikit-learn' \
    'pandas' \
    'numpy' \
    'tqdm' \
    'tensorboard' \
    'jupyter' \
    'ipykernel'


echo ">>> Installing build dependencies for flash-attn..."

# 4. Install the build tools needed for flash-attn
uv pip install \
    --python "$VENV_NAME/bin/python" \
    --upgrade wheel setuptools ninja

echo ">>> Installing flash-attn with build isolation disabled..."

# 5. Install flash-attn using the pre-installed build tools
uv pip install \
    --python "$VENV_NAME/bin/python" \
    flash-attn --no-build-isolation

uv pip install -U \
    --python "$VENV_NAME/bin/python" \
    'vllm'

uv pip install -U \
    --python "$VENV_NAME/bin/python" \
    'trl'
uv pip install -U \
    --python "$VENV_NAME/bin/python" \
    'numba'
    
echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"