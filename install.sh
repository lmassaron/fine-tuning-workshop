#!/bin/bash
#
# Consolidated installation script to set up the Python environment for the
# fine-tuning workshop notebooks using uv.

set -e # Exit immediately if a command exits with a non-zero status.

VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it from https://astral.sh/uv" >&2
    exit 1
fi
echo ">>> Found uv version: $(uv --version)"


echo ">>> Creating virtual environment '$VENV_NAME' with Python >= $PYTHON_VERSION..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --seed

echo ">>> Virtual environment created."
echo ">>> Installing dependencies..."

# Install the main set of packages
# Using a single block for better readability and management.
uv pip install -U \
    --python "$VENV_NAME/bin/python" \
    'torch' --extra-index-url https://download.pytorch.org/whl/cu124 \
    'transformers[torch]' \
    'trl[peft]' \
    'datasets' \
    'accelerate' \
    'bitsandbytes' \
    'evaluate' \
    'sentencepiece' \
    'sentence_transformers' \
    'scikit-learn' \
    'pandas' \
    'numpy' \
    'tqdm' \
    'tensorboard' \
    'jupyter' \
    'ipykernel' \
    'matplotlib' \
    'wikipedia-api' \
    'tenacity' \
    'unsloth' \
    'vllm' \
    'synthetic-data-kit==0.0.3' \
    'xformers' --extra-index-url https://download.pytorch.org/whl/cu124

# Install build tools needed for flash-attn, then flash-attn itself.
# This is kept for completeness, though notebook 03 will be modified not to use it.
echo ">>> Installing flash-attn..."
uv pip install \
    --python "$VENV_NAME/bin/python" \
    --upgrade wheel setuptools ninja

uv pip install \
    --python "$VENV_NAME/bin/python" \
    flash-attn --no-build-isolation

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"