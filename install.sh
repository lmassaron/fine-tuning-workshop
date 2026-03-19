#!/bin/bash
# 
# Installation script to set up the Python environment for the fine-tuning workshop
# using uv (https://astral.sh/uv).
#
# This script detects the OS (macOS vs. Linux) and GPU availability to optimize 
# the installation of PyTorch and related libraries. It includes vllm-metal
# installation specifically for Apple Silicon macOS.

set -e # Exit on error

VENV_NAME=".venv"
PYTHON_VERSION="3.12"

# ==============================================================================
# Helper Functions for macOS vllm-metal Installation
# ==============================================================================
fetch_latest_release() {
    local repo_owner="$1"
    local repo_name="$2"

    echo ">>> Fetching latest release for ${repo_owner}/${repo_name}..." >&2
    local latest_release_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
    local release_data

    if ! release_data=$(curl -fsSL "$latest_release_url" 2>&1); then
        echo "Error: Failed to fetch release information." >&2
        echo "Please check your internet connection and try again." >&2
        exit 1
    fi

    if [[ -z "$release_data" ]] || [[ "$release_data" == *"Not Found"* ]]; then
        echo "Error: No releases found for this repository." >&2
        exit 1
    fi

    echo "$release_data"
}

extract_wheel_url() {
    local release_data="$1"
    local python_exec="$2"

    "$python_exec" -c "
import sys
import json
try:
    data = json.loads('''$release_data''')
    assets = data.get('assets',[])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
"
}

download_and_install_wheel() {
    local wheel_url="$1"
    local package_name="$2"
    local python_exec="$3"

    local wheel_name
    wheel_name=$(basename "$wheel_url")
    echo ">>> Latest release found: $wheel_name"

    local tmp_dir
    tmp_dir=$(mktemp -d)

    echo ">>> Downloading wheel..."
    local wheel_path="$tmp_dir/$wheel_name"

    if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
        echo "Error: Failed to download wheel." >&2
        rm -rf "$tmp_dir"
        exit 1
    fi

    echo ">>> Installing ${package_name}..."
    if ! uv pip install --python "$python_exec" "$wheel_path"; then
        echo "Error: Failed to install ${package_name}." >&2
        rm -rf "$tmp_dir"
        exit 1
    fi

    rm -rf "$tmp_dir"
    echo ">>> Successfully installed ${package_name}"
}
# ==============================================================================

# 1. Check if uv is installed
if ! command -v uv &>/dev/null; then
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
# Use an absolute path so it doesn't break if we change directories (e.g. for vLLM)
VENV_PYTHON="$(pwd)/$VENV_NAME/bin/python"

# 2. Install core ML libraries based on hardware
if [ "$OS_TYPE" == "Darwin" ]; then
    echo ">>> Installing macOS-optimized (MPS/CPU) stack..."
    # On macOS, we install standard torch (which includes MPS support)
    uv pip install -U \
        --python "$VENV_PYTHON" \
        "torch" \
        "torchvision" \
        "torchaudio" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate"
    
    echo "⚠️  Note: 'xformers' and 'unsloth' are primarily for Linux/CUDA and will be skipped on macOS."

    # Detect Apple Silicon to install vllm-metal
    if [[ $(uname -m) == 'arm64' ]]; then
        echo ">>> Apple Silicon detected. Installing vLLM and vllm-metal..."
        
        # Build base vLLM from source 
        VLLM_V="0.14.1"
        URL_BASE="https://github.com/vllm-project/vllm/releases/download"
        FILENAME="vllm-$VLLM_V.tar.gz"
        
        echo ">>> Fetching vLLM source ($FILENAME)..."
        curl -OL "$URL_BASE/v$VLLM_V/$FILENAME"
        tar xf "$FILENAME"
        cd "vllm-$VLLM_V"
        
        echo ">>> Installing build dependencies..."
        uv pip install --python "$VENV_PYTHON" ninja cmake setuptools_scm grpcio-tools
        
        echo ">>> Installing vLLM CPU requirements..."
        uv pip install --python "$VENV_PYTHON" setuptools numba
        
        echo ">>> Installing vLLM from source..."
        # Create symlinks to ensure ninja and cmake are findable
        # VENV_PYTHON is already an absolute path, so extract the bin directory
        VENV_BIN_DIR="$(dirname "$VENV_PYTHON")"
        ln -sf /opt/homebrew/bin/ninja "$VENV_BIN_DIR/ninja"
        ln -sf /opt/homebrew/bin/cmake "$VENV_BIN_DIR/cmake"
        
        # Set comprehensive environment for the build
        export PATH="$VENV_BIN_DIR:/opt/homebrew/bin:$PATH"
        export CMAKE_MAKE_PROGRAM="$VENV_BIN_DIR/ninja"
        export CMAKE_GENERATOR="Ninja"
        
        cd ..
        rm -rf "vllm-$VLLM_V"*
        
        # Download and install vllm-metal wheel
        REPO_OWNER="vllm-project"
        REPO_NAME="vllm-metal"
        PACKAGE_NAME="vllm-metal"
        
        RELEASE_DATA=$(fetch_latest_release "$REPO_OWNER" "$REPO_NAME")
        WHEEL_URL=$(extract_wheel_url "$RELEASE_DATA" "$VENV_PYTHON")
        
        if [[ -z "$WHEEL_URL" ]]; then
            echo "Error: No wheel file found in the latest release of vllm-metal." >&2
            exit 1
        fi
        
        download_and_install_wheel "$WHEEL_URL" "$PACKAGE_NAME" "$VENV_PYTHON"
    else
        echo "⚠️  Note: 'vLLM' macOS support requires Apple Silicon (arm64). Skipping vllm-metal."
    fi

elif [ "$HAS_NVIDIA_GPU" = true ]; then
    echo ">>> Installing Linux/CUDA-optimized (NVIDIA GPU) stack..."
    
    # 1. Detect maximum supported CUDA version from nvidia-smi's output
    CUDA_VERSION_STRING=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9\.]*\).*/\1/p')
    echo ">>> Detected maximum supported CUDA Version: $CUDA_VERSION_STRING"
    
    CUDA_MAJOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION_STRING" | cut -d. -f2)
    
    # Detect System Architecture
    ARCH=$(uname -m)
    echo ">>> Detected Architecture: $ARCH"
    
    # 2. Map system CUDA version and Architecture to the supported PyTorch package
    if [ "$ARCH" = "aarch64" ]; then
        echo ">>> ARM64 (aarch64) detected. PyTorch requires at least cu128 for ARM+CUDA."
        PT_CU_VERSION="cu128"
    else
        # Standard x86_64 logic
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            PT_CU_VERSION="cu128"
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            if [ "$CUDA_MINOR" -ge 8 ]; then           # FIX 1: added space after `if`
                PT_CU_VERSION="cu128"
            elif [ "$CUDA_MINOR" -ge 4 ]; then
                PT_CU_VERSION="cu124"
            elif [ "$CUDA_MINOR" -ge 1 ]; then
                PT_CU_VERSION="cu121"
            else
                PT_CU_VERSION="cu118"
            fi
        elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then   # FIX 2: added space before `[`
            PT_CU_VERSION="cu118"
        else
            echo "⚠️  Unsupported/Old CUDA version detected ($CUDA_VERSION_STRING)."
            PT_CU_VERSION="cu118"
        fi
    fi

    echo ">>> Selected PyTorch CUDA index: $PT_CU_VERSION"
    
    # 3. Install PyTorch using the dynamically selected index
    uv pip install -U \
        --python "$VENV_PYTHON" \
        --extra-index-url "https://download.pytorch.org/whl/$PT_CU_VERSION" \
        "torch" \
        "torchvision" \
        "transformers" \
        "trl" \
        "peft" \
        "accelerate" \
        "bitsandbytes" \
        "datasets" \
        "evaluate"
    
    echo ">>> Installing Linux/GPU workshop tools..."
    # 4. Install specialized workshop tools
    # FIX 3: Removed the ARM64 block that incorrectly skipped unsloth.
    # Unsloth no longer strictly requires xformers — it falls back to its own
    # Triton kernels when xformers is unavailable, making it fully compatible
    # with ARM64+CUDA systems such as the NVIDIA DGX Spark (GB10/GH200).
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
