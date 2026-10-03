#!/bin/bash
set -e

echo "====================================="
echo " Starting Coding Multi-Agent Pipeline (TRL)"
echo "====================================="

echo "1. Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "2. Upgrading pip..."
pip install --upgrade pip

echo "3. Installing PyTorch (CUDA 13.0 for GB10)..."
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "4. Installing TRL and HF Dependencies..."
pip install -r requirements.txt

echo "====================================="
echo " Starting LoRA Factory Training"
echo "====================================="
python train_loras.py

echo "====================================="
echo " Testing Agent CLI"
echo "====================================="
python coding_agent.py

echo "Pipeline finished successfully!"
