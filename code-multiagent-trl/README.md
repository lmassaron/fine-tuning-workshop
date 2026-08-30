# Local Coding Multi-Agent CLI (TRL & PEFT)

This repository implements a local, privacy-first **Coding Multi-Agent system** utilizing dynamic PEFT LoRA adapter swapping on a single base model (`unsloth/Qwen3.5-4B`) with Hugging Face **TRL** (Transformer Reinforcement Learning) and **PEFT**, completely without Unsloth. It operates under a flat consumer-hardware GPU budget of 16GB VRAM.

## Requirements & Dependencies

The required libraries are listed in `requirements.txt`:
- `torch>=2.2.0`
- `torchvision`
- `torchaudio`
- `transformers>=4.48.0`
- `accelerate>=0.34.0`
- `bitsandbytes>=0.43.0`
- `peft>=0.14.0`
- `datasets>=2.20.0`
- `trl>=0.15.0`

## Setup Instructions

Ensure you have a Python 3.12+ virtual environment set up:

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively using `uv`:
```bash
uv pip install -r requirements.txt
```

## Running the Coding Agent

The coding agent acts as a command line interface. You can run tasks by invoking `coding_agent.py` directly with an instruction:

```bash
# Run a specific task
python3 coding_agent.py "Write a python function to solve binary search, save it to 'binary_search.py' and write unit tests"
```

Alternatively, you can run the agent without arguments to execute the default LeetCode "Longest Substring Without Repeating Characters" challenge:

```bash
python3 coding_agent.py
```

### Running Sanity & Verification Tests

To run the project test suite:
```bash
python3 -m unittest discover tests
```
