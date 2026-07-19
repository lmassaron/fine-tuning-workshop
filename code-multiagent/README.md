# Local Coding Multi-Agent CLI

This repository implements a local, privacy-first **Coding Multi-Agent system** utilizing dynamic PEFT LoRA adapter swapping on a single base model (`unsloth/Qwen3.5-4B`). It operates under a flat consumer-hardware GPU budget of 16GB VRAM.

## Setup Instructions

Ensure you have [uv](https://github.com/astral-sh/uv) installed. Recreate the virtual environment and install dependencies:

```bash
# Create the environment using Python 3.12
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate

# Sync all dependencies exactly as declared in pyproject.toml
uv pip install -r pyproject.toml
```

## Running the Coding Agent

The coding agent acts as a command line interface. You can run tasks by invoking `coding_agent.py` directly with an instruction:

```bash
# Run a specific task
./coding_agent.py "Write a python function to solve binary search, save it to 'binary_search.py' and write unit tests"
```

Alternatively, you can run the agent without arguments to execute the default LeetCode "Longest Substring Without Repeating Characters" challenge:

```bash
./coding_agent.py
```

### Running Sanity & Verification Tests
To run the project tests:
```bash
python3 -m unittest discover tests
```
