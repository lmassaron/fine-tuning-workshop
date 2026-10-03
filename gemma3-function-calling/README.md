# Gemma 3 270M Function Calling Fine-Tuning

This module fine-tunes Google's compact `google/gemma-3-270m-it` on conversational tool calling and structured function invocation using Hugging Face TRL and PEFT.

## 📌 Theme & Purpose
Ultra-small models (under 1B parameters) struggle with multi-argument structured tool use when zero-shot prompted, often producing syntax errors or plain text chit-chat. This track teaches Gemma 3 270M to recognize `<tools>...</tools>` definitions and output rigorous `<tool_call>...</tool_call>` JSON payloads.

### Key Capabilities
- **ChatML Tool Template**: Employs ChatML-style function call framing with dedicated token separation.
- **Quantized SFT Training**: Employs bfloat16 / 4-bit LoRA with memory tying (`ensure_weight_tying=True`) for lightweight execution.
- **Dual Exact-Match Benchmarking**: Quantifies both general conversation accuracy and tool syntax exact match.

---

## 📂 Files & Artifacts

- **`gemma3_270m_function_calling.ipynb`**:
  - The complete runnable Jupyter notebook with baseline generation, SFT training, and post-training exact-match evaluation.
- **`gemma3_function_calling_walkthrough.md`**:
  - Detailed architectural guide covering tokenizer adaptations, chat templates, and evaluation results.
- **`generate_nb.py`**:
  - Programmatic notebook generator script used to construct the clean, executable notebook cells.

---

## 🚀 Setup & Execution

### Prerequisites
- NVIDIA GPU with CUDA support
- [`uv`](https://docs.astral.sh/uv/) installed

### Installation
Run the environment installer to create a GPU `.venv`:

```bash
chmod +x install.sh
./install.sh
```

### Activate & Launch
```bash
source .venv/bin/activate
jupyter lab
```
Select the **Python (Gemma 3 Function Calling)** kernel in Jupyter.
