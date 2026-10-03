# Fine-Tuning Language Models & Multimodal Workshop

Welcome to the **Fine-Tuning Workshop** repository. This repository brings together hands-on implementations, research notebooks, and production recipes for fine-tuning Large Language Models (LLMs) and Vision-Language Models (VLMs) across supervised fine-tuning (SFT), reinforcement learning (RL / GRPO), preference alignment (DPO), dense retrieval, and autonomous multi-agent systems.

All branches of this repository have been unified into structured, self-contained example directories. Each example is organized by **theme and purpose**, equipped with its own dedicated `README.md`, `pyproject.toml`, and GPU-optimized `install.sh` that provisions an isolated virtual environment (`.venv`) using [`uv`](https://docs.astral.sh/uv/).

---

## 🧭 Repository Structure & Examples Index

| Directory | Theme & Purpose | Model(s) | Key Frameworks |
| :--- | :--- | :--- | :--- |
| [**`knowledge-injection-sherlock/`**](file:///home/lmassaron/code/fine-tuning-workshop/knowledge-injection-sherlock/README.md) | Domain knowledge injection via Wikipedia scraping, synthetic QA generation, and QLoRA. | `google/gemma-3-1b-it` | Transformers, PEFT, TRL, Synthetic Data Kit |
| [**`reasoned-financial-sentiment/`**](file:///home/lmassaron/code/fine-tuning-workshop/reasoned-financial-sentiment/README.md) | Financial sentiment classification augmented with Chain-of-Thought (CoT) reasoning traces. | `google/gemma-3-1b-it`, `Qwen2.5-7B` | Transformers, TRL, PEFT, Datasets |
| [**`tunix-med-jax/`**](file:///home/lmassaron/code/fine-tuning-workshop/tunix-med-jax/README.md) | High-throughput SFT on cardiology consultations using Google's Tunix library and JAX/XLA. | `google/gemma-3-270M-it` | Tunix, JAX, Flax, Optax |
| [**`medical-expert-cardiology/`**](file:///home/lmassaron/code/fine-tuning-workshop/medical-expert-cardiology/README.md) | Clinical medical dialogue adaptation with target-token perplexity (PPL) evaluation. | `microsoft/Phi-4-mini-instruct` | Unsloth, TRL, PEFT, PyTorch |
| [**`vision-finetuning-latex/`**](file:///home/lmassaron/code/fine-tuning-workshop/vision-finetuning-latex/README.md) | Multimodal VLM adaptation transcribing handwritten math equations into LaTeX OCR. | `Qwen/Qwen2-VL-2B-Instruct` | Unsloth Vision, Transformers, Torchvision |
| [**`alignment-dpo/`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-dpo/README.md) | Preference alignment without separate reward models via Direct Preference Optimization (DPO). | `Qwen/Qwen2.5-3B` | TRL DPOTrainer, PEFT, PyTorch |
| [**`alignment-grpo/`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-grpo/README.md) | Reasoning reinforcement learning with Group Relative Policy Optimization (GRPO) on GSM8K. | `Qwen/Qwen2.5-0.5B-Instruct` | TRL GRPOTrainer, Unsloth, PyTorch |
| [**`gemma3-function-calling/`**](file:///home/lmassaron/code/fine-tuning-workshop/gemma3-function-calling/README.md) | Structured function calling and tool use under ChatML for ultra-compact language models. | `google/gemma-3-270m-it` | Transformers, TRL, PEFT |
| [**`nemotron-embed-finetuning/`**](file:///home/lmassaron/code/fine-tuning-workshop/nemotron-embed-finetuning/README.md) | Dense embedding model adaptation with synthetic query generation and MNRL loss. | `nvidia/Nemotron-3-Embed-1B-BF16` | Sentence-Transformers, PEFT |
| [**`code-multiagent/`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent/README.md) | Autonomous multi-agent coding system with dynamic LoRA adapter hot-swapping (Unsloth). | `Qwen/Qwen3-4B` | Unsloth, PEFT, Transformers |
| [**`code-multiagent-trl/`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent-trl/README.md) | Autonomous multi-agent coding system with dynamic LoRA adapter hot-swapping (TRL & PEFT). | `Qwen/Qwen3-4B` | TRL, PEFT, BitsAndBytes |

---

## ⚡ Quick Start: Running Any Example

Each directory is self-contained. To run an example:

1. **Install [`uv`](https://docs.astral.sh/uv/)** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate into the desired directory**:
   ```bash
   cd reasoned-financial-sentiment   # or any directory above
   ```

3. **Run the local installer**:
   The installer automatically inspects your GPU hardware, identifies the installed CUDA version, creates a `.venv` using `uv`, downloads the corresponding PyTorch or JAX wheels, and registers a Jupyter kernel:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Activate and launch**:
   ```bash
   source .venv/bin/activate
   jupyter lab   # or run the scripts directly
   ```

---

## 🖥️ Hardware Compatibility

- **NVIDIA GPU**: Recommended 16GB+ VRAM (e.g., RTX 3090/4080/4090, A10, L4, A100, H100, GB10/GH200). 
- **CUDA Support**: Scripts automatically configure indices for CUDA 11.8, 12.1, 12.4, 12.8, and 13.0 on `x86_64` and `aarch64`.
- **Apple Silicon (MPS)**: Compatible notebooks support PyTorch MPS fallback for local experimentation on M-series Macs with unified memory.
- **Google Colab**: Free-tier T4 and L4 runtimes are supported using 4-bit NormalFloat (NF4) quantization.

---

## 🔑 Hugging Face Authentication & Setup

Several base models (such as `google/gemma-3-1b-it`, `nvidia/Nemotron-3-Embed-1B-BF16`, and gated datasets) require accepting their respective license terms on the Hugging Face Hub:

1. Log in to your Hugging Face account via CLI:
   ```bash
   huggingface-cli login
   ```
2. Accept the model agreements on the respective model cards:
   - [Google Gemma 3](https://huggingface.co/google/gemma-3-1b-it)
   - [NVIDIA Nemotron 3 Embed](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
   - [Qwen Series](https://huggingface.co/Qwen)

For a step-by-step onboarding walkthrough, see [`hugging_face_setup_guide.pdf`](file:///home/lmassaron/code/fine-tuning-workshop/hugging_face_setup_guide.pdf).
