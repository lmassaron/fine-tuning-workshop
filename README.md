# Fine-Tuning Language Models & Multimodal Workshop

Welcome to the **Fine-Tuning Workshop** repository. This repository brings together hands-on implementations, research notebooks, and production recipes for fine-tuning Large Language Models (LLMs) and Vision-Language Models (VLMs) across supervised fine-tuning (SFT), reinforcement learning (RL / GRPO), preference alignment (DPO), dense retrieval, and autonomous multi-agent systems.

All branches of this repository have been unified into structured, self-contained example directories. Each example is organized by **theme and purpose**, equipped with its own dedicated `README.md`, `pyproject.toml`, and GPU-optimized `install.sh` that provisions an isolated virtual environment (`.venv`) using [`uv`](https://docs.astral.sh/uv/).

---

## Repository Structure & Examples Index

| # | Directory | Theme & Purpose | Key Assets |
| :-: | :--- | :--- | :--- |
| 1 | [**`knowledge-injection-sherlock/`**](file:///home/lmassaron/code/fine-tuning-workshop/knowledge-injection-sherlock/README.md) | Domain QA factual injection via Wikipedia scraping & QLoRA | Notebooks 01-04, setup guide |
| 2 | [**`reasoned-financial-sentiment/`**](file:///home/lmassaron/code/fine-tuning-workshop/reasoned-financial-sentiment/README.md) | Financial sentiment classification with Chain-of-Thought reasoning | Notebooks 05-07, 09, unified SFT notebook & walkthrough |
| 3 | [**`tunix-med-jax/`**](file:///home/lmassaron/code/fine-tuning-workshop/tunix-med-jax/README.md) | Cardiology SFT with Google's Tunix library & JAX/XLA GPU acceleration | Notebooks 05-08, synthetic kit configs, eval metrics |
| 4 | [**`medical-expert-cardiology/`**](file:///home/lmassaron/code/fine-tuning-workshop/medical-expert-cardiology/README.md) | Clinical dialogue adaptation with token-level perplexity validation | Phi-4-mini SFT notebook & comprehensive walkthrough |
| 5 | [**`vision-finetuning-latex/`**](file:///home/lmassaron/code/fine-tuning-workshop/vision-finetuning-latex/README.md) | Multimodal VLM adaptation for handwritten formula LaTeX OCR | Qwen2-VL Unsloth notebook & walkthrough |
| 6 | [**`alignment-dpo/`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-dpo/README.md) | Direct Preference Optimization (DPO) pairwise alignment | Qwen2.5-3B DPO notebook & walkthrough |
| 7 | [**`alignment-grpo/`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-grpo/README.md) | Mathematical reasoning RL via Group Relative Policy Optimization | Qwen2.5 GSM8K notebook (TRL + Unsloth), training curves |
| 8 | [**`gemma3-function-calling/`**](file:///home/lmassaron/code/fine-tuning-workshop/gemma3-function-calling/README.md) | Structured tool use and ChatML JSON function calling on Gemma 3 270M | Function calling notebook, generator script & walkthrough |
| 9 | [**`nemotron-embed-finetuning/`**](file:///home/lmassaron/code/fine-tuning-workshop/nemotron-embed-finetuning/README.md) | Dense embedding model adaptation with synthetic queries & MNRL loss | Nemotron-3 notebook, schema, and walkthrough |
| 10 | [**`code-multiagent-unsloth/`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent-unsloth/README.md) | Autonomous multi-agent coding system with dynamic LoRA swapping (Unsloth) | Agent CLI, LoRA training pipeline, test suite, adapters |
| 11 | [**`code-multiagent-trl/`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent-trl/README.md) | Autonomous multi-agent coding system with dynamic LoRA swapping (TRL & PEFT) | Agent CLI, pure HF TRL/PEFT pipeline, benchmark results |


---

## Quick Start: Running Any Example

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

## Hardware Compatibility & DGX Spark VRAM Benchmarks

Every track has been benchmarked on the local **DGX Spark** instance equipped with an **NVIDIA GB10** (Grace Blackwell architecture, 121.69 GB unified memory, Driver 580.178.04, CUDA 13.0).

The table below details hardware compatibility (NVIDIA GPU vs Apple Silicon MPS) and the measured peak VRAM footprint during model loading and training/inference execution:

| Track # | Directory | Base Model | Platform Support | Peak VRAM (GB10) | Notes |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | [**`knowledge-injection-sherlock`**](file:///home/lmassaron/code/fine-tuning-workshop/knowledge-injection-sherlock/README.md) | `google/gemma-3-1b-it` | CUDA + Apple Silicon (MPS) | **3.72 GB** | QLoRA SFT (`adamw_torch_fused`, bf16) |
| **2** | [**`reasoned-financial-sentiment`**](file:///home/lmassaron/code/fine-tuning-workshop/reasoned-financial-sentiment/README.md) | `google/gemma-4-E2B-it` | CUDA + Apple Silicon (MPS) | **8.86 GB** | 4-bit NF4 QLoRA SFT with Chain-of-Thought reasoning |
| **3** | [**`tunix-med-jax`**](file:///home/lmassaron/code/fine-tuning-workshop/tunix-med-jax/README.md) | `google/gemma-3-270m-it` | CUDA + Apple Silicon (MPS) | **0.54 GB** | Tunix JAX/XLA SFT & PyTorch evaluation |
| **4** | [**`medical-expert-cardiology`**](file:///home/lmassaron/code/fine-tuning-workshop/medical-expert-cardiology/README.md) | `unsloth/Phi-4-mini-instruct` | **NVIDIA GPU only** (Unsloth) | **6.57 GB** | 4-bit SFT with Unsloth kernel acceleration |
| **5** | [**`vision-finetuning-latex`**](file:///home/lmassaron/code/fine-tuning-workshop/vision-finetuning-latex/README.md) | `unsloth/Qwen2-VL-2B-Instruct` | **NVIDIA GPU only** (Unsloth) | **2.43 GB** | 4-bit VLM vision & language LoRA |
| **6** | [**`alignment-dpo`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-dpo/README.md) | `unsloth/Qwen2.5-3B-Instruct` | **NVIDIA GPU only** (Unsloth) | **2.29 GB** | 4-bit Direct Preference Optimization (DPO) |
| **7** | [**`alignment-grpo`**](file:///home/lmassaron/code/fine-tuning-workshop/alignment-grpo/README.md) | `Qwen/Qwen2.5-0.5B-Instruct` | **NVIDIA GPU only** (Unsloth/vLLM) | **3.04 GB** | Group Relative Policy Optimization (RL) |
| **8** | [**`gemma3-function-calling`**](file:///home/lmassaron/code/fine-tuning-workshop/gemma3-function-calling/README.md) | `google/gemma-3-1b-it` | CUDA + Apple Silicon (MPS) | **1.54 GB** | Structured JSON tool-calling SFT |
| **9** | [**`nemotron-embed-finetuning`**](file:///home/lmassaron/code/fine-tuning-workshop/nemotron-embed-finetuning/README.md) | `nvidia/llama-3.2-nv-embedqa-1b-v2` | CUDA + Apple Silicon (MPS) | **9.06 GB** | MultipleNegativesRankingLoss dense retrieval SFT |
| **10** | [**`code-multiagent-unsloth`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent-unsloth/README.md) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | **NVIDIA GPU only** (Unsloth) | **3.66 GB** | Dual LoRA agents with fast Unsloth inference |
| **11** | [**`code-multiagent-trl`**](file:///home/lmassaron/code/fine-tuning-workshop/code-multiagent-trl/README.md) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | CUDA + Apple Silicon (MPS) + CPU | **3.01 GB** | Cross-platform Hugging Face TRL multi-agent system |

### Platform Notes:
- **NVIDIA GPU (CUDA)**: Full support across all 11 tracks. Supports 4-bit bitsandbytes quantization, flash-attention, fused optimizers, and Unsloth acceleration.
- **Apple Silicon (MPS)**: Supported across all pure Hugging Face / PyTorch / JAX tracks (Tracks 1, 2, 3, 8, 9, and 11). Tracks gracefully fall back to native half-precision (`bfloat16`/`float16`) to leverage Apple Silicon unified memory without bitsandbytes CUDA constraints.
- **Unsloth Tracks (GPU Only)**: Tracks 4, 5, 6, 7, and 10 rely on Unsloth custom CUDA / Triton kernels and require an NVIDIA GPU.

---

## Hardware Compatibility

- **NVIDIA GPU**: Recommended 16GB+ VRAM (e.g., RTX 3090/4080/4090, A10, L4, A100, H100, GB10/GH200). 
- **CUDA Support**: Scripts automatically configure indices for CUDA 11.8, 12.1, 12.4, 12.8, and 13.0 on `x86_64` and `aarch64`.
- **Apple Silicon (MPS)**: Compatible notebooks support PyTorch MPS fallback for local experimentation on M-series Macs with unified memory.
- **Google Colab**: Free-tier T4 and L4 runtimes are supported using 4-bit NormalFloat (NF4) quantization.

---

## Hugging Face Authentication & Setup

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
