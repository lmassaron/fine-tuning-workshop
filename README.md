# Fine-Tuning Language Models & Multimodal Workshop

Welcome to the **Fine-Tuning Workshop** repository. This repository brings together hands-on implementations, research notebooks, and production recipes for fine-tuning Large Language Models (LLMs) and Vision-Language Models (VLMs) across supervised fine-tuning (SFT), reinforcement learning (RL / GRPO), preference alignment (DPO), dense retrieval, and autonomous multi-agent systems.

All branches of this repository have been unified into structured, self-contained example directories. Each example is organized by **theme and purpose**, equipped with its own dedicated `README.md`, `pyproject.toml`, and GPU-optimized `install.sh` that provisions an isolated virtual environment (`.venv`) using [`uv`](https://docs.astral.sh/uv/).

---

## Repository Structure & Examples Index

| # | Directory | Theme & Purpose | Key Assets |
| :-: | :--- | :--- | :--- |
| 1 | [**`knowledge-injection-sherlock/`**](knowledge-injection-sherlock/README.md) | Domain QA factual injection via Wikipedia scraping & QLoRA | Notebooks 01-04, setup guide |
| 2 | [**`reasoned-financial-sentiment/`**](reasoned-financial-sentiment/README.md) | Financial sentiment classification with Chain-of-Thought reasoning | Notebooks 01-04, unified SFT notebook & walkthrough |
| 3 | [**`tunix-med-jax/`**](tunix-med-jax/README.md) | Cardiology SFT with Google's Tunix library & JAX/XLA GPU acceleration | Notebooks 01-04, synthetic kit configs, eval metrics |
| 4 | [**`medical-expert-cardiology/`**](medical-expert-cardiology/README.md) | Clinical dialogue adaptation with token-level perplexity validation | Phi-4-mini SFT notebook & comprehensive walkthrough |
| 5 | [**`vision-finetuning-latex/`**](vision-finetuning-latex/README.md) | Multimodal VLM adaptation for handwritten formula LaTeX OCR | Qwen2-VL Unsloth notebook & walkthrough |
| 6 | [**`alignment-dpo/`**](alignment-dpo/README.md) | Direct Preference Optimization (DPO) pairwise alignment | Qwen2.5-3B DPO notebook & walkthrough |
| 7 | [**`alignment-grpo/`**](alignment-grpo/README.md) | Mathematical reasoning RL via Group Relative Policy Optimization | Qwen2.5 GSM8K notebook (TRL + Unsloth), training curves |
| 8 | [**`gemma3-function-calling/`**](gemma3-function-calling/README.md) | Structured tool use and ChatML JSON function calling on Gemma 3 270M | Function calling notebook, generator script & walkthrough |
| 9 | [**`nemotron-embed-finetuning/`**](nemotron-embed-finetuning/README.md) | Dense embedding model adaptation with synthetic queries & MNRL loss | Nemotron-3 notebook, schema, and walkthrough |
| 10 | [**`code-multiagent-unsloth/`**](code-multiagent-unsloth/README.md) | Autonomous multi-agent coding system with dynamic LoRA swapping (Unsloth) | Agent CLI, LoRA training pipeline, test suite, adapters |
| 11 | [**`code-multiagent-trl/`**](code-multiagent-trl/README.md) | Autonomous multi-agent coding system with dynamic LoRA swapping (TRL & PEFT) | Agent CLI, pure HF TRL/PEFT pipeline, benchmark results |


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

## Google Colab Free Tier Compatibility

All **20 notebooks** in this repository are designed and benchmarked to run within the hardware constraints of the **Google Colab Free Tier** (equipped with an **NVIDIA T4 GPU** with 15.36 GB VRAM and ~12.7 GB system RAM). Every notebook includes an **Open in Colab** badge at the top for one-click launching.

### Recommended Colab Setup:
- **Select GPU Runtime**: Go to `Runtime > Change runtime type` and select **T4 GPU** (required for CUDA acceleration, 4-bit quantization, and Unsloth).
- **Hugging Face Authentication**: For gated models (`google/gemma-3-*`, `unsloth/Phi-4-mini-instruct`), add your Hugging Face access token under Colab's **Secrets** tab (`HF_TOKEN`) or authenticate via `huggingface_hub.login()`.
- **Precision on T4**: The T4 GPU architecture (`sm_75`) does not have native hardware bfloat16 tensor cores. Notebooks automatically use `float16` or 4-bit NormalFloat (`NF4`) quantization to ensure fast throughput and low VRAM usage.

### Colab Free Tier Notebook Index

| Track # | Track Directory | Notebook | Base Model | Free Tier (T4) | Peak VRAM | Key Functionality |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | [`knowledge-injection-sherlock`](knowledge-injection-sherlock/README.md) | [`01_knowledge_evaluation.ipynb`](knowledge-injection-sherlock/01_knowledge_evaluation.ipynb) | `google/gemma-3-1b-it` | ✅ Supported | ~3.7 GB | Baseline evaluation across 25 factual Sherlock queries |
| **1** | [`knowledge-injection-sherlock`](knowledge-injection-sherlock/README.md) | [`02_synthetic_data_preparation.ipynb`](knowledge-injection-sherlock/02_synthetic_data_preparation.ipynb) | Synthetic Data Kit | ✅ Supported | ~5.5 GB | Wikipedia scraping & synthetic QA pair generation |
| **1** | [`knowledge-injection-sherlock`](knowledge-injection-sherlock/README.md) | [`03_fine_tuning_QA.ipynb`](knowledge-injection-sherlock/03_fine_tuning_QA.ipynb) | `google/gemma-3-1b-it` | ✅ Supported | ~3.7 GB | QLoRA SFT on Sherlock Holmes domain QA |
| **1** | [`knowledge-injection-sherlock`](knowledge-injection-sherlock/README.md) | [`04_knowledge_evaluation_final.ipynb`](knowledge-injection-sherlock/04_knowledge_evaluation_final.ipynb) | `gemma-3-1b-sherlock-expert` | ✅ Supported | ~3.7 GB | Comparative post-training evaluation vs baseline |
| **2** | [`reasoned-financial-sentiment`](reasoned-financial-sentiment/README.md) | [`01_generate_sentiment_explanations.ipynb`](reasoned-financial-sentiment/01_generate_sentiment_explanations.ipynb) | `Qwen/Qwen2.5-7B-Instruct` | ✅ Supported | ~5.5 GB | 4-bit NF4 teacher generation of Chain-of-Thought rationales |
| **2** | [`reasoned-financial-sentiment`](reasoned-financial-sentiment/README.md) | [`02_fine_tuning_sentiment.ipynb`](reasoned-financial-sentiment/02_fine_tuning_sentiment.ipynb) | `google/gemma-3-1b-it` | ✅ Supported | ~4.0 GB | 4-bit NF4 QLoRA SFT on FinancialPhraseBank |
| **2** | [`reasoned-financial-sentiment`](reasoned-financial-sentiment/README.md) | [`03_sentiment_evaluation.ipynb`](reasoned-financial-sentiment/03_sentiment_evaluation.ipynb) | Gemma-3-1B + Qwen-7B Judge | ✅ Supported | ~5.5 GB | Strict accuracy and LLM-as-a-judge scoring |
| **2** | [`reasoned-financial-sentiment`](reasoned-financial-sentiment/README.md) | [`04_augment_sentiment_data.ipynb`](reasoned-financial-sentiment/04_augment_sentiment_data.ipynb) | `Qwen/Qwen2.5-7B-Instruct` | ✅ Supported | ~5.5 GB | Dataset augmentation with judge verification |
| **2** | [`reasoned-financial-sentiment`](reasoned-financial-sentiment/README.md) | [`financial_sentiment_cot.ipynb`](reasoned-financial-sentiment/financial_sentiment_cot.ipynb) | `google/gemma-4-E2B-it` / 1B | ✅ Supported | ~8.9 GB | Complete unified end-to-end sentiment classification with CoT |
| **3** | [`tunix-med-jax`](tunix-med-jax/README.md) | [`01_medical_baseline_evaluation.ipynb`](tunix-med-jax/01_medical_baseline_evaluation.ipynb) | `google/gemma-3-270m-it` | ✅ Supported | ~0.5 GB | Baseline cardiology knowledge evaluation |
| **3** | [`tunix-med-jax`](tunix-med-jax/README.md) | [`02_medical_synthetic_data.ipynb`](tunix-med-jax/02_medical_synthetic_data.ipynb) | Synthetic Data Kit | ✅ Supported | < 1.0 GB | Multi-source clinical dialogue generation |
| **3** | [`tunix-med-jax`](tunix-med-jax/README.md) | [`03_tunix_sft_training.ipynb`](tunix-med-jax/03_tunix_sft_training.ipynb) | `google/gemma-3-270m-it` | ✅ Supported | ~0.5 GB | Supervised fine-tuning using Google Tunix & JAX/XLA |
| **3** | [`tunix-med-jax`](tunix-med-jax/README.md) | [`04_medical_evaluation_final.ipynb`](tunix-med-jax/04_medical_evaluation_final.ipynb) | Gemma-3-270M + LoRA | ✅ Supported | ~0.8 GB | Final post-SFT proof-of-knowledge evaluation |
| **4** | [`medical-expert-cardiology`](medical-expert-cardiology/README.md) | [`medical_expert_cardiology.ipynb`](medical-expert-cardiology/medical_expert_cardiology.ipynb) | `unsloth/Phi-4-mini-instruct` | ✅ Supported | ~6.6 GB | 4-bit SFT with Unsloth kernel acceleration |
| **5** | [`vision-finetuning-latex`](vision-finetuning-latex/README.md) | [`vision_finetuning_latex.ipynb`](vision-finetuning-latex/vision_finetuning_latex.ipynb) | `unsloth/Qwen2-VL-2B-Instruct` | ✅ Supported | ~2.4 GB | 4-bit multimodal VLM vision & text LoRA |
| **6** | [`alignment-dpo`](alignment-dpo/README.md) | [`alignment_dpo.ipynb`](alignment-dpo/alignment_dpo.ipynb) | `unsloth/Qwen2.5-3B-Instruct` | ✅ Supported | ~2.3 GB | Direct Preference Optimization (DPO) with Unsloth |
| **7** | [`alignment-grpo`](alignment-grpo/README.md) | [`alignment_grpo.ipynb`](alignment-grpo/alignment_grpo.ipynb) | `Qwen/Qwen2.5-0.5B-Instruct` | ✅ Supported | ~3.0 GB | Group Relative Policy Optimization (RL) via Hugging Face TRL |
| **7** | [`alignment-grpo`](alignment-grpo/README.md) | [`alignment_grpo_unsloth.ipynb`](alignment-grpo/alignment_grpo_unsloth.ipynb) | `Qwen/Qwen2.5-0.5B-Instruct` | ✅ Supported | ~3.0 GB | Group Relative Policy Optimization (RL) via Unsloth |
| **8** | [`gemma3-function-calling`](gemma3-function-calling/README.md) | [`gemma3_270m_function_calling.ipynb`](gemma3-function-calling/gemma3_270m_function_calling.ipynb) | `google/gemma-3-270m-it` | ✅ Supported | ~1.5 GB | ChatML JSON tool-calling and function dispatch SFT |
| **9** | [`nemotron-embed-finetuning`](nemotron-embed-finetuning/README.md) | [`nemotron_3_embed_finetune_hf.ipynb`](nemotron-embed-finetuning/nemotron_3_embed_finetune_hf.ipynb) | `llama-3.2-nv-embedqa-1b-v2` | ✅ Supported | ~9.1 GB | Dense embedding adaptation with MultipleNegativesRankingLoss |

> [!NOTE]
> Tracks 10 & 11 ([`code-multiagent-unsloth`](code-multiagent-unsloth/README.md) and [`code-multiagent-trl`](code-multiagent-trl/README.md)) are standalone CLI-based autonomous multi-agent systems driven by Python scripts (`coding_agent.py`) rather than interactive notebooks.

---

## DGX Spark (NVIDIA GB10) Benchmarks & Memory Usage

Every track has been benchmarked on the local **DGX Spark** instance equipped with an **NVIDIA GB10** (Grace Blackwell architecture, 121.69 GB unified memory, Driver 580.178.04, CUDA 13.0).

The table below details hardware compatibility (NVIDIA GPU vs Apple Silicon MPS) and the measured peak VRAM footprint during model loading and training/inference execution:

| Track # | Directory | Base Model | Platform Support | Peak VRAM (GB10) | Notes |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | [**`knowledge-injection-sherlock`**](knowledge-injection-sherlock/README.md) | `google/gemma-3-1b-it` | CUDA + Apple Silicon (MPS) | **3.72 GB** | QLoRA SFT (`adamw_torch_fused`, bf16) |
| **2** | [**`reasoned-financial-sentiment`**](reasoned-financial-sentiment/README.md) | `google/gemma-4-E2B-it` | CUDA + Apple Silicon (MPS) | **8.86 GB** | 4-bit NF4 QLoRA SFT with Chain-of-Thought reasoning |
| **3** | [**`tunix-med-jax`**](tunix-med-jax/README.md) | `google/gemma-3-270m-it` | CUDA + Apple Silicon (MPS) | **0.54 GB** | Tunix JAX/XLA SFT & PyTorch evaluation |
| **4** | [**`medical-expert-cardiology`**](medical-expert-cardiology/README.md) | `unsloth/Phi-4-mini-instruct` | **NVIDIA GPU only** (Unsloth) | **6.57 GB** | 4-bit SFT with Unsloth kernel acceleration |
| **5** | [**`vision-finetuning-latex`**](vision-finetuning-latex/README.md) | `unsloth/Qwen2-VL-2B-Instruct` | **NVIDIA GPU only** (Unsloth) | **2.43 GB** | 4-bit VLM vision & language LoRA |
| **6** | [**`alignment-dpo`**](alignment-dpo/README.md) | `unsloth/Qwen2.5-3B-Instruct` | **NVIDIA GPU only** (Unsloth) | **2.29 GB** | 4-bit Direct Preference Optimization (DPO) |
| **7** | [**`alignment-grpo`**](alignment-grpo/README.md) | `Qwen/Qwen2.5-0.5B-Instruct` | **NVIDIA GPU only** (Unsloth/vLLM) | **3.04 GB** | Group Relative Policy Optimization (RL) |
| **8** | [**`gemma3-function-calling`**](gemma3-function-calling/README.md) | `google/gemma-3-1b-it` | CUDA + Apple Silicon (MPS) | **1.54 GB** | Structured JSON tool-calling SFT |
| **9** | [**`nemotron-embed-finetuning`**](nemotron-embed-finetuning/README.md) | `nvidia/llama-3.2-nv-embedqa-1b-v2` | CUDA + Apple Silicon (MPS) | **9.06 GB** | MultipleNegativesRankingLoss dense retrieval SFT |
| **10** | [**`code-multiagent-unsloth`**](code-multiagent-unsloth/README.md) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | **NVIDIA GPU only** (Unsloth) | **3.66 GB** | Dual LoRA agents with fast Unsloth inference |
| **11** | [**`code-multiagent-trl`**](code-multiagent-trl/README.md) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | CUDA + Apple Silicon (MPS) + CPU | **3.01 GB** | Cross-platform Hugging Face TRL multi-agent system |

### Platform Compatibility Summary:
- **NVIDIA GPU (CUDA)**: Full support across all 11 tracks. Supports 4-bit bitsandbytes quantization, flash-attention, fused optimizers, and Unsloth acceleration. Recommended 16GB+ VRAM (RTX 3090/4080/4090, A10, L4, A100, H100, GB10/GH200). Scripts auto-detect CUDA 11.8, 12.1, 12.4, 12.8, and 13.0 on `x86_64` and `aarch64`.
- **Apple Silicon (MPS)**: Supported across all pure Hugging Face / PyTorch / JAX tracks (Tracks 1, 2, 3, 8, 9, and 11). Tracks gracefully fall back to native half-precision (`bfloat16`/`float16`) to leverage Apple Silicon unified memory without bitsandbytes CUDA constraints.
- **Unsloth Tracks (GPU Only)**: Tracks 4, 5, 6, 7, and 10 rely on Unsloth custom CUDA / Triton kernels and require an NVIDIA GPU.
- **Google Colab**: Free-tier T4 and L4 runtimes are supported across all 20 notebooks using 4-bit NormalFloat (NF4) quantization and compact base models (270M–3.8B).

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

For a step-by-step onboarding walkthrough, see [`hugging_face_setup_guide.pdf`](hugging_face_setup_guide.pdf).
