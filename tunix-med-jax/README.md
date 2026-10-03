# Tunix-Med: High-Performance Medical Intelligence (JAX & Tunix)

This module implements a Supervised Fine-Tuning (SFT) pipeline targeting clinical cardiology dialogue using Google's **Tunix** library and **JAX / Flax / XLA**.

## 📌 Theme & Purpose
Fine-tuning LLMs with JAX offers substantial throughput advantages over traditional PyTorch loops via XLA kernel fusion and deterministic execution. This example demonstrates fine-tuning `google/gemma-3-270M-it` on specialized cardiology QA dialogues using Tunix's PEFT LoRA implementation.

### Key Capabilities
- **XLA Compilation**: Uses JAX's accelerated compiler for high GPU utilization.
- **Einsum Layer LoRA**: Applies LoRA adapters to attention projection and einsum weight tensors.
- **Surgical Checkpointing**: Implements custom validation loss hooks to prevent overfitting.
- **Streaming Pipeline**: Feeds data continuously without caching massive datasets in host RAM.

---

## 📂 Pipeline Stages

1. **`05_medical_baseline_evaluation.ipynb`**:
   - Assesses baseline `gemma-3-270M-it` response quality across cardiology questions, establishing baseline perplexity and similarity scores.
2. **`06_medical_synthetic_data.ipynb`**:
   - Curates and cleans domain dialogue pairs using `synthetic_data_kit_config.yaml`.
3. **`07_tunix_sft_training.ipynb`**:
   - Executes LoRA training with Tunix, JAX, and Flax.
4. **`08_medical_evaluation_final.ipynb`**:
   - Calculates post-training cardiology metrics, quantifying factual retention and response coherence.

---

## 🚀 Setup & Execution

### Prerequisites
- NVIDIA GPU with CUDA 12+ support
- [`uv`](https://docs.astral.sh/uv/) installed

### Installation
Run the environment installer to create a GPU JAX `.venv`:

```bash
chmod +x install.sh
./install.sh
```

### Activate & Launch
```bash
source .venv/bin/activate
jupyter lab
```
Select the **Python (Tunix-Med JAX)** kernel in Jupyter.
