# Sherlock Holmes Domain Knowledge Injection

This module implements end-to-end domain-specific knowledge injection via Supervised Fine-Tuning (SFT) and QLoRA on `google/gemma-3-1b-it`.

## 📌 Theme & Purpose
Standard language models have broad knowledge, but frequently hallucinate or lack precision when queried on intricate niche lore. This track demonstrates how to:
1. Benchmark baseline factual recall and quantify knowledge gaps.
2. Automatically harvest unstructured source texts (Sherlock Holmes canon via Wikipedia) and generate high-fidelity QA pairs using a teacher model.
3. Apply QLoRA (4-bit quantized Low-Rank Adaptation) using Hugging Face TRL and PEFT to inject domain knowledge efficiently on single-GPU hardware.
4. Statistically evaluate the knowledge gain using exact match, semantic embeddings, and an LLM-as-a-judge metric.

---

## 📂 Notebook Pipeline

1. **`01_knowledge_evaluation.ipynb`**:
   - Assesses baseline performance of pre-trained `google/gemma-3-1b-it` on Sherlock Holmes trivia.
   - Computes baseline keyword F1, semantic similarity via `sentence-transformers`, and LLM-judge scores.
2. **`02_synthetic_data_preparation.ipynb`**:
   - Scrapes Wikipedia articles for Sherlock Holmes stories.
   - Employs synthetic prompt generation to synthesize question-answer pairs and curates them with quality filtering.
3. **`03_fine_tuning_QA.ipynb`**:
   - Fine-tunes `google/gemma-3-1b-it` with 4-bit NF4 quantization using `SFTTrainer` and PEFT LoRA adapters.
   - Saves adapters locally.
4. **`04_knowledge_evaluation_final.ipynb`**:
   - Evaluates fine-tuned model against identical benchmark questions to measure post-training knowledge gain and perplexity reduction.

---

## 🚀 Setup & Execution

### Prerequisites
- NVIDIA GPU with $\ge$ 16GB VRAM (or Apple Silicon MPS / Colab T4)
- [`uv`](https://docs.astral.sh/uv/) installed

### Installation
Run the environment installer to create a GPU-configured `.venv`:

```bash
chmod +x install.sh
./install.sh
```

### Activate & Launch
```bash
source .venv/bin/activate
jupyter lab
```
When running the notebooks, select the **Python (Knowledge Injection Sherlock)** kernel.
