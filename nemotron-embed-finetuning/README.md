# NVIDIA Nemotron-3 Embedding Model Fine-Tuning

This module fine-tunes `nvidia/Nemotron-3-Embed-1B-BF16` on domain-specific documentation using synthetic data generation and Parameter-Efficient Fine-Tuning (LoRA) via `sentence-transformers`.

## 📌 Theme & Purpose
Off-the-shelf dense embedding models frequently underperform on specialized technical vocabularies or proprietary internal knowledge bases. This project demonstrates how to:
1. Generate synthetic query-document pairs from raw technical docs using a lightweight instruction model (`Qwen/Qwen2.5-1.5B-Instruct` in 4-bit).
2. Train LoRA adapters directly on the embedding model using `MultipleNegativesRankingLoss` (MNRL).
3. Benchmark pre- and post-fine-tuning retrieval accuracy using strict **Recall@1** metrics.

---

## 📂 Files & Structure

- **`nemotron_3_embed_finetune_hf.ipynb`**:
  - The complete runnable notebook implementing dataset loading (`nvidia/Retrieval-Synthetic-NVDocs-v1`), synthetic question generation, LoRA training, and evaluation.
- **`nemotron_3_embed_finetune_walkthrough.md`**:
  - In-depth theoretical walkthrough and code explanation.
- **`ds_features.txt`**:
  - Reference dataset features schema.

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
Select the **Python (Nemotron Embed Fine-Tuning)** kernel in Jupyter.
