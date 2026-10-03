# Reasoned Financial Sentiment (Chain-of-Thought)

This module implements Chain-of-Thought (CoT) reasoning distillation and classification for financial text analysis.

## 📌 Theme & Purpose
Traditional sentiment classification models output discrete labels (`positive`, `negative`, `neutral`) without justification. In high-stakes financial domains, unexplained predictions are untrusted and brittle. 

This project trains smaller student models (`Gemma-3-1B` and `Gemma-4-E2B-it`) to generate structured reasoning traces inside `<reasoning>...</reasoning>` tags before asserting the final label in `<sentiment>...</sentiment>` tags.

### Key Capabilities
- **Knowledge Distillation**: Uses larger teacher models (such as `Qwen/Qwen2.5-7B-Instruct`) to generate rigorous financial rationales for the FinancialPhraseBank dataset.
- **Structured XML Output**: Enforces strict formatting constraints so downstream parsers can extract both explanation and label deterministically.
- **LLM-as-a-Judge Validation**: Validates that reasoning traces are logically sound and not post-hoc hallucinations.

---

## 📂 Notebook Pipeline

- **`05_generate_sentiment_explanations.ipynb`**:
  - Leverages a teacher model to distill financial rationale on corporate earnings, revenue growth, and market indicators.
- **`06_fine_tuning_sentiment.ipynb`**:
  - Fine-tunes Gemma 3 1B with QLoRA to generate `<reasoning>` and `<sentiment>` tags simultaneously.
- **`07_sentiment_evaluation.ipynb`**:
  - Benchmarks classification accuracy, tag integrity, and reasoning validity using LLM-as-a-judge.
- **`financial_sentiment_cot.ipynb`**:
  - Unified end-to-end recreation training Gemma 4 on augmented FinancialPhraseBank with full SFT evaluation metrics.

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
When running the notebooks, select the **Python (Reasoned Financial Sentiment)** kernel.
