# Fine-Tuning Language Models Workshop

This project provides a hands-on workshop for fine-tuning large language models (LLMs). It demonstrates a complete workflow, from evaluating a base model to generating a synthetic dataset, fine-tuning the model, and evaluating its improved performance. 

The workshop covers two primary tracks:
1.  **Sherlock Holmes Expert**: Domain-specific knowledge injection via QA fine-tuning.
2.  **Reasoned Financial Sentiment**: Enhancing classification with Chain-of-Thought (CoT) reasoning.

## Compatibility and Hardware Requirements

The examples in this repository are optimized for the following environments:

*   **NVIDIA GPU systems**: 16GB VRAM or more (e.g., RTX 3090/4080, A10/L4).
*   **Google Colab**: Compatible with the free **T4 GPU** tier.
*   **macOS (Apple Silicon)**: M-series chips with at least **16GB of unified memory**.

## Workshop Workflow

The project is divided into seven Jupyter notebooks, organized into two tracks:

### Track 1: Sherlock Holmes Expert (Knowledge Injection)
1.  **`01_knowledge_evaluation.ipynb`**: Evaluates the baseline knowledge of the pre-trained `google/gemma-3-1b-it` model on Sherlock Holmes lore.
2.  **`02_synthetic_data_preparation.ipynb`**: Demonstrates how to scrape Wikipedia and use a teacher model to generate and curate a high-quality QA dataset.
3.  **`03_fine_tuning_QA.ipynb`**: Performs Supervised Fine-Tuning (SFT) using QLoRA to inject domain-specific knowledge.
4.  **`04_knowledge_evaluation_final.ipynb`**: Re-evaluates the fine-tuned model to quantify the "knowledge gain" compared to the baseline.

### Track 2: Reasoned Financial Sentiment (Reasoning & Classification)
5.  **`05_generate_sentiment_explanations.ipynb`**: Uses a powerful teacher model (**Qwen 2.5 7B**) to generate financial explanations for the FinancialPhraseBank dataset.
6.  **`06_fine_tuning_sentiment.ipynb`**: Fine-tunes **Gemma 3 1B** to predict both sentiment and the underlying financial reasoning (CoT).
7.  **`07_sentiment_evaluation.ipynb`**: Evaluates the sentiment model on accuracy, instruction following (tags), and reasoning quality (LLM-as-a-judge).

## Getting Started

To run the notebooks, ensure you have a compatible environment. It is also necessary to have a Hugging Face account and to accept the Google Gemma model licence. For detailed, step-by-step assistance, please refer to the `hugging_face_setup_guide.pdf` included in this repository. 

You can install the necessary dependencies using the provided script:

```bash
sh install.sh
```

## Notebook Details

### 1. `01_knowledge_evaluation.ipynb`
Tests the baseline model's ability to answer domain-specific questions. It uses metrics like keyword matching, semantic similarity (using `sentence-transformers`), and an LLM-based judge to establish a performance floor.

### 2. `02_synthetic_data_preparation.ipynb`
Automates the creation of a fine-tuning dataset:
- **Scrape**: Extracts clean text from Sherlock Holmes Wikipedia articles.
- **Generate**: Uses a teacher model to produce question-answer pairs.
- **Curate**: Filters out low-quality examples using an AI judge.

### 3. `03_fine_tuning_QA.ipynb`
Covers the training process using Hugging Face's `transformers`, `peft`, and `trl`. It applies **QLoRA** (4-bit quantization) to make training possible on consumer-grade hardware.

### 5. `05_generate_sentiment_explanations.ipynb`
Moves beyond simple labels. It "distills" financial expertise from a 7B parameter model into a dataset of explanations, focusing on revenue, profitability, and market positioning.

### 6. `06_fine_tuning_sentiment.ipynb`
Trains the smaller Gemma 3 1B model to think before it classifies. By learning to generate explanations, the model becomes more robust and transparent in its predictions.

### 7. `07_sentiment_evaluation.ipynb`
A deep dive into model validation. It uses a "Judge Model" to grade the quality of generated reasoning, ensuring the fine-tuned model isn't just getting the right answer for the wrong reasons.

## Results
The workshop demonstrates how a relatively small amount of high-quality synthetic data (curated from reliable sources) can significantly boost a model's performance in niche domains and complex reasoning tasks.
