# Fine-Tuning Language Models Workshop

This project provides a hands-on workshop for fine-tuning large language models (LLMs). It demonstrates a complete workflow, from evaluating a base model to generating a synthetic dataset, fine-tuning the model, and evaluating its improved performance. The primary example focuses on creating a Sherlock Holmes expert model, with an additional example of sentiment analysis in the financial domain.

## Compatibility and Hardware Requirements

The examples in this repository are optimized for the following environments:

*   **NVIDIA GPU systems**: 16GB VRAM or more (e.g., RTX 3090/4080, A10/L4).
*   **Google Colab**: Compatible with the free **T4 GPU** tier.
*   **macOS (Apple Silicon)**: M-series chips with at least **16GB of unified memory**.

## Workflow

The project is divided into five Jupyter notebooks, each covering a specific step in the fine-tuning process:

1.  **`01_knowledge_evaluation.ipynb`**: Evaluates the baseline knowledge of a pre-trained language model (e.g., Gemma-2b) on a specific domain (Sherlock Holmes).
2.  **`02_synthetic_data_preparation.ipynb`**: Demonstrates a full pipeline to scrape Wikipedia, and use `synthetic-data-kit` (backed by a local vLLM server) to generate and curate a high-quality QA dataset.
3.  **`03_fine_tuning_QA.ipynb`**: Performs Supervised Fine-Tuning (SFT) using QLoRA to enhance the model's domain-specific knowledge while staying within memory constraints.
4.  **`04_knowledge_evaluation_final.ipynb`**: Re-evaluates the fine-tuned model to measure the improvement in its knowledge and performance compared to the baseline.
5.  **`05_fine_tuning_sentiment.ipynb`**: Provides an additional example of fine-tuning for sentiment analysis on financial news headlines.

## Getting Started

To run the notebooks, ensure you have a compatible environment. You can install the necessary dependencies using the provided script:

```bash
sh install.sh
```

## Notebook Details

### 1. `01_knowledge_evaluation.ipynb`

Tests the baseline model's ability to answer domain-specific questions. It uses metrics like keyword matching, semantic similarity (using `sentence-transformers`), and an LLM-based judge to establish a performance floor.

### 2. `02_synthetic_data_preparation.ipynb`

Automates the creation of a fine-tuning dataset:
- **Scrape**: Extracts clean text from Sherlock Holmes-related Wikipedia articles.
- **Generate**: Uses `synthetic-data-kit` to produce question-answer pairs.
- **Curate**: Filters out low-quality examples using an AI judge.
- **Format**: Prepares the data for conversation-style fine-tuning.

### 3. `03_fine_tuning_QA.ipynb`

Covers the actual training process. It leverages Hugging Face's `transformers`, `peft`, and `trl` libraries to apply Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA, making it possible to train on consumer-grade hardware or free cloud instances.

### 4. `04_knowledge_evaluation_final.ipynb`

Validates the success of the fine-tuning. By running the same battery of tests as the first notebook, it quantifies the "knowledge gain" and improvement in response quality.

### 5. `05_fine_tuning_sentiment.ipynb`

A transfer learning example focusing on classification. It adapts a language model to identify sentiment (positive, negative, neutral) in financial headlines, demonstrating the versatility of fine-tuning.

## Results

The workshop demonstrates how a relatively small amount of high-quality synthetic data (curated from reliable sources like Wikipedia) can significantly boost a model's performance in a niche domain.

Key improvements typically observed:
*   **Higher Accuracy**: More frequent correct answers to factual questions.
*   **Better Alignment**: Responses follow the desired format and tone.
*   **Reduced Hallucination**: The model stays closer to the provided domain knowledge.