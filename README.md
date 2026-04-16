# Fine-Tuning Language Models Workshop

This project provides a hands-on workshop for fine-tuning large language models (LLMs). It demonstrates a complete workflow, from evaluating a base model to generating a synthetic dataset, fine-tuning the model, and evaluating its improved performance. 

The workshop covers three primary tracks:
1.  **Sherlock Holmes Expert**: Domain-specific knowledge injection via QA fine-tuning.
2.  **Tunix-Med Medical Expert**: High-performance domain adaptation using JAX/Tunix for cardiology.
3.  **Reasoned Financial Sentiment**: Enhancing classification with data augmentation and Chain-of-Thought (CoT) reasoning.

## Compatibility and Hardware Requirements

The examples in this repository are optimized for the following environments:

*   **NVIDIA GPU systems**: 16GB VRAM or more (e.g., RTX 3090/4080, A10/L4).
*   **Google Colab**: Compatible with the free **T4 GPU** tier.
*   **macOS (Apple Silicon)**: M-series chips with at least **16GB of unified memory**.

## Workshop Workflow

The project is divided into twelve Jupyter notebooks, organized into three tracks:

### Track 1: Sherlock Holmes Expert (Knowledge Injection)
1.  **`01_knowledge_evaluation.ipynb`**: Evaluates the baseline knowledge of the pre-trained `google/gemma-3-1b-it` model on Sherlock Holmes lore.
2.  **`02_synthetic_data_preparation.ipynb`**: Demonstrates how to scrape Wikipedia and use a teacher model to generate and curate a high-quality QA dataset.
3.  **`03_fine_tuning_QA.ipynb`**: Performs Supervised Fine-Tuning (SFT) using QLoRA to inject domain-specific knowledge.
4.  **`04_knowledge_evaluation_final.ipynb`**: Re-evaluates the fine-tuned model to quantify the "knowledge gain" compared to the baseline.

### Track 2: Tunix-Med Medical Expert (Cardiology QA)
5.  **`05_medical_baseline_evaluation.ipynb`**: Establishes a baseline for cardiology knowledge using specialized medical metrics and an AI judge.
6.  **`06_medical_synthetic_data.ipynb`**: Automates the creation of a high-quality medical QA dataset focused on Cardiology pathologies.
7.  **`07_tunix_sft_training.ipynb`**: Fine-tunes Gemma 3 using **Tunix** (JAX/Flax) for highly efficient, scalable training with streaming data.
8.  **`08_medical_evaluation_final.ipynb`**: Evaluates the medical expert model on held-out clinical questions to verify factual accuracy.

### Track 3: Reasoned Financial Sentiment (Reasoning & Classification)
9.  **`09_augment_sentiment_data.ipynb`**: Implements a robust data augmentation pipeline to anonymize PII and diversify numerical figures in financial headlines.
10. **`10_generate_sentiment_explanations.ipynb`**: Uses a teacher model (**Qwen 2.5 7B**) to generate deep financial explanations for sentiment labels.
11. **`11_fine_tuning_sentiment.ipynb`**: Fine-tunes **Gemma 3 1B** to predict both sentiment and the underlying financial reasoning (CoT).
12. **`12_sentiment_evaluation.ipynb`**: Validates the model on accuracy, tag integrity, and reasoning quality using an LLM-as-a-judge.

## Getting Started

To run the notebooks, ensure you have a compatible environment. It is also necessary to have a Hugging Face account and to accept the Google Gemma model licence. For detailed, step-by-step assistance, please refer to the `hugging_face_setup_guide.pdf` included in this repository. 

You can install the necessary dependencies using the provided script:

```bash
sh install.sh
```

## Results
The workshop demonstrates how high-quality synthetic data and parameter-efficient fine-tuning (LoRA/QLoRA) can transform a general-purpose model into a specialized expert capable of complex reasoning and deep domain knowledge.
