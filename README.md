# Fine-Tuning Language Models Workshop

This project provides a hands-on workshop for fine-tuning large language models (LLMs). It demonstrates a complete workflow, from evaluating a base model to generating a synthetic dataset, fine-tuning the model, and evaluating its improved performance. The primary example focuses on creating a Sherlock Holmes expert model, with an additional example of sentiment analysis in the financial domain.

## Workflow

The project is divided into five Jupyter notebooks, each covering a specific step in the fine-tuning process:

1.  **`01_knowledge_evaluation.ipynb`**: Evaluates the baseline knowledge of a pre-trained language model on a specific domain (Sherlock Holmes).
2.  **`02_synthetic_data_preparation.ipynb`**: Generates a synthetic dataset of questions and answers about Sherlock Holmes to be used for fine-tuning.
3.  **`03_fine_tuning_QA.ipynb`**: Fine-tunes the language model on the synthetic dataset to enhance its domain-specific knowledge.
4.  **`04_knowledge_evaluation_final.ipynb`**: Evaluates the fine-tuned model to measure the improvement in its knowledge and performance.
5.  **`05_fine_tuning_sentiment.ipynb`**: Provides an additional example of fine-tuning a model for sentiment analysis on financial news.

## Getting Started

To run the notebooks in this project, you need to have Python and Jupyter Notebook installed. You will also need to install the required libraries, which can be done by running the `install.sh` script:

```bash
sh install.sh
```

## Notebooks

### 1. `01_knowledge_evaluation.ipynb`

This notebook evaluates the baseline knowledge of a pre-trained language model on the Sherlock Holmes domain. It uses a series of questions to test the model's ability to answer correctly and measures its performance using various metrics, including keyword matching, semantic similarity, and an AI-based judge.

### 2. `02_synthetic_data_preparation.ipynb`

This notebook demonstrates how to create a synthetic dataset for fine-tuning. It scrapes Wikipedia for information about Sherlock Holmes, processes the text, and then uses a language model to generate question-and-answer pairs. This synthetic dataset is then used to fine-tune the model.

### 3. `03_fine_tuning_QA.ipynb`

This notebook covers the fine-tuning process. It uses the synthetic dataset created in the previous step to fine-tune the pre-trained language model. The notebook demonstrates how to use the Hugging Face `transformers` and `trl` libraries to perform supervised fine-tuning.

### 4. `04_knowledge_evaluation_final.ipynb`

This notebook evaluates the fine-tuned model's performance. It uses the same evaluation metrics as the first notebook to measure the improvement in the model's knowledge and accuracy.

### 5. `05_fine_tuning_sentiment.ipynb`

This notebook provides a second example of fine-tuning a language model, this time for sentiment analysis. It uses a dataset of financial news headlines to fine-tune a model to classify the sentiment as positive, negative, or neutral.

## Results

The project demonstrates a significant improvement in the model's knowledge and performance after fine-tuning. The fine-tuned model is able to answer questions about Sherlock Holmes with much higher accuracy than the base model. The sentiment analysis example also shows that fine-tuning can be used to adapt a language model to a specific task and domain.

The evaluation results from the notebooks show the following improvements:

*   **Keyword Matching Accuracy**: The fine-tuned model shows a significant improvement in its ability to generate answers that contain the correct keywords.
*   **Semantic Similarity Accuracy**: The fine-tuned model's answers are more semantically similar to the expected answers.
*   **AI Judge Accuracy**: The fine-tuned model's answers are judged to be more accurate by an AI-based evaluator.
*   **Perplexity**: The fine-tuned model has a lower perplexity, indicating that it is more confident in its answers.