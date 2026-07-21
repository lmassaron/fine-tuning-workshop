# Walkthrough: Reasoned Financial Sentiment Fine-Tuning (Chain-of-Thought)

This walkthrough provides a complete technical analysis, implementation details, training dynamics, and empirical evaluation results for [financial_sentiment_cot.ipynb](file:///home/lmassaron/code/sft-examples/financial_sentiment_cot.ipynb).

---

## 1. Executive Summary & Goal

Standard scalar sentiment classifiers output rigid labels (`positive`, `neutral`, `negative`) without providing underlying financial rationale. This track fine-tunes **`google/gemma-4-E2B-it`** using 4-bit QLoRA and Hugging Face `trl` on the `lmassaron/FinancialPhraseBank_explained` dataset.

### Key Objectives
1. **Chain-of-Thought (CoT) Reasoning**: Force the LLM to generate an explanatory rationale *before* declaring its sentiment decision.
2. **Dual XML Tag Integrity**: Enforce strict `<sentiment>` and `<reasoning>` XML tags without external runtime parsers or constrained decoding libraries.
3. **VRAM-Constrained Fine-Tuning**: Optimize memory footprint to fit training within 16GB VRAM.

---

## 2. Technical Stack & Architecture

- **Base Model**: `google/gemma-4-E2B-it` (Gemma 4 E2B Instruct)
- **Quantization**: 4-bit NormalFloat (NF4) via `BitsAndBytesConfig` with `torch.bfloat16` compute dtype
- **PEFT LoRA Config**:
  - Rank $r=8$, $\alpha=32$, Dropout $= 0.05$
  - Target Modules: `'all-linear'` (all query, key, value, output, and MLP projections)
- **Dataset**: `lmassaron/FinancialPhraseBank_explained`
- **Frameworks**: Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`, `SFTConfig`)

---

## 3. Practical Implementation & Prompt Formatting

### System Prompt
```python
SYSTEM_PROMPT = (
    "You are a financial analyst with expertise in equity markets and corporate finance.\n"
    "Analyze the following financial news headline and determine its market sentiment "
    "from an investor's perspective.\n\n"
    "Classify the sentiment as positive, neutral, or negative based on the likely "
    "impact on stock price, investor confidence, or financial performance.\n\n"
    "Respond using exactly these two tags:\n"
    "<sentiment>positive|neutral|negative</sentiment>\n"
    "<reasoning>brief financial explanation</reasoning>\n"
)
```

### Chat Template Mapping
We map each example into Gemma's chat template format using `tokenizer.apply_chat_template(..., tokenize=False)` so that `SFTTrainer` handles tokenization cleanly:
```python
def format_prompt(example):
    messages = [
        {
            "role": "user",
            "content": (
                SYSTEM_PROMPT + f'\nHeadline: "{example["text"]}"'
            ),
        },
        {
            "role": "assistant",
            "content": (
                f'<sentiment>{example["sentiment"]}</sentiment>\n'
                f'<reasoning>{example["reasoning"]}</reasoning>'
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": prompt}
```

### Critical Practicality: Preventing Double BOS Tokens
When passing pre-formatted chat template strings to `SFTTrainer`, set `dataset_kwargs={'add_special_tokens': False}` in `SFTConfig`. Otherwise, `SFTTrainer` prepends a second BOS token, corrupting positional embeddings.

---

## 4. Training Configuration & Hyperparameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Max Sequence Length** | `512` | Fits full news headline + system prompt + CoT explanation |
| **Batch Size** | `2` per device | Fits within 16GB VRAM limit |
| **Gradient Accumulation** | `4` | Yields an **Effective Batch Size of 8** |
| **Learning Rate** | `2e-4` | Optimal learning rate for QLoRA adaptation |
| **LR Scheduler** | `cosine` with `0.1` warmup | Smooth convergence during training epoch |
| **Optimizer** | `adamw_torch_fused` | Fuses kernel updates to reduce CUDA overhead |
| **Gradient Checkpointing**| Enabled (`use_reentrant=False`) | Discards intermediate activations during forward pass |
| **KV Cache** | Disabled (`use_cache=False`) | Reduces memory consumption during backward pass |

---

## 5. Quantitative Results & Evaluation

Evaluation was conducted on the 484 original (non-augmented) headlines in the test split using left-padded batch generation (`batch_size=16`):

### Global Performance Metrics
- **Sentiment Classification Accuracy**: **81.82%**
- **Tag Integrity (Valid Output Formats)**: **94.83%**

### Scikit-Learn Classification Report
```
              precision    recall  f1-score   support

    negative       0.88      0.85      0.87        61
     neutral       0.84      0.89      0.87       287
    positive       0.91      0.65      0.76       136

    accuracy                           0.82       484
   macro avg       0.66      0.60      0.62       484
weighted avg       0.87      0.82      0.84       484
```

### Analysis of Results
- **High Precision**: Precision across all classes remains very high (88% negative, 84% neutral, 91% positive).
- **Conservative Shift**: The lower recall on positive headlines (65%) reflects a conservative financial bias: ambiguous corporate news is classified as `neutral` rather than falsely speculative `positive`.

---

## 6. Concrete Inference Output Examples

### Example 1: Positive Earnings Growth
- **Headline**: *"Operating profit increased to EUR 14.5 mn from EUR 10.2 mn in the previous year."*
- **Model Output**:
  ```xml
  <sentiment>positive</sentiment>
  <reasoning>A substantial increase in operating profit from EUR 10.2 mn to EUR 14.5 mn demonstrates strong earnings growth and improved operational efficiency.</reasoning>
  ```

### Example 2: Negative Cost Expansion
- **Headline**: *"The company announced the lay-off of 150 employees due to declining demand in European markets."*
- **Model Output**:
  ```xml
  <sentiment>negative</sentiment>
  <reasoning>Workforce reductions triggered by falling regional market demand reflect revenue headwinds and operational distress.</reasoning>
  ```

---

## 7. Artifact Outputs
- Fine-tuned adapter saved at: `gemma4-sentiment-lora-adapter/`
