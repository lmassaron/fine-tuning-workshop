"""
Generate reasoned financial sentiment explanations for FinancialPhraseBank dataset.

Model: Qwen/Qwen2.5-7B-Instruct
- Fits comfortably in RTX 3090 (24GB) in nf4 (~6GB VRAM)
- Strong instruction following and financial reasoning

Output: HuggingFace dataset with original fields + 'explanation' column,
        saved locally and optionally pushed to the Hub.

Usage:
    pip install transformers datasets torch accelerate
    python generate_sentiment_explanations.py
"""

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import BitsAndBytesConfig

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "lmassaron/FinancialPhraseBank"
OUTPUT_PATH = "FinancialPhraseBank_explained"
HF_REPO_ID = "lmassaron/FinancialPhraseBank_explained"  # Set to "your-username/dataset-name" to push to Hub
BATCH_SIZE = 16  # Adjust down if you hit OOM
MAX_NEW_TOKENS = 512  # Enough for a concise but thorough explanation
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # saves a bit more memory
    bnb_4bit_quant_type="nf4",  # best quality for 4-bit
)

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior financial analyst with deep expertise in equity markets, "
    "corporate finance, and macroeconomics. "
    "Your task is to explain, in 2-4 sentences, why a financial news headline "
    "carries a specific market sentiment. "
    "Focus strictly on the financial implications: how the news affects revenue, "
    "profitability, cash flow, investor confidence, or market positioning. "
    "Be concise and precise. Do not repeat the sentence verbatim."
)


def build_prompt(sentence: str, sentiment: str) -> str:
    return (
        f'Financial news headline:\n"{sentence}"\n\n'
        f"This headline has been classified as **{sentiment}** sentiment "
        f"from a financial markets perspective.\n\n"
        f"Explain why, focusing on the financial implications for investors, "
        f"the company, or the broader market."
    )


# ── Model loading ──────────────────────────────────────────────────────────────


def load_model(model_id: str):
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # hf_device_map only exists with multi-device or quantized loading
    device_info = getattr(model, "hf_device_map", None) or str(model.device)
    print(f"Model loaded. Device: {device_info}")

    return tokenizer, model


# ── Inference ──────────────────────────────────────────────────────────────────


def generate_explanations_batch(
    tokenizer,
    model,
    sentences: list[str],
    labels: list[int],
) -> list[str]:
    """Run inference on a batch and return explanation strings."""

    # Build chat-formatted prompts
    chats = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(s, LABEL_MAP[l])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s, l in zip(sentences, labels)
    ]

    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy — deterministic and fast
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    input_len = inputs["input_ids"].shape[1]
    explanations = tokenizer.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
    )
    return [e.strip() for e in explanations]


def process_dataset(tokenizer, model, dataset) -> list[dict]:
    """Iterate over the dataset in batches, collecting results."""
    results = []
    sentences = dataset["sentence"]
    labels = dataset["label"]
    n = len(sentences)

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Generating explanations"):
        batch_sentences = sentences[start : start + BATCH_SIZE]
        batch_labels = labels[start : start + BATCH_SIZE]

        explanations = generate_explanations_batch(
            tokenizer, model, batch_sentences, batch_labels
        )

        for sentence, label, explanation in zip(
            batch_sentences, batch_labels, explanations
        ):
            results.append(
                {
                    "sentence": sentence,
                    "label": label,
                    "sentiment": LABEL_MAP[label],
                    "explanation": explanation,
                }
            )

    return results


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # 1. Load dataset
    print(f"Loading dataset: {DATASET_ID}")
    raw = load_dataset(DATASET_ID)
    print(raw)

    # 2. Load model once — reused across all splits
    tokenizer, model = load_model(MODEL_ID)

    # 3. Process every split, preserving the original DatasetDict structure
    output_splits = {}
    for split_name, data in raw.items():
        print(f"\n── Processing split: '{split_name}' ({len(data)} examples) ──")
        results = process_dataset(tokenizer, model, data)
        output_splits[split_name] = Dataset.from_list(results)

    output_ds = DatasetDict(output_splits)

    # 4. Print summary and one example
    print("\nOutput dataset:")
    print(output_ds)
    first_split = list(output_splits.keys())[0]
    ex = output_ds[first_split][0]
    print(f"\nExample from '{first_split}':")
    print(f"  Sentence  : {ex['sentence']}")
    print(f"  Sentiment : {ex['sentiment']} (label={ex['label']})")
    print(f"  Explanation:\n    {ex['explanation']}")

    # 5. Save locally
    output_ds.save_to_disk(OUTPUT_PATH)
    print(f"\nDataset saved to: {OUTPUT_PATH}")

    # 6. Optionally push to Hub
    if HF_REPO_ID:
        print(f"Pushing to Hub: {HF_REPO_ID}")
        output_ds.push_to_hub(HF_REPO_ID)
        print("Upload complete!")


if __name__ == "__main__":
    main()
