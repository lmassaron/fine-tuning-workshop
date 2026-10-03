import os
import torch
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")


def test_financial_sentiment():
    print("\n=== Testing Track 1: Financial Sentiment ===")
    MODEL_ID = "google/gemma-4-E2B-it"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit...")
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto"
    )

    print("Setting up PEFT...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print("Financial model and PEFT loaded successfully!")

    print("Checking sample prompt tokenization...")
    test_headline = (
        "Nokia reports Q4 profit beat but warns of challenging environment in 2026."
    )
    SYSTEM_PROMPT = "You are a financial analyst... Classify sentiment as positive, neutral, or negative."
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + f'\nHeadline: "{test_headline}"'}
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    print("Inputs shape:", inputs.shape)
    print("Track 1 Dry-Run passed successfully!\n")


def test_medical_expert():
    print("\n=== Testing Track 2: Cardiology Medical Expert (Unsloth) ===")
    MODEL_ID = "unsloth/Phi-4-mini-instruct"

    print("Loading Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    print("Getting PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("Unsloth model and PEFT loaded successfully!")

    print("Checking test inference setup...")
    FastLanguageModel.for_inference(model)
    test_messages = [
        {"role": "system", "content": "You are a cardiology assistant."},
        {
            "role": "user",
            "content": "What is the primary treatment for acute myocardial infarction?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    print("Inputs shape:", inputs.shape)
    print("Track 2 Dry-Run passed successfully!\n")


if __name__ == "__main__":
    test_financial_sentiment()
    test_medical_expert()
