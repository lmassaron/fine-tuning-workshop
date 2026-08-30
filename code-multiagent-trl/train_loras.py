import os
import gc
import torch
from datasets import load_dataset, disable_caching
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

disable_caching()

# --- CONFIGURATION ---
BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
HF_USERNAME = "lmassaron"  # Replace with your Hugging Face username
MAX_SEQ_LENGTH = 2048


# Standard chat template wrapper
def create_prompt(instruction, output):
    return f"<|im_start|>system\nYou are a specialized AI agent.<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def format_planner_data(examples):
    texts = []
    for q, r in zip(examples["question"], examples["response"]):
        if "step" in q.lower() or "plan" in q.lower():
            texts.append(create_prompt(q, r))
    return {"text": texts}


def format_coder_data(examples):
    texts = []
    for conversations in examples.get("conversations", []):
        try:
            instruction = conversations[0]["value"]
            output = conversations[1]["value"]
            texts.append(create_prompt(instruction, output))
        except Exception:
            continue
    return {"text": texts}


def format_reviewer_data(examples):
    texts = []
    for code, review in zip(examples.get("query", []), examples.get("answer", [])):
        instruction = f"Review this code snippet:\n{code}"
        output = f"Review Analysis:\n{review}"
        texts.append(create_prompt(instruction, output))
    return {"text": texts}


def train_lora(model, tokenizer, dataset, output_name, push_repo_name):
    print(f"\n--- Training {output_name} with TRL SFTTrainer ---")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            dataset_kwargs={"skip_prepare_dataset": False},
            dataset_num_proc=None,
            eos_token="<|im_end|>",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=60,
            learning_rate=2e-4,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=f"outputs_{output_name}",
        ),
    )

    trainer.train()

    local_path = f"adapters/{output_name}-lora"
    print(
        f"Saving to {local_path} and pushing to Hugging Face Hub: {push_repo_name}..."
    )

    trainer.model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    try:
        trainer.model.push_to_hub(push_repo_name)
        tokenizer.push_to_hub(push_repo_name)
    except Exception as e:
        print(f"Skipping Hugging Face push due to error (perhaps not logged in?): {e}")


def run_pipeline():
    print("Initializing LoRA Factory Pipeline (TRL Edition)...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token = "<|im_end|>"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 1. PLANNER
    # planner_ds = load_dataset("Open-Orca/OpenOrca", split="train[:5000]")
    # planner_ds = planner_ds.filter(lambda x: x["system_prompt"] != "")
    # planner_ds = planner_ds.select(range(min(1000, len(planner_ds))))
    # planner_ds = planner_ds.map(format_planner_data, batched=True, remove_columns=planner_ds.column_names)
    # train_lora(model, tokenizer, planner_ds, "planner", f"{HF_USERNAME}/planner-lora")

    # 2. CODER / TOOL USER
    # coder_ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train[:1000]")
    # coder_ds = coder_ds.map(format_coder_data, batched=True, remove_columns=coder_ds.column_names)
    # train_lora(model, tokenizer, coder_ds, "coder", f"{HF_USERNAME}/coder-lora")

    gc.collect()
    torch.cuda.empty_cache()

    # 3. REVIEWER
    reviewer_ds = load_dataset(
        "m-a-p/CodeFeedback-Filtered-Instruction", split="train[:1000]"
    )
    reviewer_ds = reviewer_ds.map(
        format_reviewer_data, batched=True, remove_columns=reviewer_ds.column_names
    )

    train_lora(
        model, tokenizer, reviewer_ds, "reviewer", f"{HF_USERNAME}/reviewer-lora"
    )

    print("\n✨ All LoRAs trained and saved successfully with TRL!")


if __name__ == "__main__":
    run_pipeline()
