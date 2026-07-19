import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, disable_caching
disable_caching()
from trl import SFTTrainer
from trl.trainer import SFTConfig

# --- CONFIGURATION ---
BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
HF_USERNAME = "lmassaron" # Replace with your Hugging Face username

MAX_SEQ_LENGTH = 2048

# Standard chat template wrapper
def create_prompt(instruction, output):
    return f"<|im_start|>system\nYou are a specialized AI agent.<|im_end|>\n<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
# ---------------------

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
        except:
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
    print(f"\n--- Training {output_name} ---")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    tokenizer.eos_token = "<|im_end|>"
    
    def tokenize_dataset(examples):
        return tokenizer(text=examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)
    
    dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = MAX_SEQ_LENGTH,
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = None,
            eos_token = "<|im_end|>",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            max_steps = 60,
            learning_rate = 2e-4,
            bf16 = True,
            logging_steps = 1,
            output_dir = f"outputs_{output_name}",
        ),
    )
    
    trainer.train()
    
    local_path = f"adapters/{output_name}-lora"
    print(f"Saving to {local_path} and pushing to Hugging Face Hub: {push_repo_name}...")
    
    model.save_pretrained(local_path)
    try:
        model.push_to_hub(push_repo_name)
        tokenizer.push_to_hub(push_repo_name)
    except Exception as e:
        print(f"Skipping Hugging Face push due to error (perhaps not logged in?): {e}")

def run_pipeline():
    print("Initializing LoRA Factory Pipeline (Unsloth Edition)...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # 1. PLANNER
    # planner_ds = load_dataset("Open-Orca/OpenOrca", split="train[:5000]")
    # planner_ds = planner_ds.filter(lambda x: x["system_prompt"] != "")
    # planner_ds = planner_ds.select(range(min(1000, len(planner_ds))))
    # planner_ds = planner_ds.map(format_planner_data, batched=True, remove_columns=planner_ds.column_names)
    
    # train_lora(model, tokenizer, planner_ds, "planner", f"{HF_USERNAME}/planner-lora")
    
    # Note: Unsloth might require deleting the trainer to free memory.
    # import gc; gc.collect(); torch.cuda.empty_cache()

    # 2. CODER / TOOL USER
    # coder_ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train[:1000]")
    # coder_ds = coder_ds.map(format_coder_data, batched=True, remove_columns=coder_ds.column_names)
    
    # train_lora(model, tokenizer, coder_ds, "coder", f"{HF_USERNAME}/coder-lora")
    
    import gc; gc.collect(); torch.cuda.empty_cache()

    # 3. REVIEWER
    reviewer_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:1000]")
    reviewer_ds = reviewer_ds.map(format_reviewer_data, batched=True, remove_columns=reviewer_ds.column_names)
    
    train_lora(model, tokenizer, reviewer_ds, "reviewer", f"{HF_USERNAME}/reviewer-lora")

    print("\n✨ All LoRAs trained and saved successfully!")

if __name__ == "__main__":
    run_pipeline()
