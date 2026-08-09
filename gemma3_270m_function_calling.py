import os
from enum import Enum
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# Set seed for reproducibility
set_seed(42)

class ChatmlSpecialTokens(str, Enum):
    """Enum class defining special tokens used for function calling in the ChatML format."""
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

class Config:
    """Configuration class for the fine-tuning experiment."""
    model_name = "google/gemma-3-270m-it"
    dataset_name = "lmassaron/hermes-function-calling-v1"
    output_dir = "gemma-3-270M-it-function_calling"
    
    # LoRA parameters for Parameter-Efficient Fine-Tuning
    lora_arguments = {
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": [
            "embed_tokens", "q_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj", "o_proj", "lm_head"
        ],
    }
    
    # SFTTrainer training arguments
    training_arguments = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 4096,
        "packing": False,
        "optim": "adamw_torch_fused",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "gradient_checkpointing": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "logging_steps": 5,
        "report_to": "tensorboard",
    }
    
def main():
    config = Config()
    
    # 1. Determine Compute Dtype (use bfloat16 if Ampere architecture is available)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, dtype: {compute_dtype}")

    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    
    # Configure the Chat Template (essential for instruction following)
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn><eos>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"

    # 3. Load Model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=compute_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="cpu", # Initial load to CPU to save GPU memory during init
    )
    
    # Resize embeddings for new special tokens and move model to GPU
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # 4. Prepare Dataset
    print("Preparing dataset...")
    def preprocess_and_filter(sample):
        """Formats the conversation and filters out sequences that are too long."""
        messages = sample["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text, truncation=False)
        
        if len(tokens) <= config.training_arguments["max_length"]:
            return {"text": text}
        else:
            return None

    data = (
        load_dataset(config.dataset_name, split="train")
        .rename_column("conversations", "messages")
        .map(preprocess_and_filter, remove_columns="messages")
        .filter(lambda x: x is not None, keep_in_memory=False)
    )
    
    # Create train and validation splits
    dataset_train = data.train_test_split(test_size=0.2, shuffle=True, seed=0)
    train_data = dataset_train["train"]
    eval_data = dataset_train["test"]
    print(f"Train size: {len(train_data)}, Validation size: {len(eval_data)}")

    # 5. Initialize LoRA Configuration
    peft_config = LoraConfig(
        r=config.lora_arguments["r"],
        lora_alpha=config.lora_arguments["lora_alpha"],
        lora_dropout=config.lora_arguments["lora_dropout"],
        target_modules=config.lora_arguments["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    # 6. Initialize Training Configuration
    training_args = SFTConfig(
        output_dir=config.output_dir,
        dataset_text_field="text", # The key containing our processed text
        **config.training_arguments
    )

    # 7. Setup Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 8. Train the model
    print("Starting training process...")
    trainer.train()
    
    # 9. Save the fine-tuned model
    print(f"Training complete. Saving model adapters to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
