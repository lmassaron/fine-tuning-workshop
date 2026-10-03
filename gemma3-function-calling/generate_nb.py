import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1
nb.cells.append(nbf.v4.new_markdown_cell("# Fine-Tuning Gemma 3 270M for Function Calling\n\nThis notebook demonstrates how to fine-tune the `google/gemma-3-270m-it` model for function calling using LoRA."))

# Cell 2
nb.cells.append(nbf.v4.new_code_cell("!pip install -q -U transformers accelerate datasets peft trl"))

# Cell 3
imports_and_config = """import os
from enum import Enum
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

set_seed(42)

class ChatmlSpecialTokens(str, Enum):
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
    model_name = "google/gemma-3-270m-it"
    dataset_name = "lmassaron/hermes-function-calling-v1"
    output_dir = "gemma-3-270M-it-function_calling"
    
    lora_arguments = {
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": [
            "embed_tokens", "q_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj", "o_proj", "lm_head"
        ],
    }
    
    training_arguments = {
        "num_train_epochs": 1,
        "max_steps": -1, # Full 1 epoch run
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 2048,
        "packing": True,
        "optim": "adamw_torch_fused",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "logging_steps": 10,
        "report_to": "none", # Disabled tensorboard to avoid extra dependencies during notebook run
        "loss_type": "nll",
    }

config = Config()

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, dtype: {compute_dtype}")"""
nb.cells.append(nbf.v4.new_code_cell(imports_and_config))

# Cell 4
load_model = """# Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    pad_token=ChatmlSpecialTokens.pad_token.value,
    additional_special_tokens=ChatmlSpecialTokens.list(),
)

tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn><eos>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"

# Load Model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=compute_dtype,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    device_map=device, 
)

model.resize_token_embeddings(len(tokenizer))
print("Model loaded and token embeddings resized.")"""
nb.cells.append(nbf.v4.new_code_cell(load_model))

# Cell 5
nb.cells.append(nbf.v4.new_markdown_cell("## Pre-Training Evaluation\n\nLet's see how the baseline model responds to a prompt requiring a tool call."))

# Cell 6
eval_code = """eval_prompt = [
    {"role": "user", "content": '''You are a helpful assistant with access to the following tools:
<tools>
[{"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}}, "required": ["location"]}}]
</tools>
What's the weather like in Paris?'''}
]

text = tokenizer.apply_chat_template(eval_prompt, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

print("--- BASE MODEL GENERATION ---")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False))"""
nb.cells.append(nbf.v4.new_code_cell(eval_code))

# Cell 7
nb.cells.append(nbf.v4.new_markdown_cell("## Dataset & Training\n\nWe will now prepare the dataset and launch the training process using SFTTrainer."))

# Cell 8
dataset_setup = """print("Preparing dataset...")
def preprocess_and_filter(sample):
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

dataset_train = data.train_test_split(test_size=0.2, shuffle=True, seed=0)
train_data = dataset_train["train"]
eval_data = dataset_train["test"]
print(f"Train size: {len(train_data)}, Validation size: {len(eval_data)}")"""
nb.cells.append(nbf.v4.new_code_cell(dataset_setup))

# Cell 8.5
nb.cells.append(nbf.v4.new_markdown_cell("## Quantitative Pre-Training Evaluation\n\nWe define an exact match evaluation function to test accuracy on a subset of validation data."))

# Cell 8.6
eval_func_code = """from tqdm import tqdm

def evaluate_exact_match(dataset, model, tokenizer, num_samples=100):
    correct_tool, total_tool = 0, 0
    correct_chat, total_chat = 0, 0
    
    # Take a small subset for quick evaluation
    eval_subset = dataset.select(range(min(num_samples, len(dataset))))
    
    for item in tqdm(eval_subset, desc="Evaluating"):
        # Handle both datasets formats ("messages" or "conversations")
        conversations = item.get("messages", item.get("conversations"))
        if not conversations or conversations[-1]["role"] != "model":
            continue
            
        target_message = conversations[-1]["content"].strip()
        query_messages = conversations[:-1]
        
        # Prepare inputs
        text = tokenizer.apply_chat_template(query_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        generated_raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
        
        # Clean outputs for exact match comparison
        generated_clean = generated_raw.replace(tokenizer.eos_token, "").strip()
        expected_clean = target_message.replace(tokenizer.eos_token, "").strip()
        
        is_tool = "<tool_call>" in expected_clean
        is_correct = (expected_clean == generated_clean)
        
        if is_tool:
            total_tool += 1
            if is_correct: correct_tool += 1
        else:
            total_chat += 1
            if is_correct: correct_chat += 1
            
    tool_acc = correct_tool / total_tool if total_tool > 0 else 0
    chat_acc = correct_chat / total_chat if total_chat > 0 else 0
    
    print(f"\\nTool Calling Exact Match Accuracy: {tool_acc:.2%} ({correct_tool}/{total_tool})")
    print(f"General Chat Exact Match Accuracy: {chat_acc:.2%} ({correct_chat}/{total_chat})")
    return tool_acc, chat_acc

print("--- QUANTITATIVE PRE-TRAINING EVALUATION ---")
evaluate_exact_match(eval_data, model, tokenizer, num_samples=50)"""
nb.cells.append(nbf.v4.new_code_cell(eval_func_code))

# Cell 9
train_setup = """model.config.use_cache = False

peft_config = LoraConfig(
    r=config.lora_arguments["r"],
    lora_alpha=config.lora_arguments["lora_alpha"],
    lora_dropout=config.lora_arguments["lora_dropout"],
    target_modules=config.lora_arguments["target_modules"],
    task_type="CAUSAL_LM",
    bias="none",
    ensure_weight_tying=True,
)

training_args = SFTConfig(
    output_dir=config.output_dir,
    dataset_text_field="text",
    **config.training_arguments
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Starting training process...")
trainer.train()"""
nb.cells.append(nbf.v4.new_code_cell(train_setup))

# Cell 10
nb.cells.append(nbf.v4.new_markdown_cell("## Post-Training Evaluation\n\nNow, let's see how the fine-tuned model responds to the same prompt."))

# Cell 11
post_eval = """print("--- FINE-TUNED MODEL GENERATION ---")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False))"""
nb.cells.append(nbf.v4.new_code_cell(post_eval))

# Cell 12
nb.cells.append(nbf.v4.new_markdown_cell("## Quantitative Post-Training Evaluation\n\nLet's evaluate exact match accuracy after fine-tuning."))

# Cell 13
post_eval_quant = """print("--- QUANTITATIVE POST-TRAINING EVALUATION ---")
evaluate_exact_match(eval_data, model, tokenizer, num_samples=50)"""
nb.cells.append(nbf.v4.new_code_cell(post_eval_quant))

# Save notebook
with open('gemma3_270m_function_calling.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
