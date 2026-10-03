import nbformat as nbf
import os


def create_financial_sentiment_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Track 1: Reasoned Financial Sentiment Fine-Tuning\n"
            "This notebook demonstrates fine-tuning the **Gemma 4 E2B-IT** model on the "
            "`lmassaron/FinancialPhraseBank_explained` dataset using **TRL** and **LoRA** (4-bit QLoRA).\n\n"
            "### Objectives\n"
            "- Learn how to inject Chain-of-Thought (CoT) reasoning into a classification task.\n"
            "- Perform Supervised Fine-Tuning (SFT) using Hugging Face `trl` and `peft` libraries.\n"
            "- Optimize training parameters to fit within a 16GB VRAM constraint."
        ),
        nbf.v4.new_code_cell(
            "# Install extra dependencies if needed (e.g. on Google Colab)\n"
            "# %pip install -U transformers trl peft accelerate bitsandbytes datasets"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import torch\n"
            "import warnings\n"
            "from datasets import load_dataset\n"
            "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n"
            "from peft import LoraConfig\n"
            "from trl import SFTTrainer, SFTConfig\n\n"
            "warnings.filterwarnings('ignore')\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n\n"
            "# Check device and capability\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n"
            "print(f'Using device: {device} | Dtype: {compute_dtype}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Dataset\n"
            "We pull the `lmassaron/FinancialPhraseBank_explained` dataset which contains news sentences, ground-truth sentiments, and human-like explanations."
        ),
        nbf.v4.new_code_cell(
            "DATASET_ID = 'lmassaron/FinancialPhraseBank_explained'\n"
            "print(f'Loading {DATASET_ID}...')\n"
            "dataset = load_dataset(DATASET_ID)\n\n"
            "# Rename columns to standard names\n"
            "dataset = dataset.rename_columns({\n"
            "    'sentence': 'text',\n"
            "    'explanation': 'reasoning'\n"
            "})\n"
            "train_ds = dataset['train']\n"
            "eval_ds = dataset['validation']\n"
            "print(train_ds)\n"
            "print('Sample sentence:', train_ds[0]['text'])\n"
            "print('Sample sentiment:', train_ds[0]['sentiment'])\n"
            "print('Sample explanation:', train_ds[0]['reasoning'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Formatting Prompts using Chat Template\n"
            "We structure the news headline, instructions, and target sentiment/explanations into a chat template using Gemma 4's format."
        ),
        nbf.v4.new_code_cell(
            "SYSTEM_PROMPT = (\n"
            '    "You are a financial analyst with expertise in equity markets and corporate finance.\\n"\n'
            '    "Analyze the following financial news headline and determine its market sentiment "\n'
            '    "from an investor\'s perspective.\\n\\n"\n'
            '    "Classify the sentiment as positive, neutral, or negative based on the likely "\n'
            '    "impact on stock price, investor confidence, or financial performance.\\n\\n"\n'
            '    "Respond using exactly these two tags:\\n"\n'
            '    "<sentiment>positive|neutral|negative</sentiment>\\n"\n'
            '    "<reasoning>brief financial explanation</reasoning>\\n"\n'
            ")\n\n"
            "MODEL_ID = 'google/gemma-4-E2B-it'\n"
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
            "if tokenizer.pad_token is None:\n"
            "    tokenizer.pad_token = tokenizer.eos_token\n\n"
            "def format_prompt(example):\n"
            "    messages = [\n"
            "        {'role': 'user', 'content': SYSTEM_PROMPT + f'\\nHeadline: \"{example[\"text\"]}\"'},\n"
            "        {\n"
            "            'role': 'assistant',\n"
            "            'content': f'<sentiment>{example[\"sentiment\"]}</sentiment>\\n<reasoning>{example[\"reasoning\"]}</reasoning>'\n"
            "        }\n"
            "    ]\n"
            "    # Format without tokenizing so SFTTrainer can tokenize it\n"
            "    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n"
            "    return {'text': prompt}\n\n"
            "train_mapped = train_ds.map(format_prompt)\n"
            "eval_mapped = eval_ds.map(format_prompt)\n"
            "print('Mapped Prompt Preview:\\n', train_mapped[0]['text'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Load Model with 4-bit Quantization\n"
            "We load the base `google/gemma-4-E2B-it` model in 4-bit precision to fit within the 16GB VRAM constraint, and configure LoRA adapters targeting all linear modules."
        ),
        nbf.v4.new_code_cell(
            "quant_config = BitsAndBytesConfig(\n"
            "    load_in_4bit=True,\n"
            "    bnb_4bit_quant_type='nf4',\n"
            "    bnb_4bit_compute_dtype=compute_dtype,\n"
            ")\n\n"
            "model = AutoModelForCausalLM.from_pretrained(\n"
            "    MODEL_ID,\n"
            "    quantization_config=quant_config,\n"
            "    device_map='auto'\n"
            ")\n\n"
            "peft_config = LoraConfig(\n"
            "    r=8,\n"
            "    lora_alpha=32,\n"
            "    target_modules='all-linear',\n"
            "    lora_dropout=0.05,\n"
            "    bias='none',\n"
            "    task_type='CAUSAL_LM'\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Run SFT Trainer\n"
            "We configure the Hugging Face `trl` SFT Config and start fine-tuning. This is optimized for a batch size of 2 and gradient accumulation of 4 (effective batch size of 8)."
        ),
        nbf.v4.new_code_cell(
            "training_args = SFTConfig(\n"
            "    output_dir='gemma4-sentiment-lora',\n"
            "    dataset_text_field='text',\n"
            "    max_length=512,\n"
            "    num_train_epochs=1,\n"
            "    per_device_train_batch_size=2,\n"
            "    gradient_accumulation_steps=4,\n"
            "    learning_rate=2e-4,\n"
            "    optim='adamw_torch_fused',\n"
            "    gradient_checkpointing=True,\n"
            "    gradient_checkpointing_kwargs={'use_reentrant': False},\n"
            "    lr_scheduler_type='cosine',\n"
            "    warmup_ratio=0.1,\n"
            "    bf16=(compute_dtype == torch.bfloat16),\n"
            "    logging_steps=10,\n"
            "    eval_strategy='steps',\n"
            "    eval_steps=20,\n"
            "    save_steps=20,\n"
            "    report_to='none',\n"
            "    dataset_kwargs={\n"
            "        'add_special_tokens': False,\n"
            "    },\n"
            ")\n\n"
            "trainer = SFTTrainer(\n"
            "    model=model,\n"
            "    train_dataset=train_mapped,\n"
            "    eval_dataset=eval_mapped,\n"
            "    peft_config=peft_config,\n"
            "    processing_class=tokenizer,\n"
            "    args=training_args,\n"
            ")\n\n"
            "# Disable KV cache during training to save VRAM\n"
            "model.config.use_cache = False\n\n"
            "trainer.train()\n\n"
            "# Save the fine-tuned adapter\n"
            "trainer.model.save_pretrained('gemma4-sentiment-lora-adapter')\n"
            "tokenizer.save_pretrained('gemma4-sentiment-lora-adapter')\n"
            "print('Adapter successfully saved!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Evaluation and Inference\n"
            "We load the fine-tuned adapter, merge it with the base model, and run batch evaluation on the original (non-augmented) test set Headlines. "
            "We measure accuracy, tag format integrity, and output a detailed classification report."
        ),
        nbf.v4.new_code_cell(
            "from peft import PeftModel\n"
            "from sklearn.metrics import accuracy_score, classification_report\n"
            "from tqdm import tqdm\n"
            "import re\n"
            "import pandas as pd\n\n"
            "# 1. Load fine-tuned model\n"
            "print('Loading merged base model and adapter...')\n"
            "eval_model = PeftModel.from_pretrained(model, 'gemma4-sentiment-lora-adapter')\n"
            "eval_model = eval_model.eval()\n\n"
            "# 2. Prepare test data (filtering out augmented samples for clean evaluation)\n"
            "test_df = dataset['test'].to_pandas()\n"
            "if 'is_augmented' in test_df.columns:\n"
            "    test_df = test_df[test_df['is_augmented'] == False].reset_index(drop=True)\n"
            "print(f'Evaluating on {len(test_df)} original test headlines.')\n\n"
            "# 3. Configure tokenizer for batch left-padding\n"
            "tokenizer.padding_side = 'left'\n"
            "if tokenizer.pad_token is None:\n"
            "    tokenizer.pad_token = tokenizer.eos_token\n\n"
            "def extract_tag(text, tag):\n"
            "    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.IGNORECASE | re.DOTALL)\n"
            "    return match.group(1).strip() if match else None\n\n"
            "results = []\n"
            "batch_size = 16\n\n"
            "for start in tqdm(range(0, len(test_df), batch_size), desc='Evaluating'):\n"
            "    batch_df = test_df.iloc[start : start + batch_size]\n"
            "    prompts = [\n"
            "        tokenizer.apply_chat_template(\n"
            "            [\n"
            "                {'role': 'user', 'content': SYSTEM_PROMPT + f'Headline: \"{row[\"text\"]}\"'}\n"
            "            ],\n"
            "            tokenize=False,\n"
            "            add_generation_prompt=True,\n"
            "        )\n"
            "        for _, row in batch_df.iterrows()\n"
            "    ]\n"
            "    \n"
            "    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(eval_model.device)\n"
            "    with torch.no_grad():\n"
            "        outputs = eval_model.generate(\n"
            "            **inputs,\n"
            "            max_new_tokens=150,\n"
            "            do_sample=False,\n"
            "            pad_token_id=tokenizer.pad_token_id,\n"
            "        )\n"
            "    \n"
            "    input_len = inputs['input_ids'].shape[1]\n"
            "    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)\n"
            "    \n"
            "    for i, raw_output in enumerate(decoded):\n"
            "        sentiment_tag = extract_tag(raw_output, 'sentiment')\n"
            "        reasoning_pred = extract_tag(raw_output, 'reasoning')\n"
            "        results.append({\n"
            "            'sentiment_true': batch_df.iloc[i]['sentiment'],\n"
            "            'sentiment_pred': sentiment_tag.lower() if sentiment_tag else 'none',\n"
            "            'reasoning_pred': reasoning_pred if reasoning_pred else 'none',\n"
            "            'tag_integrity': sentiment_tag is not None and reasoning_pred is not None\n"
            "        })\n\n"
            "results_df = pd.DataFrame(results)\n"
            "accuracy = accuracy_score(results_df['sentiment_true'], results_df['sentiment_pred'])\n"
            "integrity = results_df['tag_integrity'].mean()\n"
            "print(f'\\nAccuracy: {accuracy:.2%}')\n"
            "print(f'Tag Integrity: {integrity:.2%}')\n"
            'print("\\nClassification Report:")\n'
            "print(classification_report(results_df['sentiment_true'], results_df['sentiment_pred']))"
        ),
    ]
    nb.cells = cells
    with open("financial_sentiment_cot.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 'financial_sentiment_cot.ipynb'")


def create_medical_expert_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Track 2: Cardiology Medical Expert Fine-Tuning\n"
            "This notebook demonstrates domain-specific fine-tuning for Cardiology clinical QA using **Unsloth** and **QLoRA** on the `lmassaron/medical-cardiology-qa` dataset.\n\n"
            "### Why Unsloth?\n"
            "- Up to 5x faster training speeds.\n"
            "- Up to 60% memory savings, allowing larger models (or larger batch sizes) on a single 16GB VRAM GPU.\n"
            "- Keeps native model quality without performance degradation."
        ),
        nbf.v4.new_code_cell(
            "# Install extra dependencies if needed\n"
            "# %pip install -U unsloth trl peft bitsandbytes datasets"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import torch\n"
            "import numpy as np\n"
            "from datasets import load_dataset\n"
            "from unsloth import FastLanguageModel\n"
            "from trl import SFTTrainer, SFTConfig\n\n"
            "max_seq_length = 1024\n"
            "dtype = None  # None for auto detection (Float16/Bfloat16 based on hardware)\n"
            "load_in_4bit = True  # NF4 quantization for memory savings\n\n"
            "# Check device capabilities\n"
            "compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Model & Setup LoRA\n"
            "We load the Unsloth-optimized **Phi-4 Mini Instruct** model (3.8B parameters) and inject PEFT adapters."
        ),
        nbf.v4.new_code_cell(
            "model, tokenizer = FastLanguageModel.from_pretrained(\n"
            "    model_name='unsloth/Phi-4-mini-instruct',\n"
            "    max_seq_length=max_seq_length,\n"
            "    dtype=dtype,\n"
            "    load_in_4bit=load_in_4bit,\n"
            ")\n\n"
            "# Setup LoRA adapters targeting all attention and MLP projections\n"
            "model = FastLanguageModel.get_peft_model(\n"
            "    model,\n"
            "    r=16,\n"
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n"
            "    lora_alpha=32,\n"
            "    lora_dropout=0,  # Unsloth optimized to 0\n"
            "    bias='none',\n"
            "    use_gradient_checkpointing='unsloth',\n"
            "    random_state=3407,\n"
            ")"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Load and Prepare the Cardiology QA Dataset\n"
            "We load the cardiology QA dataset, split it into training/validation, and map the chat messages using the model's chat template."
        ),
        nbf.v4.new_code_cell(
            "DATASET_ID = 'lmassaron/medical-cardiology-qa'\n"
            "print(f'Loading {DATASET_ID}...')\n"
            "dataset = load_dataset(DATASET_ID, split='train')\n\n"
            "# Sample evaluation split (10%) reproducible seed\n"
            "n = len(dataset)\n"
            "rng = np.random.default_rng(42)\n"
            "all_idx = rng.permutation(n)\n"
            "cut = int(n * 0.9)\n"
            "train_idx, eval_idx = all_idx[:cut], all_idx[cut:]\n\n"
            "train_ds = dataset.select(train_idx)\n"
            "eval_ds = dataset.select(eval_idx)\n"
            "print(f'Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}')\n\n"
            "# Let's print a sample structure\n"
            "print('Sample message sequence:', train_ds[0]['messages'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Formatting Prompts\n"
            "We map the dataset using the conversation structure."
        ),
        nbf.v4.new_code_cell(
            "def formatting_prompts_func(examples):\n"
            "    convos = examples['messages']\n"
            "    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]\n"
            "    return { 'text' : texts }\n\n"
            "train_mapped = train_ds.map(formatting_prompts_func, batched=True)\n"
            "eval_mapped = eval_ds.map(formatting_prompts_func, batched=True)\n"
            "print('Prompt Preview:\\n', train_mapped[0]['text'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Run SFT Trainer using Unsloth\n"
            "We configure the SFTTrainer using Unsloth's optimized training engine."
        ),
        nbf.v4.new_code_cell(
            "trainer = SFTTrainer(\n"
            "    model=model,\n"
            "    tokenizer=tokenizer,\n"
            "    train_dataset=train_mapped,\n"
            "    eval_dataset=eval_mapped,\n"
            "    dataset_text_field='text',\n"
            "    max_seq_length=max_seq_length,\n"
            "    dataset_num_proc=2,\n"
            "    packing=False,\n"
            "    args=SFTConfig(\n"
            "        per_device_train_batch_size=2,\n"
            "        gradient_accumulation_steps=4,\n"
            "        warmup_steps=10,\n"
            "        max_steps=100,\n"
            "        learning_rate=2e-4,\n"
            "        fp16=not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,\n"
            "        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,\n"
            "        logging_steps=10,\n"
            "        eval_strategy='steps',\n"
            "        eval_steps=20,\n"
            "        save_steps=20,\n"
            "        output_dir='unsloth-medical-model',\n"
            "        report_to='none',\n"
            "        dataset_kwargs={\n"
            "            'add_special_tokens': False,\n"
            "        },\n"
            "    ),\n"
            ")\n\n"
            "# Disable KV cache during training to save memory\n"
            "model.config.use_cache = False\n\n"
            "trainer_stats = trainer.train()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Save the Adapter\nWe save the fine-tuned adapter weights to disk."
        ),
        nbf.v4.new_code_cell(
            "model.save_pretrained('unsloth-medical-adapter')\n"
            "tokenizer.save_pretrained('unsloth-medical-adapter')\n"
            "print('Adapter successfully saved!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Evaluation and Inference\n"
            "We put the model into Unsloth's optimized inference mode and run evaluations on our held-out test split. "
            "We calculate the average **Perplexity** on the cardiology reference answers (demonstrating how 'surprised' the model is by the clinical ground truth) "
            "and print a few example outputs for comparison."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from tqdm import tqdm\n\n"
            "FastLanguageModel.for_inference(model) # 2x faster inference\n\n"
            "eval_results = []\n"
            "# We evaluate on a sample of 20 evaluation conversations to run quickly\n"
            "num_eval_samples = min(20, len(eval_ds))\n"
            "print(f'Calculating perplexity and generating answers for {num_eval_samples} evaluation examples...')\n\n"
            "for i in tqdm(range(num_eval_samples)):\n"
            "    messages = eval_ds[i]['messages']\n"
            "    user_question = next(m['content'] for m in messages if m['role'] == 'user')\n"
            "    expected_answer = next(m['content'] for m in messages if m['role'] == 'assistant')\n"
            "    \n"
            "    # Format inputs for model generation\n"
            "    encoded = tokenizer.apply_chat_template(\n"
            "        messages[:-1],\n"
            "        tokenize=True,\n"
            "        add_generation_prompt=True,\n"
            "        return_tensors='pt'\n"
            "    ).to('cuda')\n"
            "    \n"
            "    # Generate answer\n"
            "    with torch.no_grad():\n"
            "        outputs = model.generate(\n"
            "            input_ids=encoded,\n"
            "            max_new_tokens=256,\n"
            "            use_cache=True,\n"
            "            pad_token_id=tokenizer.eos_token_id\n"
            "        )\n"
            "    generated = tokenizer.decode(outputs[0][encoded.shape[1]:], skip_special_tokens=True).strip()\n"
            "    \n"
            "    # Calculate perplexity over target answer tokens\n"
            "    ref_ids = tokenizer(expected_answer, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')\n"
            "    full_ids = torch.cat([encoded, ref_ids], dim=1)\n"
            "    labels = full_ids.clone()\n"
            "    labels[:, :encoded.shape[1]] = -100 # Mask out prompt tokens\n"
            "    \n"
            "    with torch.no_grad():\n"
            "        outputs_loss = model(full_ids, labels=labels)\n"
            "        loss = outputs_loss.loss\n"
            "        perplexity = torch.exp(loss).item()\n"
            "        \n"
            "    eval_results.append({\n"
            "        'question': user_question,\n"
            "        'expected': expected_answer,\n"
            "        'generated': generated,\n"
            "        'perplexity': perplexity\n"
            "    })\n\n"
            "eval_df = pd.DataFrame(eval_results)\n"
            "print(f'\\nAverage Evaluation Perplexity: {eval_df[\"perplexity\"].mean():.4f}')\n\n"
            "# Print a sample evaluation comparison\n"
            "print('\\n--- Sample Comparison ---')\n"
            "print('Question:', eval_df.iloc[0]['question'])\n"
            "print('\\nExpected Answer:', eval_df.iloc[0]['expected'])\n"
            "print('\\nGenerated Answer:', eval_df.iloc[0]['generated'])\n"
            "print(f'Perplexity: {eval_df.iloc[0][\"perplexity\"]:.4f}')"
        ),
    ]
    nb.cells = cells
    with open("medical_expert_cardiology.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 'medical_expert_cardiology.ipynb'")


def create_vision_latex_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Multimodal Fine-Tuning: Handwritten LaTeX OCR Expert\n\n"
            "This tutorial demonstrates how to fine-tune a state-of-the-art vision-language model, **Qwen2-VL 2B Instruct**, for document understanding and LaTeX formula transcription. "
            "We use **Unsloth** for memory-efficient QLoRA training and the **unsloth/LaTeX_OCR** dataset from the Hugging Face hub."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Install Dependencies & Setup Environment\n"
            "We install the required libraries (`unsloth`, `trl`, `peft`, `bitsandbytes`, `transformers`) to run our training."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import torch\n"
            "from datasets import load_dataset\n"
            "from unsloth import FastVisionModel\n"
            "from trl import SFTTrainer, SFTConfig\n"
            "from unsloth.trainer import UnslothVisionDataCollator\n\n"
            "compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n"
            "print('Environment initialized successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Load Model and Processor (FastVisionModel)\n"
            "We load `unsloth/Qwen2-VL-2B-Instruct` in 4-bit precision to fit within the 16GB VRAM constraint, and enable gradient checkpointing for lower VRAM footprints."
        ),
        nbf.v4.new_code_cell(
            "MODEL_ID = 'unsloth/Qwen2-VL-2B-Instruct'\n\n"
            "model, tokenizer = FastVisionModel.from_pretrained(\n"
            "    model_name=MODEL_ID,\n"
            "    load_in_4bit=True,\n"
            "    use_gradient_checkpointing='unsloth',\n"
            ")\n\n"
            "model = FastVisionModel.get_peft_model(\n"
            "    model,\n"
            "    finetune_vision_layers=True,\n"
            "    finetune_language_layers=True,\n"
            "    finetune_attention_modules=True,\n"
            "    finetune_mlp_modules=True,\n"
            "    r=16,\n"
            "    lora_alpha=16,\n"
            "    lora_dropout=0,\n"
            "    bias='none',\n"
            "    random_state=3407,\n"
            ")\n"
            "print('Qwen2-VL and PEFT adapters loaded successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Load and Format Dataset (LaTeX_OCR)\n"
            "We load `unsloth/LaTeX_OCR` directly from Hugging Face and map the examples into a conversational format suitable for vision-language models."
        ),
        nbf.v4.new_code_cell(
            "dataset = load_dataset('unsloth/LaTeX_OCR', split='train')\n\n"
            "# Select a subset to train and validate quickly\n"
            "shuffled_dataset = dataset.shuffle(seed=42)\n"
            "train_ds = shuffled_dataset.select(range(500))\n"
            "eval_ds = shuffled_dataset.select(range(500, 550))\n\n"
            "def convert_to_conversation(sample):\n"
            "    conversation = [\n"
            "        {\n"
            "            'role': 'user',\n"
            "            'content': [\n"
            "                {'type': 'text', 'text': 'Write the LaTeX representation for this image.'},\n"
            "                {'type': 'image'}\n"
            "            ]\n"
            "        },\n"
            "        {\n"
            "            'role': 'assistant',\n"
            "            'content': [\n"
            "                {'type': 'text', 'text': sample['text']}\n"
            "            ]\n"
            "        },\n"
            "    ]\n"
            "    return {\n"
            "        'messages': conversation,\n"
            "        'images': [sample['image']]\n"
            "    }\n\n"
            "train_mapped = train_ds.map(convert_to_conversation, remove_columns=train_ds.column_names)\n"
            "eval_mapped = eval_ds.map(convert_to_conversation, remove_columns=eval_ds.column_names)\n"
            "print('Dataset sample mapped successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Run SFT Trainer\n"
            "We configure the Hugging Face `trl` SFTTrainer with Unsloth's optimized vision data collator. "
            "We set `remove_unused_columns = False` to prevent losing the image data during training."
        ),
        nbf.v4.new_code_cell(
            "training_args = SFTConfig(\n"
            "    output_dir='qwen2-vl-latex',\n"
            "    dataset_text_field='text',\n"
            "    max_seq_length=512,\n"
            "    remove_unused_columns=False,\n"
            "    per_device_train_batch_size=2,\n"
            "    gradient_accumulation_steps=4,\n"
            "    learning_rate=2e-4,\n"
            "    warmup_steps=5,\n"
            "    max_steps=30,\n"
            "    bf16=(compute_dtype == torch.bfloat16),\n"
            "    logging_steps=5,\n"
            "    eval_strategy='steps',\n"
            "    eval_steps=10,\n"
            "    save_steps=10,\n"
            "    report_to='none',\n"
            ")\n\n"
            "trainer = SFTTrainer(\n"
            "    model=model,\n"
            "    tokenizer=tokenizer,\n"
            "    data_collator=UnslothVisionDataCollator(model, tokenizer),\n"
            "    train_dataset=train_mapped,\n"
            "    eval_dataset=eval_mapped,\n"
            "    args=training_args,\n"
            ")\n\n"
            "# Disable KV cache during training to save memory\n"
            "model.config.use_cache = False\n\n"
            "trainer.train()\n\n"
            "# Save the fine-tuned adapter\n"
            "model.save_pretrained('qwen2-vl-latex-adapter')\n"
            "tokenizer.save_pretrained('qwen2-vl-latex-adapter')\n"
            "print('Vision adapter successfully saved!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Evaluation and Inference\n"
            "We switch the model to inference mode and generate a LaTeX translation for a validation image."
        ),
        nbf.v4.new_code_cell(
            "FastVisionModel.for_inference(model)\n\n"
            "sample = eval_ds[0]\n"
            "image = sample['image']\n"
            "expected_latex = sample['text']\n\n"
            "messages = [\n"
            "    {\n"
            "        'role': 'user',\n"
            "        'content': [\n"
            "            {'type': 'text', 'text': 'Write the LaTeX representation for this image.'},\n"
            "            {'type': 'image', 'image': image}\n"
            "        ]\n"
            "    }\n"
            "]\n\n"
            "input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)\n"
            "inputs = tokenizer(\n"
            "    image,\n"
            "    input_text,\n"
            "    add_special_tokens=False,\n"
            "    return_tensors='pt'\n"
            ").to('cuda')\n\n"
            "with torch.no_grad():\n"
            "    outputs = model.generate(**inputs, max_new_tokens=128)\n\n"
            "generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()\n\n"
            "print('--- Expected LaTeX ---')\n"
            "print(expected_latex)\n"
            "print('\\n--- Generated LaTeX ---')\n"
            "print(generated_text)"
        ),
    ]
    nb.cells = cells
    with open("vision_finetuning_latex.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 'vision_finetuning_latex.ipynb'")


def create_alignment_dpo_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Track 4: Aligned Response Refinement with Direct Preference Optimization (DPO)\n\n"
            "This notebook demonstrates how to align a model with preferred human feedback using "
            "Direct Preference Optimization (DPO) and `DPOTrainer` from Hugging Face `trl`. "
            "We use **Unsloth** for memory-efficient and fast LoRA training of the **Qwen2.5-3B-Instruct** model."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Setup Environment and Imports\n"
            "We load our dependencies and identify the compute device capability to determine if bfloat16 is supported."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import torch\n"
            "import warnings\n"
            "from datasets import load_dataset\n"
            "from transformers import AutoTokenizer\n"
            "from unsloth import FastLanguageModel, PatchDPOTrainer\n"
            "from trl import DPOTrainer, DPOConfig\n\n"
            "warnings.filterwarnings('ignore')\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n\n"
            "compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n"
            "print(f'Using device: cuda | Dtype: {compute_dtype}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Load Model & Enable PEFT\n"
            "We load `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` using Unsloth's optimized 4-bit precision to fit within small VRAM budgets, "
            "and attach LoRA adapters to all projection layers."
        ),
        nbf.v4.new_code_cell(
            "MODEL_ID = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'\n\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(\n"
            "    model_name=MODEL_ID,\n"
            "    max_seq_length=1024,\n"
            "    dtype=None,\n"
            "    load_in_4bit=True,\n"
            ")\n\n"
            "model = FastLanguageModel.get_peft_model(\n"
            "    model,\n"
            "    r=8,\n"
            "    lora_alpha=16,\n"
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n"
            "    lora_dropout=0,\n"
            "    bias='none',\n"
            "    use_gradient_checkpointing='unsloth',\n"
            "    random_state=3407,\n"
            ")\n"
            "print('Model and PEFT adapters loaded successfully.')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Load and Format Dataset (Orca DPO Pairs)\n"
            "We load the `Intel/orca_dpo_pairs` dataset from Hugging Face, shuffle it, and select a small subset of 250 training examples and 50 validation examples. "
            "We map the inputs to a standard DPO format where the user prompt is compiled using the model's native chat template."
        ),
        nbf.v4.new_code_cell(
            "dataset = load_dataset('Intel/orca_dpo_pairs', split='train')\n"
            "shuffled = dataset.shuffle(seed=42)\n"
            "train_ds = shuffled.select(range(250))\n"
            "eval_ds = shuffled.select(range(250, 300))\n\n"
            "def format_dpo_example(example):\n"
            "    messages = [{'role': 'user', 'content': example['question']}]\n"
            "    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
            "    return {\n"
            "        'prompt': prompt_str,\n"
            "        'chosen': example['chosen'],\n"
            "        'rejected': example['rejected']\n"
            "    }\n\n"
            "train_mapped = train_ds.map(format_dpo_example, remove_columns=train_ds.column_names)\n"
            "eval_mapped = eval_ds.map(format_dpo_example, remove_columns=eval_ds.column_names)\n"
            "print('Prompt Preview:\\n', train_mapped[0]['prompt'])\n"
            "print('Chosen Preview:\\n', train_mapped[0]['chosen'])\n"
            "print('Rejected Preview:\\n', train_mapped[0]['rejected'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Run Aligned DPO Fine-Tuning\n"
            "We run Unsloth's patched version of the TRL `DPOTrainer` (`PatchDPOTrainer()`). We run training for 60 steps with a learning rate of 5e-6, "
            "using cosine annealing decay."
        ),
        nbf.v4.new_code_cell(
            "PatchDPOTrainer()\n\n"
            "training_args = DPOConfig(\n"
            "    output_dir='qwen2.5-3b-dpo-output',\n"
            "    beta=0.1,\n"
            "    max_length=1024,\n"
            "    max_prompt_length=512,\n"
            "    per_device_train_batch_size=2,\n"
            "    gradient_accumulation_steps=4,\n"
            "    learning_rate=5e-6,\n"
            "    max_steps=60,\n"
            "    lr_scheduler_type='cosine',\n"
            "    warmup_ratio=0.1,\n"
            "    bf16=(compute_dtype == torch.bfloat16),\n"
            "    fp16=(compute_dtype == torch.float16),\n"
            "    logging_steps=10,\n"
            "    eval_strategy='steps',\n"
            "    eval_steps=20,\n"
            "    save_steps=20,\n"
            "    report_to='none',\n"
            ")\n\n"
            "trainer = DPOTrainer(\n"
            "    model=model,\n"
            "    ref_model=None,\n"
            "    args=training_args,\n"
            "    train_dataset=train_mapped,\n"
            "    eval_dataset=eval_mapped,\n"
            "    processing_class=tokenizer,\n"
            ")\n\n"
            "model.config.use_cache = False\n"
            "trainer.train()\n\n"
            "model.save_pretrained('qwen2.5-3b-dpo-adapter')\n"
            "tokenizer.save_pretrained('qwen2.5-3b-dpo-adapter')\n"
            "print('DPO adapter successfully saved!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Evaluation and Inference comparison\n"
            "We put the model in inference mode and query it with a test prompt to check if it adheres to Aligned Response preferences."
        ),
        nbf.v4.new_code_cell(
            "FastLanguageModel.for_inference(model)\n\n"
            'test_question = "Explain why the sky is blue in one concise sentence."\n'
            "messages = [{'role': 'user', 'content': test_question}]\n"
            "inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')\n\n"
            "with torch.no_grad():\n"
            "    outputs = model.generate(\n"
            "        input_ids=inputs,\n"
            "        max_new_tokens=100,\n"
            "        use_cache=True,\n"
            "        pad_token_id=tokenizer.eos_token_id\n"
            "    )\n\n"
            "response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()\n"
            "print('Question:', test_question)\n"
            "print('\\nAligned Response:', response)"
        ),
    ]
    nb.cells = cells
    with open("alignment_dpo.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 'alignment_dpo.ipynb'")


def create_alignment_grpo_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Track 5: Group Relative Policy Optimization (GRPO) for Mathematical Reasoning\n\n"
            "This notebook demonstrates fine-tuning **Qwen2.5-3B-Instruct** on mathematical reasoning problems "
            "from the `openai/gsm8k` dataset using **Group Relative Policy Optimization (GRPO)**. "
            "We configure reinforcement learning rewards based on format correctness and solution accuracy, "
            "and leverage **vLLM** inside `GRPOTrainer` to accelerate policy rollout generation."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Setup Environment and Imports\n"
            "We load libraries and establish system prompt instructions enforcing a strict XML reasoning and answer format."
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "import sys\n"
            "import torch\n"
            "import warnings\n"
            "import re\n"
            "# Patch all TRL import helper functions to return boolean instead of tuple, bypassing import bugs\n"
            "import trl.import_utils as utils\n"
            "for name, attr in list(vars(utils).items()):\n"
            "    if name.startswith('is_') and name.endswith('_available') and callable(attr):\n"
            "        def make_wrapper(func):\n"
            "            return lambda *a, **k: func(*a, **k)[0] if isinstance(func(*a, **k), tuple) else func(*a, **k)\n"
            "        setattr(utils, name, make_wrapper(attr))\n"
            "utils._vllm_available = False\n\n"
            "from datasets import load_dataset\n"
            "from transformers import AutoTokenizer\n"
            "from peft import LoraConfig\n"
            "from trl import GRPOConfig, GRPOTrainer\n\n"
            "warnings.filterwarnings('ignore')\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n\n"
            "compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n"
            "print(f'Using device: cuda | Dtype: {compute_dtype}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Dataset Preparation (GSM8K)\n"
            "We load the GSM8K dataset. We format each question with a system instruction that instructs the model to put its reasoning inside `<reasoning>` tags and the final numeric answer inside `<answer>` tags."
        ),
        nbf.v4.new_code_cell(
            "SYSTEM_PROMPT = (\n"
            '    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\\n"\n'
            '    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\\n"\n'
            '    "The reasoning process and answer are enclosed within tags. The answer must be a single integer.\\n"\n'
            '    "Example:\\n"\n'
            '    "<reasoning>\\n"\n'
            '    "We know that 2 + 2 = 4.\\n"\n'
            '    "</reasoning>\\n"\n'
            '    "<answer>4</answer>"\n'
            ")\n\n"
            "def extract_hash_answer(text: str) -> str | None:\n"
            "    if '####' not in text:\n"
            "        return None\n"
            "    return text.split('####')[1].strip()\n\n"
            "dataset = load_dataset('openai/gsm8k', 'main', split='train')\n"
            "# Select small subset of 300 prompts for fast notebook execution\n"
            "dataset = dataset.shuffle(seed=42).select(range(300))\n\n"
            "def format_gsm8k(example):\n"
            "    return {\n"
            "        'prompt': [\n"
            "            {'role': 'system', 'content': SYSTEM_PROMPT},\n"
            "            {'role': 'user', 'content': example['question']}\n"
            "        ],\n"
            "        'answer': extract_hash_answer(example['answer'])\n"
            "    }\n\n"
            "gsm8k_train = dataset.map(format_gsm8k)\n"
            "print('Sample Prompt Structure Preview:\\n', gsm8k_train[0]['prompt'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Define RL Rewards\n"
            "We define two reward functions:\n"
            "1. **Format Reward**: Returns `1.0` if the output strictly matches the `<reasoning>...</reasoning>\\n<answer>...</answer>` tags.\n"
            "2. **Correctness Reward**: Returns `2.0` if the extracted answer matches the ground truth, and `0.0` otherwise."
        ),
        nbf.v4.new_code_cell(
            "def extract_last_xml_answer(text, start_tag='<answer>', end_tag='</answer>'):\n"
            "    pattern = re.escape(start_tag) + r'(.*?)' + re.escape(end_tag)\n"
            "    matches = re.findall(pattern, text, re.DOTALL)\n"
            "    if matches:\n"
            "        answer = matches[-1]\n"
            "        answer = re.sub(r'[%$]', '', answer).strip()\n"
            "        return answer\n"
            "    return ''\n\n"
            "def format_reward_func(completions, **kwargs):\n"
            "    pattern = r'^<reasoning>[\\s\\S]*?<\\/reasoning>\\s*<answer>[\\s\\S]*?<\\/answer>$'\n"
            "    responses = [completion[0]['content'] for completion in completions]\n"
            "    rewards = [1.0 if re.match(pattern, response) else 0.0 for response in responses]\n"
            "    return rewards\n\n"
            "def correctness_reward_func(completions, answer, **kwargs):\n"
            "    responses = [completion[0]['content'] for completion in completions]\n"
            "    extracted = [extract_last_xml_answer(response) for response in responses]\n"
            "    rewards = [2.0 if ext == ans else 0.0 for ext, ans in zip(extracted, answer)]\n"
            "    return rewards"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Run GRPOTrainer using vLLM\n"
            "We load the tokenizer and define the LoRA parameters. "
            "We use `vLLM` inside the trainer with a configured device and VRAM footprint to scale policy rollouts rapidly."
        ),
        nbf.v4.new_code_cell(
            "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'\n"
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n\n"
            "peft_config = LoraConfig(\n"
            "    lora_alpha=64,\n"
            "    lora_dropout=0.0,\n"
            "    r=64,\n"
            "    bias='none',\n"
            "    task_type='CAUSAL_LM',\n"
            "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n"
            ")\n\n"
            "training_args = GRPOConfig(\n"
            "    use_vllm=False,\n"
            "    learning_rate=1e-5,\n"
            "    adam_beta1=0.9,\n"
            "    adam_beta2=0.99,\n"
            "    weight_decay=0.1,\n"
            "    warmup_ratio=0.1,\n"
            "    beta=0.005,\n"
            "    lr_scheduler_type='cosine',\n"
            "    optim='adamw_8bit',\n"
            "    bf16=(compute_dtype == torch.bfloat16),\n"
            "    fp16=(compute_dtype == torch.float16),\n"
            "    gradient_checkpointing=True,\n"
            "    gradient_checkpointing_kwargs={'use_reentrant': False},\n"
            "    gradient_accumulation_steps=4,\n"
            "    per_device_train_batch_size=4,\n"
            "    num_generations=4,\n"
            "    temperature=0.5,\n"
            "    max_prompt_length=256,\n"
            "    max_completion_length=256,\n"
            "    max_steps=50,\n"
            "    logging_steps=10,\n"
            "    save_steps=50,\n"
            "    max_grad_norm=0.1,\n"
            "    report_to='none',\n"
            "    output_dir='qwen2.5-3b-grpo-output',\n"
            ")\n\n"
            "from transformers import AutoModelForCausalLM\n"
            "model = AutoModelForCausalLM.from_pretrained(\n"
            "    MODEL_NAME,\n"
            "    torch_dtype=compute_dtype,\n"
            "    device_map='auto'\n"
            ")\n"
            "if not hasattr(model, 'warnings_issued'):\n"
            "    model.warnings_issued = {}\n\n"
            "trainer = GRPOTrainer(\n"
            "    model=model,\n"
            "    processing_class=tokenizer,\n"
            "    reward_funcs=[correctness_reward_func, format_reward_func],\n"
            "    args=training_args,\n"
            "    train_dataset=gsm8k_train,\n"
            "    peft_config=peft_config,\n"
            ")\n\n"
            "trainer.train()\n\n"
            "merged_model = trainer.model.merge_and_unload()\n"
            "tokenizer.save_pretrained('qwen2.5-3b-grpo-adapter')\n"
            "merged_model.save_pretrained('qwen2.5-3b-grpo-adapter')\n"
            "print('GRPO training completed and model saved!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Inference Verification\n"
            "We query our trained model to verify that it generates reasoning and mathematical solutions structured under correct tags."
        ),
        nbf.v4.new_code_cell(
            'test_question = "Natalia sold clips to 48 of her friends in April, and then half as many in May. How many clips did Natalia sell in total?"\n'
            "messages = [\n"
            "    {'role': 'system', 'content': SYSTEM_PROMPT},\n"
            "    {'role': 'user', 'content': test_question}\n"
            "]\n"
            "inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')\n"
            "input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs['input_ids']\n"
            "input_ids = input_ids.to('cuda')\n\n"
            "merged_model.eval()\n"
            "with torch.no_grad():\n"
            "    outputs = merged_model.generate(\n"
            "        input_ids=input_ids,\n"
            "        max_new_tokens=256,\n"
            "        pad_token_id=tokenizer.eos_token_id\n"
            "    )\n\n"
            "response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()\n"
            "print('Question:', test_question)\n"
            "print('\\nModel response:\\n', response)"
        ),
    ]
    nb.cells = cells
    with open("alignment_grpo.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 'alignment_grpo.ipynb'")


if __name__ == "__main__":
    create_financial_sentiment_notebook()
    create_medical_expert_notebook()
    create_vision_latex_notebook()
    create_alignment_dpo_notebook()
    create_alignment_grpo_notebook()
