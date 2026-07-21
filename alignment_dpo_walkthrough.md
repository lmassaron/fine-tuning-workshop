# Walkthrough: Direct Preference Optimization (DPO) Alignment

This walkthrough details the technical architecture, implementation workflow, and evaluation results of [alignment_dpo.ipynb](file:///home/lmassaron/code/sft-examples/alignment_dpo.ipynb).

---

## 1. Executive Summary & Goal

Standard Supervised Fine-Tuning (SFT) teaches language models to follow instructions, but does not align output style, conciseness, or formatting with human preferences. This track performs **Direct Preference Optimization (DPO)** on **`unsloth/Qwen2.5-3B-bnb-4bit`** using preference pairs from `argilla/distilabel-intel-orca-dpo-pairs`.

### Key Objectives
1. **Style & Structural Alignment**: Align the model to prefer well-structured, verbose, markdown-formatted responses over dispreferred, flat completions.
2. **Implicit Reference Model**: Optimize policy log-ratios without maintaining a separate reward model or PPO actor-critic infrastructure.
3. **ChatML Formatting**: Format preference pairs into ChatML system/user/assistant conversation framing.

---

## 2. Technical Stack & Architecture

- **Base Model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Base Qwen 2.5 3B parameters with 4-bit NF4 quantization)
- **Framework**: Unsloth + Hugging Face `trl` (`DPOTrainer`, `DPOConfig`) + `peft` (LoRA)
- **Preference Dataset**: `argilla/distilabel-intel-orca-dpo-pairs`
- **Chat Template**: `chatml` via `unsloth.get_chat_template`

---

## 3. Practical Implementation Workflow

### Step 1: Unsloth DPO Patching & Model Loading
```python
from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()  # Patches TRL DPOTrainer with Unsloth memory kernels

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### Step 2: ChatML Template Setup & LoRA Injection
```python
from unsloth import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

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
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
```

### Step 3: Dataset Formatting
Each sample in `argilla/distilabel-intel-orca-dpo-pairs` contains `input`, `chosen`, and `rejected`. We convert them into ChatML formatted prompt strings:
```python
def format_dpo(sample):
    prompt_msgs = [{"role": "user", "content": sample["input"]}]
    chosen_msgs = [{"role": "assistant", "content": sample["chosen"]}]
    rejected_msgs = [{"role": "assistant", "content": sample["rejected"]}]

    return {
        "prompt": tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        ),
        "chosen": tokenizer.apply_chat_template(chosen_msgs, tokenize=False),
        "rejected": tokenizer.apply_chat_template(
            rejected_msgs, tokenize=False
        ),
    }
```

---

## 4. DPO Loss Formulation & Hyperparameters

DPO optimizes policy parameters $\theta$ directly using preference pairs $(x, y_w, y_l)$:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### Hyperparameter Table
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **KL Penalty ($\beta$)** | `0.1` | Controls divergence from implicit reference model |
| **Learning Rate** | `5e-6` | Low learning rate prevents policy collapse during DPO |
| **Max Steps** | `120` | Full optimization over Orca DPO dataset |
| **Batch Size** | `2` per device $\times$ `4` grad accum | Effective Batch Size of 8 |
| **Max Prompt Length** | `1024` | Captures long input instructions |
| **Max Sequence Length** | `2048` | Accommodates multi-paragraph chosen completions |

---

## 5. Empirical Behavioral Shifting & Evaluation

To evaluate DPO alignment, we run inference on diverse domain prompts comparing the **Un-adapted Base Model** against the **DPO Fine-tuned Model**:

### Prompt 1 (Philosophy): *"What is the difference between ethics and morality?"*
- **Base Model (Pre-DPO)**: Outputted brief, direct definitions without structured formatting.
- **DPO Model (Post-DPO)**: Produced a well-structured, multi-paragraph comparison contrasting philosophical definitions, historical origins, and practical applications, incorporating bullet points and clear distinctions.

### Prompt 2 (History): *"Explain the key factors that led to the fall of the Western Roman Empire."*
- **Base Model (Pre-DPO)**: Listed a short summary of military pressures.
- **DPO Model (Post-DPO)**: Provided a comprehensive, multi-faceted analysis categorized into Political Instability, Economic Troubles, Military Overextension, and External Invasions.

---

## 6. Artifact Outputs
- Saved PEFT LoRA adapter: `qwen2.5-3b-dpo-adapter/`
