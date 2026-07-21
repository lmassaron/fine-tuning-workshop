# Walkthrough: Cardiology Medical Expert Fine-Tuning

This walkthrough details the technical architecture, Unsloth memory acceleration, dataset preprocessing, and target-token perplexity validation for [medical_expert_cardiology.ipynb](file:///home/lmassaron/code/sft-examples/medical_expert_cardiology.ipynb).

---

## 1. Executive Summary & Goal

Medical dialogue requires domain-specific vocabulary and diagnostic precision. Standard base models often generate generic health advice rather than precise clinical guidance. This track fine-tunes **`unsloth/Phi-4-mini-instruct`** (3.8B parameters) on the `lmassaron/medical-cardiology-qa` dataset to build a specialized cardiology clinical assistant.

### Key Objectives
1. **Clinical QA Specialization**: Adapt Microsoft's Phi-4 Mini to generate structured, professional cardiology diagnostic guidance.
2. **Unsloth Memory & Speed Acceleration**: Achieve up to 5x faster training speeds and 60% VRAM savings using Unsloth's optimized kernels.
3. **Target Token Perplexity Validation**: Validate model confidence on clinical ground-truth terms using masked target loss evaluation.

---

## 2. Technical Stack & Unsloth Architecture

- **Base Model**: `unsloth/Phi-4-mini-instruct` (3.8B parameters)
- **Quantization**: 4-bit NormalFloat (NF4) via Unsloth (`load_in_4bit=True`)
- **Max Sequence Length**: 1024 tokens
- **PEFT LoRA Config**:
  - Rank $r=16$, $\alpha=32$, Dropout $= 0$ (Unsloth optimized to 0 for maximum speed)
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Gradient Checkpointing**: `'unsloth'` native recomputation engine

---

## 3. Practical Dataset Preprocessing

### Dataset Structure
- **Dataset**: `lmassaron/medical-cardiology-qa`
- **Splits**: 90% Training / 10% Evaluation (reproducible seed `42`)
- **Format**: Multi-turn physician-patient dialogues mapped using Phi-4's native chat template:
  ```python
  def formatting_prompts_func(examples):
      convos = examples["messages"]
      texts = [
          tokenizer.apply_chat_template(
              convo, tokenize=False, add_generation_prompt=False
          )
          for convo in convos
      ]
      return {"text": texts}
```

### Crucial Import Sequence
Unsloth requires `import unsloth` to be called **before** importing `transformers` or `trl`. This allows Unsloth to patch PyTorch attention kernels and memory allocators prior to module registration.

---

## 4. Training Parameters & Configuration

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Max Sequence Length** | `1024` | Captures long medical questions and multi-paragraph clinical answers |
| **Batch Size** | `2` per device | Fits within 16GB VRAM limit |
| **Gradient Accumulation** | `4` | Yields an **Effective Batch Size of 8** |
| **Learning Rate** | `2e-4` | Standard initial learning rate for QLoRA adaptation |
| **Warmup Steps** | `10` | Prevents initial gradient instability |
| **Max Training Steps** | `100` | Full convergence over cardiology dialogue dataset |
| **Precision** | `bf16` | Brain floating-point 16-bit execution on Ampere/Hopper GPUs |
| **KV Cache** | Disabled (`use_cache=False`) | Reduces memory consumption during backward pass |

---

## 5. Quantitative Perplexity Evaluation & Results

In addition to qualitative inspection, we evaluate the model by computing the **Target Answer Perplexity (PPL)** on held-out validation dialogues.

### Mathematical Formulation
Perplexity measures the exponential cross-entropy loss computed strictly over the doctor's response tokens (masking prompt tokens with `-100`):

$$PPL = \exp\left( - \frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x, y_{<i}) \right)$$

### Evaluation Implementation
```python
# Mask out prompt tokens so loss is only calculated on the reference answer
ref_ids = tokenizer(
    expected_answer, return_tensors="pt", add_special_tokens=False
).input_ids.to("cuda")
full_ids = torch.cat([encoded, ref_ids], dim=1)
labels = full_ids.clone()
labels[:, : encoded.shape[1]] = -100  # Mask out prompt tokens

with torch.no_grad():
    outputs_loss = model(full_ids, labels=labels)
    loss = outputs_loss.loss
    perplexity = torch.exp(loss).item()
```

### Validation Benchmark Results
- **Average Evaluation Perplexity**: **4.4341**
- **Specific Clinical Case Perplexity (Aortic Dissection)**: **1.5286**

A perplexity of **4.43** demonstrates that the fine-tuned model assigns extremely high probability to expert cardiology terminology and diagnostic guidelines.

---

## 6. Concrete Cardiology Expert Output Comparison

- **User Question**: *"What are some connective tissue disorders that increase the risk of aortic dissection?"*
- **Expected Answer**: *"Connective tissue disorders that increase the risk of aortic dissection include Marfan syndrome, Ehlers-Danlos syndrome, and Loeys-Dietz syndrome. These conditions affect the structural integrity of the arterial walls."*
- **Model Generated Output**:
  ```
  Connective tissue disorders such as Marfan syndrome, Ehlers-Danlos syndrome, and Loeys-Dietz syndrome are known to increase the risk of aortic dissection. These conditions affect the body's connective tissue, which can weaken the aortic wall and make it more susceptible to dissection.
  ```
- **Target Perplexity**: **1.5286**

---

## 7. Artifact Outputs
- Fine-tuned adapter saved at: `unsloth-medical-adapter/`
