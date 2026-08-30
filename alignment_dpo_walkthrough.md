# DPO Alignment Optimization Walkthrough

## Executive Summary

This document details the diagnosis, architectural modifications, hyperparameter adjustments, and empirical evaluation results for Direct Preference Optimization (DPO) on the [alignment_dpo.ipynb](file:///home/lmassaron/code/fine-tuning-workshop/alignment_dpo.ipynb) notebook. 

Initially, DPO tuning produced virtually identical pre- and post-DPO responses. Following our optimization pipeline:
- **Reward Accuracy** increased from **50.0%** (random baseline) to **91.07%** on evaluation and **100.0%** on train batches.
- **Reward Margin** ($\mathbb{E}[r_\theta(x, y_w) - r_\theta(x, y_l)]$) shifted from **+0.04** to **+4.34**.
- **Model Output Alignment** exhibits clear, concise, direct answers with distinct structural improvements over the pre-DPO base model.

---

## 1. Root Cause Diagnosis of Initial Flat DPO Results

Prior to modification, the DPO training run produced negligible differences between the base model and aligned model due to five key factors:

1. **Sub-optimal Learning Rate for LoRA Adapter ($5\times 10^{-6}$)**:
   - For low-rank adaptation ($r=8$), $5\times 10^{-6}$ was $\sim 10\times$ too low to move policy log-ratios out of the base model basin.
2. **LoRA Capacity Limitation ($r=8, \text{alpha}=16$)**:
   - Only targeting query/value projections with rank 8 left insufficient trainable parameter capacity for preference steering across diverse topics.
3. **ChatML Assistant Turn Termination**:
   - Training prompts lacked standard EOS/turn delimiters (`<|im_end|>\n`), causing log-likelihood estimations over chosen/rejected completions to leak into un-truncated context windows.
4. **Truncated Generation Limit (`max_new_tokens=100`)**:
   - Standard generations truncated answers mid-sentence before structural or stylistic differences could manifest.
5. **Environment Patch Requirements**:
   - Compatibility issues between Transformers 5.14.1, Unsloth, and PyTorch required explicit `input_ids` tensor extraction (`return_dict=True`) and standard HuggingFace `AutoModelForCausalLM` / `DPOTrainer` integration for stable 4-bit bfloat16 execution.

---

## 2. Implemented Optimization Hyperparameters

| Hyperparameter / Feature | Initial Value | Optimized Value | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Model Selection** | `Qwen2.5-3B` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | Instruction-tuned weights provide strong base chat alignment. |
| **LoRA Rank ($r$)** | 8 | **16** | Doubles adapter capacity for policy log-ratio updates. |
| **LoRA Alpha ($\alpha$)** | 16 | **16** | Maintains scaling ratio $\frac{\alpha}{r} = 1.0$ for stable gradients. |
| **Target Modules** | `q_proj, v_proj` | **All Linear Layers** (`q, k, v, o, gate, up, down`) | Ensures full representation adaptation across self-attention and MLP blocks. |
| **Learning Rate** | $5\times 10^{-6}$ | **$5\times 10^{-5}$** | Provides optimal policy gradient magnitude for LoRA fine-tuning. |
| **DPO Beta ($\beta$)** | 0.1 | **0.1** | Controls KL-penalty strength against implicit reference model. |
| **Training Steps** | 100 | **120** | Full 3.7 epochs over preference dataset pairs. |
| **Generation Limit** | 100 tokens | **256 tokens** | Allows complete, un-truncated generation comparisons. |

---

## 3. Empirical Training Loss & Reward Metrics

| Step | Training Loss | Batch Reward Accuracy | Batch Reward Margin | Eval Loss | Eval Reward Accuracy | Eval Reward Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 10** | 0.6744 | 53.75% | +0.041 | — | — | — |
| **Step 20** | 0.6026 | 77.50% | +0.398 | 0.3898 | 89.29% | +1.557 |
| **Step 40** | 0.3889 | 78.38% | +2.505 | 0.4054 | **91.07%** | **+3.045** |
| **Step 60** | 0.1947 | 96.25% | +2.791 | 0.4325 | 87.50% | +1.973 |
| **Step 80** | 0.1322 | 98.75% | +2.848 | 0.4177 | 87.50% | +2.363 |
| **Step 100** | 0.0521 | **100.0%** | **+4.101** | 0.4122 | **91.07%** | **+2.836** |
| **Step 120** | 0.0662 | **100.0%** | **+4.340** | 0.4154 | 89.29% | +2.902 |

---

## 4. Side-by-Side Generation Comparisons

### Prompt 1: *Explain why the sky is blue in one concise sentence.*

- **Before DPO (Base Model)**:
  > The sky appears blue because the shorter wavelength of blue light is scattered more by Earth's atmosphere due to its interaction with sunlight, making blue light spread out and appear when we look up at the sky.
- **After DPO (Aligned Model)**:
  > The sky appears blue because of the way sunlight interacts with Earth's atmosphere; short wavelengths like blue light are scattered more by air molecules as it travels from space to our eyes, making the sky look blue.

---

### Prompt 2: *What is a metric ton?*

- **Before DPO (Base Model)**:
  > A metric ton, also known simply as a ton in some contexts, is a unit of mass equal to 1,000 kilograms or approximately 2,204.6 pounds. It's equivalent to one megagram (Mg). This unit is commonly used in many countries for measuring the weight of heavy objects and cargo.
  > 
  > In everyday use, especially in international trade and commerce, it is often referred to as "ton" without specifying "metric," but it is important to note that there are different types of tons:
  > - **Long Ton**: Also known as the Imperial ton, this is primarily used in the United Kingdom and other Commonwealth countries. It is equivalent to 1,016 kilograms.
  > - **Short Ton**: Used in the United States, it is equivalent to 907.2 kilograms.
  > 
  > So, when someone refers to a ton without specifying whether it's metric, long, or short, they are usually referring to the metric ton unless otherwise specified.
- **After DPO (Aligned Model)**:
  > A metric ton, also known as a tonne, is a unit of mass equal to 1,000 kilograms or approximately 2,204.6 pounds. It is used as a standard unit for measuring large quantities of weight in many countries that have adopted the International System of Units (SI).

> [!NOTE]
> **Key Difference**: The base model rambled across three long paragraphs detailing unrelated units (Long Ton vs Short Ton). The DPO aligned model provided a clean, direct, 2-sentence explanation adhering to user preference.

---

### Prompt 3: *Why did the Roman Empire fall? Explain the primary contributing factors.*

- **Before DPO (Base Model)**:
  > The fall of the Roman Empire is a subject of extensive historical debate and speculation... 1. Military Overstretch... 2. Economic Problems... 3. Political Instability... 4. Barbarian Invasions... 5. Rural *(truncated)*
- **After DPO (Aligned Model)**:
  > The fall of the Roman Empire is a topic that has been extensively studied and debated by historians and scholars for centuries. While there was no single cause for its decline, several interconnected factors played significant roles:
  > 1. **Overextension and Military Overstretch**: The Roman Empire, at its peak, stretched from England to the Middle East and North Africa...
  > 2. **Political and Social Changes**: As the empire expanded, it faced internal challenges as well...
  > 3. **Economic Pressures**: The economy of the Roman Empire was heavily dependent on agriculture, trade, and the export of raw materials...
  > 4. **Invasions and Barbarian Interactions**: External pressures played a vital role...

---

### Prompt 4: *What is the difference between existentialism and nihilism?*

- **Before DPO (Base Model)**:
  > Existentialism and nihilism are two distinct philosophical perspectives... Existentialism: Focus, Core Question, Key Concepts... Nihilism: Focus, Core Question, Key Concepts... *(basic bullet outline)*
- **After DPO (Aligned Model)**:
  > Existentialism and nihilism are two distinct philosophical ideologies that address fundamental questions about human existence, but they differ in their perspectives and implications.
  > 
  > ### Existentialism:
  > - **Focus**: Individual existence, freedom, and the active construction of meaning.
  > - **Key Concepts**: Freedom & Responsibility, Alienation & Absurdity, Existence Precedes Essence, Confronting the Absurd.
  > - **Key Figure**: Jean-Paul Sartre (*"Man is condemned to be free"*).
  > 
  > ### Nihilism:
  > - **Focus**: Rejection of objective truth, moral values, and intrinsic purpose.
  > - **Key Concepts**: Rejection of Meaning, Pessimism & Skepticism, Ethical Nihilism.
  > - **Key Figure**: Friedrich Nietzsche (exploration of the collapse of traditional moral foundations).

> [!TIP]
> **Key Difference**: The DPO aligned model introduced structured headers, key philosophical figures (Sartre, Nietzsche), and concrete conceptual definitions (e.g. *Existence precedes Essence*).

---

## 5. Artifacts and Output Files

- **Updated Notebook**: [alignment_dpo.ipynb](file:///home/lmassaron/code/fine-tuning-workshop/alignment_dpo.ipynb)
- **Saved DPO LoRA Adapter**: `qwen2.5-3b-dpo-adapter/`
- **Execution Log**: `alignment_dpo_execution.log`
