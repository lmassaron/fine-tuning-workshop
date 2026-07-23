# Walkthrough: Group Relative Policy Optimization (GRPO) Alignment

This walkthrough details the technical architecture, mathematical formulation, reward functions, implementation workflow, and 100-problem benchmark evaluation results for [alignment_grpo.ipynb](file:///home/lmassaron/code/sft-examples/alignment_grpo.ipynb).

---

## 1. Executive Summary & Goal

Mathematical reasoning requires both strict structural compliance (enforcing step-by-step thinking inside `<reasoning>` tags and final numbers inside `<answer>` tags) and numerical accuracy. Standard SFT maps inputs to static answers but fails to incentivize exploration of logical deduction paths.

This track implements **Group Relative Policy Optimization (GRPO)** on **`Qwen/Qwen2.5-0.5B-Instruct`** over 1000 steps using reinforcement learning rewards on the `openai/gsm8k` dataset with **TRL v1.9.0** (no monkey-patches required).

### Key Highlights
- **100-Problem GSM8K Benchmark**: Evaluates pre- and post-GRPO performance on a 100-problem slice of the GSM8K test set.
- **Decoupled Math Extraction Rule**: Decouples formatting compliance from raw reasoning ability by extracting the last number appearing in the generated text (`extract_last_number`).
- **Decoupled Math Reasoning Accuracy**: Increases raw math reasoning accuracy from **30.0% to 41.0% (+11.0% Net Reasoning Delta)**.
- **Strict Formatted Math Accuracy**: Increases strict XML-wrapped accuracy from **0.0% to 41.0% (+41.0% Strict Delta)**.
- **Format Compliance Jump**: Boosts XML tag adherence from **0.0% to 81.0% (+81.0% Format Delta)**.

---

## 2. Mathematical Formulation & Architecture

Unlike traditional PPO which requires maintaining a separate critic model $V_\phi(s)$ of identical size to the policy, GRPO samples a group of $G$ candidate outputs $\{\text{y}_1, \text{y}_2, \dots, \text{y}_G\}$ for each prompt $x$. It computes their rewards $\{\text{r}_1, \text{r}_2, \dots, \text{r}_G\}$ and normalizes them into group-relative advantages:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

The policy $\pi_\theta$ is updated using PPO's clipped surrogate objective penalized by KL divergence against the reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)} A_i, \, \text{clip}\left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

---

## 3. Decoupled Answer Extraction & Reward Functions

We extract the final numerical answer using `extract_last_number(text)`, which searches inside `<answer>...</answer>` if tags exist, or extracts the last number from the free-form text if tags are omitted:

```python
def extract_last_number(text, start_tag="<answer>", end_tag="</answer>"):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, re.DOTALL)
    text_to_search = matches[-1] if matches else text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text_to_search)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""
```

### Reward Weights
1. **Format Reward (`format_reward`) — Weight: `0.5`**:
   Checks whether the generated text strictly matches the XML structure `^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>$`.

2. **Correctness Reward (`correctness_reward`) — Weight: `2.0`**:
   Extracts the numeric answer using `extract_last_number` and compares it to the ground truth value.

---

## 4. Practical Training Parameters & Setup

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B parameters; ideal capacity for fast, non-trivial RL exploration |
| **Framework Version** | `trl==1.9.0` | Modern TRL release; returns pure booleans without monkey patches |
| **Generations per Prompt** | `num_generations=8` | Samples 8 rollouts per prompt to estimate group advantage |
| **Batch Size** | `2` per device $\times$ `4` grad accum | Effective Batch Size of 8 prompts (64 rollouts per optimization step) |
| **Max Completion Length**| `384` tokens | Prevents reasoning truncation |
| **Max Training Steps** | `1000` steps (`warmup_steps=25`) | Full convergence over GSM8K training dataset |
| **KL Penalty ($\beta$)** | `0.05` | Low KL penalty permits policy exploration while preventing divergence |

---

## 5. Quantitative Decoupled GSM8K Benchmark Results

```
============================================================
>>> DECOUPLED GSM8K 100-PROBLEM BENCHMARK RESULTS <<<
Extraction Method: Last Number in Text (extract_last_number)
Pre-GRPO Format Compliance:  0.0%
Post-GRPO Format Compliance: 81.0%
Format Compliance Delta:    +81.0%

Pre-GRPO Math Accuracy:      30.0%
Post-GRPO Math Accuracy:     41.0%
Math Accuracy Delta:        +11.0%
============================================================
```

### Result Discussion
1. **Decoupled Math Reasoning (30.0% $\rightarrow$ 41.0%, +11.0% Delta)**: Extracting the last number from the generated text reveals that the un-finetuned base model possessed an underlying 30.0% mathematical reasoning capacity. GRPO training improved raw mathematical reasoning accuracy to **41.0% (+11.0% net reasoning gain)**.
2. **Format Compliance (0.0% $\rightarrow$ 81.0%, +81.0% Delta)**: Pre-GRPO generated plain text prose without XML tags (0/100). GRPO alignment achieved **81.0% XML tag compliance**.
3. **Strict Formatted Accuracy (0.0% $\rightarrow$ 41.0%, +41.0% Delta)**: Combining format compliance and mathematical accuracy, strict XML-wrapped accuracy increased from **0.0% to 41.0%**.

---

## 6. Artifact Outputs
- Saved PEFT LoRA adapter: `qwen2.5-0.5b-grpo-adapter/`
