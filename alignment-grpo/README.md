# Group Relative Policy Optimization (GRPO)

This module implements Reinforcement Learning for mathematical reasoning and XML format compliance using DeepSeek's **Group Relative Policy Optimization (GRPO)** on `Qwen/Qwen2.5-0.5B-Instruct` using Hugging Face TRL and Unsloth.

## 📌 Theme & Purpose
Traditional PPO requires an auxiliary Critic (value model) that is often as large as the policy itself, consuming double the VRAM. GRPO eliminates the value network by sampling a group of $G$ candidate outputs for each prompt, computing their mean and standard deviation, and normalizing rewards relative to the group:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

### Key Capabilities
- **Dual Rule-Based Reward Functions**:
  1. `xml_format_reward_func`: Rewards strict `<reasoning>...</reasoning><answer>...</answer>` XML compliance (+1.0 for valid formatting).
  2. `accuracy_reward_func`: Extracts numerical values from `<answer>` tags and compares against GSM8K ground truth.
- **Dramatic Format & Accuracy Gains**: Shifts format compliance from 0% to 90%+ and significantly improves GSM8K math reasoning solve rates.
- **Unsloth & TRL Implementations**: Includes standard TRL implementation and Unsloth memory-optimized variant.

---

## 📂 Files & Artifacts

- **`alignment_grpo.ipynb`**:
  - Full GRPO training notebook using TRL `GRPOTrainer` on GSM8K problems.
- **`alignment_grpo_unsloth.ipynb`**:
  - Memory-optimized variant using Unsloth's fast LoRA kernels.
- **`alignment_grpo_walkthrough.md`**:
  - Detailed mathematical walkthrough, reward function specifications, and benchmark tables.
- **`execute_alignment_grpo.py`**:
  - Standalone headless Python runner for automated headless cluster training.
- **`checkpoint_curve.png`**, **`training_curves.png`**, **`training_curves_ma.png`**:
  - Training trajectory plots showing reward convergence.

---

## 🚀 Setup & Execution

### Prerequisites
- NVIDIA GPU with CUDA support
- [`uv`](https://docs.astral.sh/uv/) installed

### Installation
Run the environment installer to create a GPU `.venv`:

```bash
chmod +x install.sh
./install.sh
```

### Activate & Launch
```bash
source .venv/bin/activate
jupyter lab
```
Select the **Python (Alignment GRPO)** kernel in Jupyter.
