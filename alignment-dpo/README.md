# Direct Preference Optimization (DPO)

This module implements pairwise human preference alignment on `Qwen/Qwen2.5-3B` using Hugging Face TRL's `DPOTrainer`.

## 📌 Theme & Purpose
Direct Preference Optimization (DPO) replaces traditional complex Reinforcement Learning from Human Feedback (RLHF) pipelines (which require training a separate reward model followed by PPO) by directly optimizing a closed-form implicit reward:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### Key Capabilities
- **Direct Implicit Reward**: Steers responses towards chosen completions ($y_w$) and away from rejected completions ($y_l$) without reward model instability.
- **Reference Model Caching**: Uses PEFT adapters to evaluate both policy $\pi_\theta$ and reference $\pi_{\text{ref}}$ with a single base model weights load.
- **Benchmark Stability**: Verified on preference datasets (Argilla / UltraFeedback).

---

## 📂 Notebook & Documentation

- **`alignment_dpo.ipynb`**:
  - Full execution pipeline loading preference data, formatting chosen/rejected pairs, training via `DPOTrainer`, and validating preference probability margins.
- **`alignment_dpo_walkthrough.md`**:
  - Comprehensive guide covering loss mathematics, hyperparameters ($\beta$, learning rate), and empirical output evaluations.

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
Select the **Python (Alignment DPO)** kernel in Jupyter.
