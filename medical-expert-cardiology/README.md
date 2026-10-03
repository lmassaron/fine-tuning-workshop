# Clinical Cardiology QA Expert (Phi-4-mini + Unsloth)

This module fine-tunes Microsoft's `unsloth/Phi-4-mini-instruct` on doctor-patient cardiology consultations using 4-bit QLoRA and Unsloth acceleration.

## 📌 Theme & Purpose
Clinical medical reasoning demands high factual fidelity and specialized terminology without prompt dilution. This track fine-tunes Phi-4-mini on the `lmassaron/medical-cardiology-qa` dataset, which contains doctor-patient dialogues spanning pathophysiology, diagnostic procedures, and pharmacological management.

### Key Capabilities
- **Target-Token Perplexity (PPL)**: Masks out user prompt tokens (`-100` label values) to compute perplexity strictly on clinical doctor response tokens.
- **Unsloth Memory Optimization**: Reduces VRAM consumption by 70% using custom Triton cross-entropy kernels, enabling fine-tuning on single 16GB GPUs.
- **Low Validation Perplexity**: Drops validation perplexity to 4.43 PPL (with specific diagnostic questions reaching ~1.52 PPL).

---

## 📂 Notebook & Documentation

- **`medical_expert_cardiology.ipynb`**:
  - Unsloth-accelerated training notebook with 4-bit Phi-4-mini, custom chat formatting, training loop, and token-level PPL evaluation.
- **`medical_expert_cardiology_walkthrough.md`**:
  - Detailed cell-by-cell architectural walkthrough and evaluation analysis.

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
Select the **Python (Medical Expert Cardiology)** kernel in Jupyter.
