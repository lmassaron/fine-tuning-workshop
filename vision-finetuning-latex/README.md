# Multimodal LaTeX OCR Transcription (Qwen2-VL + Unsloth)

This module fine-tunes the lightweight Vision-Language Model `unsloth/Qwen2-VL-2B-Instruct` to transcribe images of complex handwritten and printed mathematical formulas into compilable LaTeX code.

## 📌 Theme & Purpose
Vision-Language Models (VLMs) bridge vision transformers and autoregressive language decoders via projection layers. Fine-tuning a multimodal architecture requires:
- Preserving general visual features by freezing the Vision Transformer (ViT) encoder.
- Applying Low-Rank Adaptation (LoRA) specifically to projection and language attention modules.
- Handling PIL image columns distinctly to prevent Hugging Face `datasets` serialization errors.

### Key Capabilities
- **FastVisionModel Training**: Utilizes Unsloth's optimized vision collators and forward pass.
- **Accurate Mathematical Syntax**: Generalizes across integrals, Greek characters, and matrix indices.
- **Inference Acceleration**: Switches dynamically to `FastVisionModel.for_inference` for high-throughput transcription.

---

## 📂 Notebook & Documentation

- **`vision_finetuning_latex.ipynb`**:
  - Complete multimodal fine-tuning notebook on `unsloth/LaTeX_OCR`.
- **`vision_finetuning_latex_walkthrough.md`**:
  - Detailed architectural guide covering multimodal tokens, collator construction, and training logs.

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
Select the **Python (Vision Fine-Tuning LaTeX)** kernel in Jupyter.
