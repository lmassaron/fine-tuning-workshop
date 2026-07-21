# Walkthrough: Multimodal Vision-Language LaTeX OCR Fine-Tuning

This walkthrough details the multimodal architecture, Unsloth vision collator, dataset formatting, and evaluation for [vision_finetuning_latex.ipynb](file:///home/lmassaron/code/sft-examples/vision_finetuning_latex.ipynb).

---

## 1. Executive Summary & Goal

Multimodal Vision-Language Models (VLMs) process image inputs and generate text. While general-purpose VLMs can caption photos, they struggle to transcribe complex mathematical equations into compilable LaTeX due to fine-grained visual symbols and nested spatial structures. This track fine-tunes **`unsloth/Qwen2-VL-2B-Instruct`** on the **`unsloth/LaTeX_OCR`** dataset to build an expert model capable of converting formula images into clean LaTeX code.

---

## 2. Multimodal Architecture & Vision-PEFT

- **Base VLM**: `unsloth/Qwen2-VL-2B-Instruct` (2.2B parameters)
- **Quantization**: 4-bit NormalFloat (`load_in_4bit=True`)
- **FastVisionModel Adapters**:
  - `finetune_vision_layers = True` (Adapts visual attention encoder blocks)
  - `finetune_language_layers = True` (Adapts causal LLM decoder)
  - `finetune_attention_modules = True` & `finetune_mlp_modules = True`
  - Rank $r=16$, $\alpha=16$, Dropout $= 0$
- **Gradient Checkpointing**: Enabled via Unsloth (`use_gradient_checkpointing='unsloth'`)

---

## 3. Practical Dataset Formatting & Column Retention

### Dataset Preparation
We load `unsloth/LaTeX_OCR`, selecting 500 training samples and 50 validation samples:
```python
def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write the LaTeX representation for this image.",
                },
                {"type": "image"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation, "images": [sample["image"]]}
```

### Critical Practicality 1: Dedicated Images Column
PIL Image objects must be returned in a separate `'images'` column rather than nested inside the `'messages'` list. Nesting image objects inside lists causes Hugging Face `datasets` to fail serialization.

### Critical Practicality 2: Disabling Column Pruning
In `SFTConfig`, set `remove_unused_columns = False`. Standard Hugging Face trainers prune any dataset column not matching standard text model inputs (`input_ids`, `attention_mask`). Disabling column pruning preserves the `'images'` tensor column.

---

## 4. Training Engine & Vision Collator

- **Data Collator**: `UnslothVisionDataCollator(model, tokenizer)` — handles image patchification, 2D positional grid embeddings, and token sequence alignment automatically.
- **Batch Size**: 2 per device $\times$ 4 gradient accumulation steps = **Effective Batch Size 8**
- **Learning Rate**: `2e-4` with warmup over 5 steps for 30 max training steps.
- **Precision**: `bf16` compute dtype.

---

## 5. Quantitative & Qualitative Inference Evaluation

At inference time, the model is switched to fast vision inference mode:
```python
FastVisionModel.for_inference(model)  # Enables 2x faster visual generation
```

### Concrete LaTeX Transcription Output
- **Input Image**: Complex integral formula containing sums and matrix traces.
- **Expected Reference LaTeX**:
  ```latex
  \Gamma _ { \sigma } + \Gamma _ { m } = \int d ^ { 2 } x [ - \frac { 1 } { 8 \pi } T r ( \partial _ { \mu } U \partial _ { \mu } U ^ { \dag } ) + \frac { 1 } { 2 } m ^ { 2 } T r ( U + U ^ { \dag } - 2 ) ] ,
  ```
- **Generated Model LaTeX**:
  ```latex
  \Gamma _ { \sigma } + \Gamma _ { m } = \int d ^ { 2 } x [ - \frac { 1 } { 8 \pi } T r ( \partial _ { \mu } U \partial _ { \mu } U ^ { \dagger } ) + \frac { 1 } { 2 } m ^ { 2 } T r ( U + U ^ { \dagger } - 2 ) ] ,
  ```

### Analysis of Generalization
The generated LaTeX is mathematically identical to the source image. The model outputted `\dagger` instead of the literal reference shorthand `\dag`. Both compile to the identical mathematical symbol ($\dagger$), proving that the VLM acquired semantic understanding of LaTeX rendering conventions rather than rote character memorization.

---

## 6. Artifact Outputs
- Fine-tuned multimodal adapter saved at: `qwen2-vl-latex-adapter/`
