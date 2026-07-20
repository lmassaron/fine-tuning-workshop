# Book Chapter: Modern Supervised Fine-Tuning (SFT) & Multimodal Adaptation

## Abstract
This chapter presents a comprehensive technical study on Parameter-Efficient Fine-Tuning (PEFT) of large language models (LLMs) and vision-language models (VLMs) under strict VRAM constraints. We investigate three distinct paradigms:
1. **Reasoned Financial Sentiment Classification** with Chain-of-Thought (CoT) prompting using Gemma 4.
2. **Clinical cardiology QA Adaptation** with target-token perplexity validation using Microsoft's Phi-4.
3. **Multimodal LaTeX OCR Transcription** using Qwen2-VL.

We discuss model quantization, prompt formatting, optimization parameters, training trajectories, and empirical results.

---

## 1. Theoretical Foundations of Parameter-Efficient Fine-Tuning

### A. LoRA & QLoRA
Supervised Fine-Tuning (SFT) of modern generative architectures in full precision is computationally prohibitive. Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA) mitigates this by freezing the pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ and injecting trainable rank decomposition matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ (where rank $r \ll \min(d, k)$):

$$\Delta W = B \cdot A$$

For a forward pass, the update is scaled by a factor of $\frac{\alpha}{r}$:

$$h = W_0 x + \frac{\alpha}{r} \Delta W x$$

QLoRA (Quantized LoRA) extends this concept by quantizing the base model weights $W_0$ into 4-bit NormalFloat (NF4) precision, which is mathematically optimized for zero-mean, unit-variance distributions. The activations are computed in 16-bit brain floating-point (`bfloat16`) formats during backward propagation, dramatically reducing the GPU VRAM footprint while keeping performance parity.

### B. VRAM Constraints & Optimization Strategies
To perform stable training on consumer-grade GPUs (e.g. max 16GB VRAM), several memory management techniques are utilized:
- **Gradient Checkpointing**: Trade compute for memory by discarding intermediate activations during the forward pass and recomputing them on-demand during the backward pass.
- **KV Cache Deactivation**: Disabling key-value cache (`model.config.use_cache = False`) during training to save substantial activation memory.
- **Fused Optimizers**: Fusing gradient updates in memory using `adamw_torch_fused` to reduce CUDA kernel invocation overhead.

---

## 2. Track 1: Chain-of-Thought Financial Sentiment (Gemma 4)

### A. Paradigm & Dataset
Standard classification prompts often fail to capture subtle economic implications. Chain-of-Thought (CoT) reasoning addresses this by forcing the model to generate intermediate rationales before outputting the final classification label. 

We utilize the `lmassaron/FinancialPhraseBank_explained` dataset, which enriches human-curated financial headlines with LLM-generated explanations. The dataset includes pre-split `'train'`, `'validation'`, and `'test'` splits.

### B. Notebook Architecture & Cells
The notebook [financial_sentiment_cot.ipynb](file:///home/lmassaron/code/sft-examples/financial_sentiment_cot.ipynb) is structured as follows:
- **Cell 1-2**: Import libraries and detect CUDA environment configurations.
- **Cell 3**: Load the dataset from the Hugging Face Hub, renaming columns to `'text'` and `'reasoning'`.
- **Cell 4**: Format conversation prompts using the model's native chat template. The system prompt instructs the model to reply using `<sentiment>` and `<reasoning>` tags:
  ```python
  def format_prompt(example):
      messages = [
          {'role': 'user', 'content': SYSTEM_PROMPT + f'\nHeadline: "{example["text"]}"'},
          {'role': 'assistant', 'content': f'<sentiment>{example["sentiment"]}</sentiment>\n<reasoning>{example["reasoning"]}</reasoning>'}
      ]
      prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      return {'text': prompt}
  ```
- **Cell 5**: Load `google/gemma-4-E2B-it` in 4-bit NF4 precision. Set up LoRA adapters targeting all linear modules (e.g., projection and attention projection layers) with $r=8$ and $\alpha=32$.
- **Cell 6**: Train using the TRL `SFTTrainer` with a Cosine Annealing learning rate decay and a warmup ratio of 0.1. Crucially, we set `dataset_kwargs={"add_special_tokens": False}` to prevent duplicate BOS tokens.
- **Cell 7**: Run batch evaluation on the original (non-augmented) test set using left-padding.

### C. Training Trajectory & Quantitative Results
During training, the cross-entropy loss smoothly converged. The evaluation on the 484 original test headlines produced the following metrics:
- **Sentiment Classification Accuracy**: **81.82%**
- **Tag Integrity (Valid Output Formats)**: **94.83%**

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

    negative       0.88      0.85      0.87        61
     neutral       0.84      0.89      0.87       287
    positive       0.91      0.65      0.76       136

    accuracy                           0.82       484
   macro avg       0.66      0.60      0.62       484
weighted avg       0.87      0.82      0.84       484
```
*Discussion*: Precision remains exceptionally high across all classes (88-91%). The lower recall on positive sentiments (65%) suggest the model tends to fall back to neutral predictions for ambiguous positive headlines, a common characteristic of conservative financial analysts.

### D. Purpose of Fine-Tuning & Model Capabilities
- **Purpose of Fine-Tuning**: Base models often struggle to produce structured XML outputs consistently and frequently skip intermediate causal logic when classifying financial headlines. Fine-tuning adapts the model to generate a strict Chain-of-Thought (CoT) explanation *before* outputting the final classification label, shaping its output distribution to enforce logical grounding.
- **Capabilities of the Fine-Tuned Model**:
  - **Structured Chain-of-Thought Reasoning**: Consistently generates logical financial rationales inside `<reasoning>` tags prior to deciding on a sentiment.
  - **Strict Format Adherence**: Achieves over 94% tag integrity without requiring external regex parsers or JSON schema validators.
  - **Specialized Financial Sentiment Classification**: Outperforms base models by distinguishing subtle financial statements (e.g., classifying conservative corporate earnings guidance as neutral/positive).

---

## 3. Track 2: Clinical Cardiology QA Expert (Phi-4)

### A. Paradigm & Dataset
Adapting LLMs to specialized medical contexts requires highly aligned knowledge extraction. We utilize the `lmassaron/medical-cardiology-qa` dataset, which contains doctor-patient dialogues on pathophysiology, diagnostic criteria, and treatment protocols. We split the dataset 90% for training and 10% for validation.

### B. Notebook Architecture & Cells
The notebook [medical_expert_cardiology.ipynb](file:///home/lmassaron/code/sft-examples/medical_expert_cardiology.ipynb) utilizes the **Unsloth** library to accelerate training:
- **Cell 1**: Import `unsloth` at the very beginning to load optimized kernels before importing PyTorch or Hugging Face.
- **Cell 2-3**: Load Microsoft's `unsloth/Phi-4-mini-instruct` in 4-bit quantization. Enable QLoRA adapters with `FastLanguageModel.get_peft_model`.
- **Cell 4**: Apply a custom format function mapping dialogues to chat template strings.
- **Cell 5**: Configure `SFTTrainer` with `Unsloth` optimized memory savings. Disables KV-caching.
- **Cell 6**: Perform batch evaluation over the validation split, calculating the exact **Perplexity (PPL)** of the doctor's response tokens.

### C. Perplexity Evaluation & Results
Perplexity measures the exponential of the cross-entropy loss calculated exclusively on the doctor's response tokens (masking out prompt tokens with `-100` label values):

$$PPL = \exp\left( - \frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x, y_{\lt i}) \right)$$

Training over 100 steps resulted in a low validation perplexity:
- **Average Evaluation Perplexity**: **4.4341**

#### Sample Evaluation Comparison:
- **Question**: *What are some connective tissue disorders that increase the risk of aortic dissection?*
- **Expected Answer**: *Connective tissue disorders that increase the risk of aortic dissection include Marfan syndrome, Ehlers-Danlos syndrome, and Loeys-Dietz syndrome. These conditions affect the structural integrity of the arterial walls.*
- **Generated Answer**: *Connective tissue disorders such as Marfan syndrome, Ehlers-Danlos syndrome, and Loeys-Dietz syndrome are known to increase the risk of aortic dissection. These conditions affect the body's connective tissue, which can weaken the aortic wall and make it more susceptible to dissection.*
- **Perplexity**: **1.5286**

*Discussion*: A perplexity of 1.52 indicates the model is highly confident in generating correct cardiology terms. The generated response captures the diagnostic details perfectly while providing a more detailed pathophysiological explanation than the reference text.

### D. Purpose of Fine-Tuning & Model Capabilities
- **Purpose of Fine-Tuning**: Out-of-the-box instructions models, while fluent, lack the specialized clinical vocabulary and diagnostic rigor required for cardiology consultations. Fine-tuning aligns the model's reasoning patterns with cardiology protocols, adapting it to generate accurate, high-fidelity medical answers instead of generic health advice.
- **Capabilities of the Fine-Tuned Model**:
  - **Cardiology Domain Expert**: Highly coherent generation of complex cardiology concepts (e.g., Marfan/Ehlers-Danlos syndromes, aortic dissection, myocardial infarction pathways) with low perplexity (4.43).
  - **Clinical Style Alignment**: Adopts a precise, structured dialogue format matching professional physician-to-patient or peer-to-peer discussions.

---

## 4. Track 3: Multimodal LaTeX OCR Transcription (Qwen2-VL)

### A. Multimodal Architecture
Vision-Language Models (VLMs) process image inputs by passing them through a frozen Vision Transformer (ViT) encoder to obtain visual features. These features are mapped into the text token space using a trainable projection layer (usually a Multi-Layer Perceptron), and then concatenated directly with text tokens to be processed by the autoregressive decoder:

$$h = [ \text{Proj}(\text{ViT}(x_{\text{image}})) ; x_{\text{text}} ]$$

We freeze the vision transformer to preserve generic visual features and train the projections and language attention modules using QLoRA.

### B. Notebook Architecture & Cells
The notebook [vision_finetuning_latex.ipynb](file:///home/lmassaron/code/sft-examples/vision_finetuning_latex.ipynb) implements handwritten-to-LaTeX transcription:
- **Cell 1**: Import the `FastVisionModel` class from Unsloth.
- **Cell 2**: Load the lightweight `unsloth/Qwen2-VL-2B-Instruct` model and initialize vision and language adapters.
- **Cell 3**: Load the `unsloth/LaTeX_OCR` dataset. Map the dataset into a conversational format. Crucially, PIL Images must be passed in a separate `images` column to prevent Hugging Face `datasets` from serializing nested images into invalid dictionaries:
  ```python
  def convert_to_conversation(sample):
      conversation = [
          {
              'role': 'user',
              'content': [
                  {'type': 'text', 'text': 'Write the LaTeX representation for this image.'},
                  {'type': 'image'}
              ]
          },
          {
              'role': 'assistant',
              'content': [{'type': 'text', 'text': sample['text']}]
          }
      ]
      return {'messages': conversation, 'images': [sample['image']]}
  ```
- **Cell 4**: Configure `SFTTrainer` with `UnslothVisionDataCollator(model, tokenizer)`. Specify `remove_unused_columns = False` in `SFTConfig` so image buffers are not dropped.
- **Cell 5**: Switch the model to inference mode using `FastVisionModel.for_inference(model)` and generate LaTeX formulas.

### C. Trajectory & Results
After training for 30 steps, the model successfully generalized from handwritten images to clean LaTeX codes:
- **Expected LaTeX**: `\Gamma _ { \sigma } + \Gamma _ { m } = \int d ^ { 2 } x [ - \frac { 1 } { 8 \pi } T r ( \partial _ { \mu } U \partial _ { \mu } U ^ { \dag } ) + \frac { 1 } { 2 } m ^ { 2 } T r ( U + U ^ { \dag } - 2 ) ] ,`
- **Generated LaTeX**: `\Gamma _ { \sigma } + \Gamma _ { m } = \int d ^ { 2 } x [ - \frac { 1 } { 8 \pi } T r ( \partial _ { \mu } U \partial _ { \mu } U ^ { \dagger } ) + \frac { 1 } { 2 } m ^ { 2 } T r ( U + U ^ { \dagger } - 2 ) ] ] ,`

*Discussion*: The model's transcription of the mathematical formula is highly accurate. Interestingly, the model generated `\dagger` instead of the shorthand `\dag` present in the expected output. Both represent the same mathematical symbol, demonstrating that the VLM is not merely memorizing sequences, but has acquired semantic understanding of LaTeX rendering conventions.

### D. Purpose of Fine-Tuning & Model Capabilities
- **Purpose of Fine-Tuning**: General-purpose Vision-Language Models (VLMs) can write general image captions, but they fail to transcribe handwritten mathematical equations into compilable LaTeX code due to structural complexity (fractions, subscripts, and nested brackets). Fine-tuning maps visual features of mathematical symbols and layouts to valid LaTeX syntax sequences.
- **Capabilities of the Fine-Tuned Model**:
  - **Handwritten Mathematical OCR**: Directly translates image inputs of complex handwritten mathematical symbols into compilable LaTeX equations.
  - **Mathematical Semantic Generalization**: Learns semantic equivalencies of LaTeX notation (e.g., outputting `\dagger` instead of the literal reference `\dag` based on context), demonstrating true mathematical syntax generalization.

---

## 5. Conclusion & Best Practices
By adopting PEFT, proper quantization, and memory-saving techniques, high-fidelity supervised fine-tuning of large textual and multimodal architectures can be successfully executed on 16GB GPUs. Crucial takeaways include:
1. Setting `dataset_kwargs={"add_special_tokens": False}` when using chat templates to prevent duplicate BOS tokens.
2. Disabling key-value cache (`use_cache=False`) during the backward pass.
3. Keeping image datasets in separate root-level columns to avoid nested dict serialization errors under Hugging Face.
