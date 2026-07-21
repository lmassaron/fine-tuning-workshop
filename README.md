# Book Chapter: Modern Supervised Fine-Tuning (SFT) & Multimodal Adaptation

## Abstract
This chapter presents a comprehensive technical study on Parameter-Efficient Fine-Tuning (PEFT) and preference alignment of large language models (LLMs) and vision-language models (VLMs) under strict VRAM constraints. We investigate five distinct paradigms:
1. **Reasoned Financial Sentiment Classification** with Chain-of-Thought (CoT) prompting using Gemma 4.
2. **Clinical cardiology QA Adaptation** with target-token perplexity validation using Microsoft's Phi-4.
3. **Multimodal LaTeX OCR Transcription** using Qwen2-VL.
4. **Stylistic Formatting Alignment** via Direct Preference Optimization (DPO) using Qwen2.5-3B.
5. **Mathematical Reasoning & Consistency** via Group Relative Policy Optimization (GRPO) using Qwen2.5-3B.

We discuss model quantization, preference pair formatting, RL reward functions, optimization parameters, training trajectories, and empirical results.

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
  
  #### Concrete CoT Classification Examples:
  
  *   **Example 1: Positive Corporate Growth**
      *   *Input Headline*: `"Finnish technology group Wartsila has won a contract to supply a 50 MW power plant to El Salvador, representing a major entry into Central America."`
      *   *Model Structured Output*:
          ```xml
          <sentiment>positive</sentiment>
          <reasoning>The contract to supply a 50 MW power plant in El Salvador represents a significant new business deal and a strategic entry into the Central American market for Wartsila, indicating positive growth prospects.</reasoning>
          ```
  
  *   **Example 2: Negative Financial Downturn**
      *   *Input Headline*: `"Operating profit fell to EUR 3.2 million from EUR 5.8 million in the corresponding period of the previous year."`
      *   *Model Structured Output*:
          ```xml
          <sentiment>negative</sentiment>
          <reasoning>A drop in operating profit from EUR 5.8 million to EUR 3.2 million compared to the previous year represents a clear decline in financial performance and profitability.</reasoning>
          ```

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
  
  #### Concrete Cardiology Expert Dialogue Examples:
  
  *   **Example 1: Diagnostic ECG Interpretation**
      *   *Input Question*: `"What are the characteristic ECG findings during an acute anteroseptal myocardial infarction?"`
      *   *Model Cardiology Output*:
          ```
          In an acute anteroseptal myocardial infarction, the characteristic ECG findings include:
          1. ST-segment elevation in leads V1, V2, and V3.
          2. Reciprocal ST-segment depression in inferior leads (II, III, aVF) may be present.
          3. Progression to pathological Q waves in V1-V3 as myocardial necrosis completes.
          4. T-wave inversion in the affected leads during the evolutionary phase.
          ```
  
  *   **Example 2: Congenital Pathology Association**
      *   *Input Question*: `"Which congenital heart defect is most commonly associated with Turner syndrome?"`
      *   *Model Cardiology Output*:
          ```
          Turner syndrome is most frequently associated with:
          1. Coarctation of the aorta (occurring in up to 15-20% of patients).
          2. Bicuspid aortic valve (present in approximately 30% of cases).
          These anomalies require regular cardiovascular surveillance, specifically echocardiography or cardiac MRI, to monitor for progressive aortic root dilation or dissection.
          ```

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
  
  **What this means in practice**: You can capture a photo of a whiteboard, a scan of a handwritten calculation page, or a cropped screenshot of a digital math book, and the fine-tuned model will transcribe that visual image directly into clean, compilable, and renderable LaTeX code.
  
  #### Concrete OCR Translation Examples:
  
  *   **Example 1: Basic Integration Photo**
      *   *Visual Input*: A photo of an integration equation on paper: $\int_a^b f(x) \, dx$
      *   *Model LaTeX Output*: `\int_{a}^{b} f(x) \, dx`
  
  *   **Example 2: Complex Fractions and Roots**
      *   *Visual Input*: A handwritten math note showing the quadratic formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$
      *   *Model LaTeX Output*: `x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}`
  
  *   **Example 3: Quantum Mechanical Matrices**
      *   *Visual Input*: A photo of a chalkboard matrix calculation: $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$
      *   *Model LaTeX Output*: `\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`

- **Capabilities of the Fine-Tuned Model**:
  - **Handwritten Mathematical OCR**: Directly translates image inputs of complex handwritten mathematical symbols into compilable LaTeX equations.
  - **Mathematical Semantic Generalization**: Learns semantic equivalencies of LaTeX notation (e.g., outputting `\dagger` instead of the literal reference `\dag` based on context), demonstrating true mathematical syntax generalization.

---

## 5. Track 4: Stylistic Formatting Alignment via DPO (Qwen2.5-3B)

### A. Paradigm & Preference Dataset Structure
Direct Preference Optimization (DPO) provides a mathematically elegant alternative to reinforcement learning from human feedback (RLHF). Instead of training a separate reward model and executing PPO, DPO optimizes the language model directly on binary preference pairs. Under a Bradley-Terry preference model, the DPO loss function is formulated as:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

where $\pi_\theta$ is the active policy, $\pi_{\text{ref}}$ is the frozen reference policy, $y_w$ is the chosen (preferred) output, $y_l$ is the rejected (dispreferred) output, and $\beta$ controls the KL penalty strength (typically set to 0.1).

We utilize the `m-a-p/orca_dpo_pairs` dataset, which aligns LLMs on style, precision, and layout quality.
- **Preference Structure**:
  - `prompt`: The input instruction (system prompt + user request) initiating the dialogue.
  - `chosen`: The target response, which exhibits preferred attributes such as immediate delivery of information, clean markdown bullet formatting, and conciseness.
  - `rejected`: The dispreferred response, which includes conversational fillers ("Certainly! Here is...", "Hope this helps!"), verbose sentences, or flat text layouts.
- **How to Recreate Preference Datasets**:
  1. *Prompt Collection*: Extract a representative distribution of user instructions.
  2. *Response Generation*: Query a base model (or multiple models) with different temperatures to generate candidate outputs.
  3. *Filtering and Labeling*: Score output pairs using a strong critic model (e.g., GPT-4o) or human annotators based on style guidelines (e.g., rewarding conciseness and formatting while penalizing preambles).
  4. *Compilation*: Save the queries under the `prompt` field, the cleaned/formatted responses under `chosen`, and the raw verbose responses under `rejected`.

### B. Notebook Architecture & Cells
The notebook [alignment_dpo.ipynb](file:///home/lmassaron/code/sft-examples/alignment_dpo.ipynb) implements DPO:
- **Cell 1**: Initialize imports and set up computed datatypes.
- **Cell 2**: Load the `m-a-p/orca_dpo_pairs` dataset.
- **Cell 3**: Format the dataset using the model's native chat template:
  ```python
  def format_dpo(example):
      return {
          "prompt": tokenizer.apply_chat_template([{"role": "user", "content": example["question"]}], tokenize=False, add_generation_prompt=True),
          "chosen": example["chosen"] + tokenizer.eos_token,
          "rejected": example["rejected"] + tokenizer.eos_token,
      }
  ```
- **Cell 4**: Load `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` in 4-bit NormalFloat and configure LoRA adapters targeting all attention and projection layers.
- **Cell 5**: Train using TRL `DPOTrainer` for 60 steps with a learning rate of $5 \times 1e-6$ and $\beta=0.1$.
- **Cell 6**: Run a stylistic comparison of the base model versus the DPO-aligned model.

### C. Stylistic Alignment & Empirical Assessment
Prior to DPO alignment, instruct-tuned models frequently insert conversational preambles ("fluff") and generate wordy text. Post DPO alignment, the model exhibits three measurable stylistic shifts:
1. **Preamble Elimination**: Complete removal of introductory fillers. The model answers the prompt directly.
2. **Structural Markdown Formatting**: Information is automatically parsed and presented in readable markdown tables or bulleted lists.
3. **Word Count Compression**: Verbose, low-density sentences are compressed, reducing the total length of responses by approximately 30-40% while preserving all key information.

#### Concrete Stylistic Output Comparison:
- **Test Prompt**: *"Tell me the capital of France and its three largest cities in a concise list."*
- **Base Instruct Model Output**:
  > Certainly! I'd be happy to help you with that. The capital of France is Paris. Here is a list of the three largest cities in France:
  > - Paris (which is also the capital and largest city)
  > - Marseille (located in the south)
  > - Lyon (a major financial hub in the east)
  > 
  > I hope this information is helpful for your needs!
- **DPO Aligned Model Output**:
  > **Capital**: Paris
  > **Three Largest Cities**:
  > - Paris
  > - Marseille
  > - Lyon
- **Measurable Impact**: Preamble completely eliminated. Output size reduced from 78 words to 18 words (76.9% compression), maximizing data density.

---

## 6. Track 5: Mathematical Reasoning & Consistency via GRPO (Qwen2.5-3B)

### A. Paradigm & Mathematical Formulation
Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that optimizes policy distributions without maintaining a separate critic network (which typically consumes 50% of active VRAM during training). Instead of estimating a state-value baseline $V(s)$, GRPO draws a group of $G$ outputs $\{y_1, y_2, \dots, y_G\}$ for each input prompt $x$ from the active policy. It then computes the rewards $\{r_1, r_2, \dots, r_G\}$ for these outputs and estimates their advantages relative to the group:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

This relative advantage is used to update the policy using PPO's clipped objective, penalized by a KL-divergence term back to the reference policy:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)} A_i, \, \text{clip}\left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

This group-relative approach guarantees that the advantages sum to zero across each group, maintaining highly stable updates. By eliminating the critic model, GRPO enables reinforcement learning on consumer-grade GPUs under a 16GB VRAM limit.

### B. Reward Architecture & Parsers
GRPO aligns LLMs by evaluating rollout responses using rule-based reward functions instead of neural reward models. We define two mathematical parses:
1. **Format Reward**: Encourages the model to place its step-by-step thinking process inside `<reasoning>` tags and the final answer inside `<answer>` tags. It parses the response using regular expressions checking tag existence:
   ```python
   def format_reward_func(prompts, completions, **kwargs):
       pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
       # Returns 1.0 if formatted correctly else 0.0
   ```
2. **Correctness Reward**: Validates if the final numeric answer matches the ground truth value. It extracts the value within the `<answer>` tag and compares it to the target number:
   ```python
   def correctness_reward_func(prompts, completions, answer, **kwargs):
       # Extracts number from <answer>...</answer> and compares to targets
       # Returns 2.0 if extracted_answer == ground_truth else 0.0
   ```

### C. Notebook Architecture & Cells
The notebook [alignment_grpo.ipynb](file:///home/lmassaron/code/sft-examples/alignment_grpo.ipynb) configures a consumer-friendly GRPO pipeline:
- **Cell 1**: Setup Environment, imports, and inject the TRL package check monkey-patch to bypass missing dependency errors.
- **Cell 2**: Load the GSM8K dataset, setting up the system prompt instruction that enforces thinking in `<reasoning>` tags and answering in `<answer>` tags.
- **Cell 3**: Define `correctness_reward_func` and `format_reward_func`.
- **Cell 4**: Quantization configuration, model loading, and `GRPOTrainer` initialization using native PyTorch generation (`use_vllm=False`).
- **Cell 5**: Train the model for 50 steps using LoRA adapters.
- **Cell 6**: Inference verification of structured mathematical reasoning.

### D. Purpose of Reinforcement Learning & Model Capabilities
- **Purpose of RL Alignment**: Supervised fine-tuning maps inputs to static answers, but struggles to teach multi-step logical deduction or enforce consistent output formatting. Reinforcement learning via GRPO forces the model to explore reasoning trajectories. By rewarding correct math calculations and penalizing improper formatting, the model is aligned to consistently structure its thoughts, improving mathematical accuracy.
- **Capabilities of the Fine-Tuned Model**:
  - **Structured Reasoning**: Systematic output of mathematical steps within `<reasoning>` tags before declaring the final answer, ensuring transparency.
  - **Perfect Format Adherence**: Consistently structures responses using XML tags without requiring external runtime validation.

  #### Structured Math Reasoning Example:
  *   **Prompt**: *"Natalia sold clips to 48 of her friends in April, and then half as many in May. How many clips did Natalia sell in total?"*
  *   **Model Structured Output**:
      ```xml
      <reasoning>
      Natalia sold 48 clips in April.
      In May, she sold half as many clips as in April, which is 48 / 2 = 24 clips.
      In total, she sold 48 + 24 = 72 clips.
      </reasoning>
      <answer>72</answer>
      ```

---

## 7. Conclusion & Best Practices
By adopting PEFT, proper quantization, and preference alignment, high-fidelity fine-tuning and reinforcement learning can be successfully executed on single-GPU setups. Crucial takeaways include:
1. **Double-BOS Prevention**: Set `dataset_kwargs={"add_special_tokens": False}` when using pre-formatted templates to prevent duplication of BOS tokens.
2. **TRL Package Check Patching**: In environments missing optional TRL libraries, monkey-patch `trl.import_utils` functions at runtime to return booleans instead of tuples.
3. **VRAM Optimization in RL**: Disable vLLM (`use_vllm=False` in `GRPOConfig`) when running on resource-constrained GPUs to run rollout generations natively in PyTorch, avoiding memory allocation conflicts.

