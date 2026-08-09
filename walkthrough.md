# Fine-Tuning Gemma 3 for Agentic Function Calling

In this chapter, we will explore how to fine-tune **Google's Gemma 3 270M IT**—the smallest model in the Gemma 3 family—to effectively perform function calling tasks. Transforming an instruction-tuned model into the "brain" of an autonomous agent requires specific data formatting, custom tokens, and Parameter-Efficient Fine-Tuning (PEFT). 

By the end of this walkthrough, you will understand the data being used, the objectives of fine-tuning, and how the results evaluate the model's performance as an agent.

---

## 1. The Goal: Why Fine-Tune for Function Calling?

> [!NOTE]
> **What is Function Calling?**
> Function calling is the ability of an LLM to reliably output structured data (often JSON) that matches a predefined schema, rather than just returning conversational text. This allows the model to interact with external tools, APIs, or databases.

When building an autonomous agent, the Large Language Model (LLM) acts as the reasoning engine. While the Gemma 3 270M IT is already instruction-tuned, it is not inherently an expert at strictly adhering to complex tool schemas, nor is it optimized for explicitly detailing its internal thought process before acting.

**The Purpose of this Fine-Tuning:**
1. **Tool Adherence:** Train the model to consistently use external tools when necessary, adhering exactly to the required JSON syntax and schema.
2. **Thought Process (`<think>`):** Encourage the model to articulate its reasoning before executing a tool.
3. **Special Tokens Handling:** Teach the model to respect custom boundaries between natural language, tool definitions, thoughts, and tool responses.

---

## 2. The Data: `hermes-function-calling-v1`

To teach our model these new skills, we use the `lmassaron/hermes-function-calling-v1` dataset. 

This dataset provides highly structured conversational turns. A typical example contains:
- **System Prompt:** Instructs the model on its identity and lists the available tools.
- **User Prompt:** A query that requires the use of one or more tools (e.g., "What is the weather in London?").
- **Agent Output:** The model's reasoning followed by the precise tool call.
- **Tool Response:** The structured data returned by the external tool.
- **Final Answer:** The agent interpreting the tool's response to answer the user.

### Special Tokens for Structure

To help Gemma 3 distinguish between these different conversational elements, we introduce several new special tokens to the ChatML format:
- `<tools>` / `</tools>`: Wraps the JSON schemas of available tools.
- `<think>` / `</think>`: Wraps the model's internal reasoning.
- `<tool_call>` / `</tool_call>`: Wraps the exact JSON invocation of the tool.
- `<tool_response>` / `</tool_response>`: Wraps the data returned by the tool execution.

> [!IMPORTANT]
> The tokenizer's vocabulary must be resized (`model.resize_token_embeddings`) so that these new tokens are assigned trainable embedding vectors. Without this, the model cannot learn their significance.

---

## 3. The Fine-Tuning Pipeline

Because retraining a model from scratch is computationally expensive, we use **LoRA (Low-Rank Adaptation)**.

### LoRA Configuration
Instead of updating all 270 million parameters, LoRA freezes the original model weights and injects small, trainable matrices into the model's layers.
- **Rank (`r=16`)**: Dictates the size of the injected matrices. A rank of 16 provides a strong balance between expressiveness and memory efficiency.
- **Target Modules**: We apply LoRA to the attention mechanism (`q_proj`, `k_proj`, etc.), the feed-forward network, and crucially, the `embed_tokens` and `lm_head`. Targeting the embedding layers ensures the model effectively learns how to use the newly introduced special tokens.

### Training Arguments (SFTTrainer)
Using the `TRL` library, we set up a `SFTTrainer`. 
Key settings include:
- **BFloat16 (`bf16=True`)**: Used for numerical stability and speed.
- **Gradient Accumulation (`steps=4`)**: Allows us to simulate a larger batch size (saving memory on smaller GPUs).
- **Max Length (`4096`)**: Ensures the model can process long conversations with extensive tool schemas.
- **Cosine Learning Rate Scheduler**: Gradually decreases the learning rate for stable convergence.

You can find the complete, simplified training script in `gemma3_270m_function_calling.py`.

---

## 4. Evaluation and Results

How do we know if our fine-tuned model is a better agent? We evaluate it using two key metrics:

### 1. Bag-of-Words Similarity (General Response Quality)
This metric computes the matching percentage of words between the model's generated conversational text and the ground-truth response. While standard instruction tuning provides a baseline, our fine-tuned model learns the *style* of the dataset, ensuring its natural language responses align with the persona of a helpful, reasoning agent.

### 2. Exact Sequence Match (Function Call Accuracy)
> [!TIP]
> This is the most critical metric for function calling.

This function tokenizes both the model's generated tool call and the expected ground truth, finding the longest contiguous exact match. 
- A high score means the model correctly generated the function name and exactly matched the required JSON syntax for the arguments.
- **Results:** Post-fine-tuning, the Gemma 3 270M model shows a dramatic increase in exact sequence matches compared to the baseline. It learns to cleanly wrap its JSON outputs in `<tool_call>` tags without introducing syntax errors or hallucinations.

## Conclusion

By meticulously formatting our dataset with custom tokens and utilizing LoRA to update the embedding and attention layers, we successfully transformed the lightweight **Gemma 3 270M IT** into a highly capable agentic component. 

This model can now reliably process tool schemas, articulate its reasoning, and output precise, parseable function calls—proving that even small models can power sophisticated autonomous workflows when fine-tuned correctly.
