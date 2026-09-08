#!/usr/bin/env python3
import json
import re
import torch
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tools import read_file, write_file, list_files, AVAILABLE_TOOLS_SCHEMA

BASE_MODEL_ID = "unsloth/Qwen3.5-4B"

# Real Hugging Face adapter paths
LORA_ADAPTERS = {
    "planner": "lmassaron/planner-lora",
    "coder": "lmassaron/coder-lora",
    "reviewer": "lmassaron/reviewer-lora",
}

def prepare_base_model(base_model_id: str):
    """Loads tokenizer and 4-bit quantized base model."""
    print(f"[LoRAManager] Loading base model: {base_model_id} (4-bit NF4)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
        trust_remote_code=True,
    )
    return base_model, tokenizer


def load_adapters(base_model, adapters: dict):
    """Loads and attaches PEFT LoRA adapters to the base model."""
    model = PeftModel.from_pretrained(
        base_model, adapters["planner"], adapter_name="planner"
    )
    model.load_adapter(adapters["coder"], adapter_name="coder")
    model.load_adapter(adapters["reviewer"], adapter_name="reviewer")
    model.eval()
    return model


class LoRAManager:
    """Dynamically swaps LoRA adapters on a single base model using pure Hugging Face PEFT."""

    def __init__(self):
        base_model, self.tokenizer = prepare_base_model(BASE_MODEL_ID)
        self.model = load_adapters(base_model, LORA_ADAPTERS)
        self.active_adapter = "planner"

    def set_role(self, role_name: str):
        """Swaps the active LoRA adapter."""
        if role_name not in LORA_ADAPTERS and role_name != "base":
            role_name = "base"
        gc.collect()
        torch.cuda.empty_cache()

        if role_name != "base":
            try:
                self.model.set_adapter(role_name)
                print(f"[LoRAManager] Swapped -> Active Role: {role_name.upper()}")
            except Exception as e:
                print(f"[LoRAManager] Failed to set adapter {role_name}: {e}")
        else:
            try:
                if hasattr(self.model, "disable_adapters"):
                    self.model.disable_adapters()
            except Exception:
                pass
            print("[LoRAManager] ⚡ Swapped -> Active Role: BASE")

        self.active_adapter = role_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text=text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response


def extract_plan_steps(plan_text: str) -> list[str]:
    """Extracts clean, numbered plan steps from model output."""
    matches = re.findall(r"<plan>(.*?)</plan>", plan_text, re.DOTALL)
    if matches:
        content = matches[-1]
    elif "<plan>" in plan_text:
        content = plan_text.split("<plan>")[-1]
    elif "Final Plan:" in plan_text:
        content = plan_text.split("Final Plan:")[-1]
    elif "Plan:" in plan_text:
        content = plan_text.split("Plan:")[-1]
    else:
        content = plan_text

    steps = []
    for line in content.split("\n"):
        line = line.strip()
        if re.match(r"^(\d+\.|\bStep\s+\d+:?)\s+.*", line, re.IGNORECASE):
            if any(h in line.lower() for h in ["analyze the request", "determine the steps", "drafting steps", "refining steps"]):
                continue
            steps.append(line)

    if not steps:
        for line in content.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line[:4]:
                steps.append(line)

    return steps


def create_plan(manager: LoRAManager, user_request: str) -> list[str]:
    """Generates and extracts execution steps using the planner role."""
    manager.set_role("planner")
    plan_prompt = f"""
    You are a Senior Software Architect. Break down this request into a series of small, executable steps.
    Request: {user_request}
    
    Output the final plan inside a <plan>...</plan> block, with each step on a new line starting with a number. Keep it concise.
    """
    plan = manager.generate(
        plan_prompt,
        system_prompt="You are a Senior Software Architect. Output only the execution steps inside <plan>...</plan> tags without drafting commentary.",
    )
    print(f"  Plan Output:\n{plan}\n")

    steps = extract_plan_steps(plan)
    if not steps:
        print("  Warning: No numbered steps found in plan. Falling back to direct task execution.")
        steps = [f"1. Implement solution for: {user_request}"]
    return steps


def execute_tool(tool_name: str, tool_data: dict) -> str:
    """Dispatches tool execution to the appropriate tool function."""
    if tool_name == "write_file":
        return write_file(tool_data.get("path", ""), tool_data.get("content", ""))
    elif tool_name == "read_file":
        return read_file(tool_data.get("path", ""))
    elif tool_name == "list_files":
        return list_files(tool_data.get("path", "."))
    return f"Unknown tool: {tool_name}"


def execute_step(manager: LoRAManager, step: str, context: str, max_retries: int = 3):
    """Executes a single step with coder role, retrying on JSON decode errors."""
    manager.set_role("coder")
    for attempt in range(max_retries):
        tool_prompt = f"Context:\n{context}\n\nTask: {step}\n{AVAILABLE_TOOLS_SCHEMA}"
        response = manager.generate(
            tool_prompt,
            system_prompt="You are a strict tool-calling engine. Output JSON only.",
        )

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                print(f"    Attempt {attempt + 1}: Failed to find JSON.")
                continue

            tool_data = json.loads(json_match.group(0))
            tool_name = tool_data.get("tool")
            result = execute_tool(tool_name, tool_data)

            print(f"    Tool used: {tool_name} on {tool_data.get('path', '')}")
            print(f"    Output: {result[:100]}...")

            new_context = context + f"\nStep '{step}' completed using tool '{tool_name}'. Result: {result}\n"
            return True, new_context

        except json.JSONDecodeError:
            print(f"    Attempt {attempt + 1}: Invalid JSON.")
        except Exception as e:
            print(f"    Error executing step: {e}")

    return False, context


def review_step(manager: LoRAManager, step: str, context: str):
    """Reviews the execution result using the reviewer role."""
    manager.set_role("reviewer")
    review_prompt = f"""
    QA Review task: "{step}"
    Output: {context}
    Did the execution achieve the task properly? Reply 'PASS' or 'FAIL'.
    """
    review = manager.generate(review_prompt, "You are a strict QA bot.")
    if "FAIL" in review.upper():
        print(f"    Reviewer evaluation: {review.strip()[:100]}")
    else:
        print("    Reviewer passed the step.")


def run_agent(user_request: str):
    print(f". Goal: {user_request}\n")
    manager = LoRAManager()

    steps = create_plan(manager, user_request)
    print(". Parsed Steps for Execution:")
    for s in steps:
        print(f"  - {s}")
    print()

    context = ""
    for step in steps:
        print(f". Executing: {step}")
        step_success, context = execute_step(manager, step, context)
        if not step_success:
            print("    Failed step.")
            continue

        review_step(manager, step, context)

    print("\n✨ Mission Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coding Multi-Agent CLI (TRL & PEFT)")
    parser.add_argument(
        "instruction",
        type=str,
        nargs="?",
        default="",
        help="The instruction/task for the agent to perform.",
    )
    args = parser.parse_args()

    if not args.instruction:
        # Fallback default task if none provided
        args.instruction = (
            "Write a Python function to solve the 'Longest Substring Without Repeating Characters' problem.\n"
            "Given a string s, find the length of the longest substring without repeating characters.\n"
            "Save the solution in 'longest_substring.py' and write a unit test file named 'tests/test_longest_substring.py' checking edge cases."
        )
        print("No instruction argument provided. Running default LeetCode task...")

    run_agent(args.instruction)
