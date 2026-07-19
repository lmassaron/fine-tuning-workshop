import unittest
import sys
import os

# Resolve imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestEnvironmentSanity(unittest.TestCase):
    def test_imports_and_torch(self):
        """Verify PyTorch version and CUDA availability."""
        import torch

        print(f"Torch Version: {torch.__version__}")
        self.assertTrue(
            torch.cuda.is_available(), "CUDA is not available on this device!"
        )

    def test_unsloth_import(self):
        """Verify that Unsloth libraries can be imported successfully."""
        print("Unsloth FastLanguageModel imported successfully!")

    def test_dataset_mapping(self):
        """Verify that Hugging Face datasets can be mapped successfully."""
        from datasets import Dataset

        ds = Dataset.from_dict({"question": ["Q1"], "response": ["R1"]})

        def format_data(examples):
            return {
                "text": [
                    f"<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
                    for q, r in zip(examples["question"], examples["response"])
                ]
            }

        ds = ds.map(format_data, batched=True)
        self.assertIn("text", ds.column_names)

    def test_sft_config(self):
        """Verify that SFTTrainer configuration can be instantiated."""
        from trl import SFTConfig

        config = SFTConfig(
            dataset_text_field="text", max_length=2048, eos_token="<|im_end|>"
        )
        self.assertEqual(config.dataset_text_field, "text")
        self.assertEqual(config.eos_token, "<|im_end|>")


if __name__ == "__main__":
    unittest.main()
