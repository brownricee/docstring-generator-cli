import pathlib

import torch

from docgen.model import load_model
from docgen.prompt import build_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints" / "lora_weights.pt"


def load_finetuned_model(checkpoint_path: pathlib.Path = CHECKPOINT_PATH):
    """Load the base model wrapped with LoRA and restore trained adapter weights.

    The loading itself lives in docgen/model.py, which the CLI also uses. This
    keeps the repo-checkout default path (checkpoints/, where the README says
    to download the release asset) for the training-side scripts, which do not
    want the CLI's user-level cache.
    """
    return load_model(checkpoint_path)


def main():
    model, tokenizer = load_finetuned_model()
    print(f"loaded adapter from {CHECKPOINT_PATH}")

    sample_code = "def add(a, b):\n    return a + b"
    prompt = build_prompt(sample_code)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(generated)


if __name__ == "__main__":
    main()
