import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.train import apply_lora, build_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints" / "lora_weights.pt"


def load_finetuned_model(checkpoint_path: pathlib.Path = CHECKPOINT_PATH):
    """Load the base model wrapped with LoRA and restore trained adapter weights."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16
    ).to(device)
    apply_lora(model)

    state_dict = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(state_dict, strict=False)
    # missing_keys is every frozen base-model weight, which never appears in an
    # adapter-only checkpoint -- only unexpected_keys indicates an actual mismatch.
    assert not result.unexpected_keys, (
        f"checkpoint has keys the model doesn't: {result.unexpected_keys}"
    )

    model.eval()
    return model, tokenizer


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
