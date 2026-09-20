import argparse
import ast
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.data_pipeline import is_google_style
from training.load_adapter import load_finetuned_model
from training.train import DATA_DIR, build_prompt, load_jsonl, trainable_fraction

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 200
# Fixed so re-running the script compares the same functions every time.
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=30, help="number of held-out test.jsonl examples to evaluate"
    )
    return parser.parse_args()


def sample_test_set(n: int) -> list[dict]:
    records = load_jsonl(DATA_DIR / "test.jsonl")
    return random.Random(SEED).sample(records, n)


def generate_all(model, tokenizer, samples: list[dict]) -> list[str]:
    outputs = []
    for i, rec in enumerate(samples):
        prompt = build_prompt(rec["code"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        )
        print(f"  generated {i + 1}/{len(samples)}")
    return outputs


def pass_rate(samples: list[dict], generations: list[str]) -> tuple[int, int]:
    # records hold exactly one top-level function each -- data_pipeline.py's own
    # filter already guarantees this, so tree.body[0] is always the FuncDef.
    passed = sum(
        is_google_style(gen, ast.parse(rec["code"]).body[0])
        for rec, gen in zip(samples, generations)
    )
    return passed, len(samples)


def main():
    args = parse_args()
    samples = sample_test_set(args.n)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"generating with base model ({len(samples)} samples)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()
    base_outputs = generate_all(base_model, tokenizer, samples)
    del base_model  # free before loading the second 1.5B model

    print(f"generating with fine-tuned model ({len(samples)} samples)...")
    ft_model, _ = load_finetuned_model()
    ft_outputs = generate_all(ft_model, tokenizer, samples)

    base_passed, n = pass_rate(samples, base_outputs)
    ft_passed, _ = pass_rate(samples, ft_outputs)
    trainable, total = trainable_fraction(ft_model)

    print()
    print(f"trainable params: {trainable}/{total} ({100 * trainable / total:.3f}%)")
    print(f"base model style pass-rate:       {base_passed}/{n} ({100 * base_passed / n:.1f}%)")
    print(f"fine-tuned model style pass-rate: {ft_passed}/{n} ({100 * ft_passed / n:.1f}%)")

    print("\n--- sample comparisons ---")
    for rec, base_out, ft_out in list(zip(samples, base_outputs, ft_outputs))[:5]:
        print("\ncode:")
        print(rec["code"][:300])
        print("reference docstring:")
        print(rec["docstring"])
        print("base output:")
        print(base_out)
        print("fine-tuned output:")
        print(ft_out)


if __name__ == "__main__":
    main()
