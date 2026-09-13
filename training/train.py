import json
import pathlib
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.lora import LoRALinear

device = "cuda" if torch.cuda.is_available() else "cpu"

# Every one of Qwen's layers contains these four nn.Linear modules -- they are
# the attention part of the layer. Wrapping only these, and leaving the much
# bigger feed-forward layers beside them alone, is what keeps the trainable
# count near 0.1-1% of the model.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def apply_lora(model, r: int = 16, alpha: int = 32) -> int:
    """Replace every targeted nn.Linear with a LoRALinear wrapping it.

    Returns how many were replaced. Check it is not zero: wrapping nothing
    trains nothing, and that looks exactly like a bad learning rate.
    """
    to_replace = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if child_name in TARGET_MODULES:
                to_replace.append((parent, child_name, child))

    for parent, child_name, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, r, alpha))

    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    return len(to_replace)


def trainable_fraction(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts.

    This is the "% of parameters trained" number for the resume. Expect
    0.1-1% on Qwen2.5-Coder-1.5B; several percent means step 3 above missed
    some weights.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_prompt(code: str) -> str:
    """Turn docstring-stripped code into the exact text fed to the model.

    Week 5's CLI has to build its prompt the same way, character for
    character. If the two differ, the model meets a format at inference that
    it never saw in training. Define it once here and import it there.
    """
    return (
        "Write a Google-style docstring for the following Python function. "
        "Respond with only the docstring text -- no code, no quotes.\n\n"
        f"{code}\n\nDocstring:\n"
    )


def collate(batch, tokenizer):
    """Tokenize a batch, and hide the prompt from the loss.

    Each example is prompt + docstring. Score the whole thing and most of the
    loss comes from re-typing the function the model was just handed. So set
    every prompt position in `labels` to -100, a value both
    nn.CrossEntropyLoss and HuggingFace skip when averaging.

    How: tokenize the prompt alone to get its token count n, tokenize
    prompt + docstring for input_ids, copy those to labels, set labels[:n] =
    -100.
    """
    prompts = [build_prompt(ex["code"]) for ex in batch]
    texts = [prompt + ex["docstring"] for prompt, ex in zip(prompts, batch)]

    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()

    for i, prompt in enumerate(prompts):
        n = len(tokenizer(prompt, truncation=True, max_length=1024)["input_ids"])
        labels[i, :n] = -100
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_epoch(model, loader, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(training):
            loss = model(**batch).loss
        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16
    ).to(device)

    n = apply_lora(model)
    assert n > 0
    print(trainable_fraction(model))

    train_data = load_jsonl(DATA_DIR / "train.jsonl")
    val_data = load_jsonl(DATA_DIR / "val.jsonl")

    collate_fn = partial(collate, tokenizer=tokenizer)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=0.01,
    )

    if "--sanity" in sys.argv:
        sanity_loader = DataLoader(train_data[:4], batch_size=4, collate_fn=collate_fn)
        for step in range(300):
            loss = run_epoch(model, sanity_loader, optimizer)
            if step % 50 == 0:
                print(f"sanity step {step}: loss {loss:.4f}")
        return

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=8, collate_fn=collate_fn)

    epochs = 3
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader)
        print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")

    lora_state = {
        k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k
    }
    out_path = DATA_DIR.parent / "lora_weights.pt"
    torch.save(lora_state, out_path)
    print(f"saved adapter weights to {out_path}")


if __name__ == "__main__":
    main()
