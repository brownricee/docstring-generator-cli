import json
import os
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

BATCH_SIZE = 2
ACCUM_STEPS = 4

MAX_LENGTH = 768

SAVE_EVERY = 500

# Set CHECKPOINT_DIR (e.g. to a mounted Google Drive folder) so checkpoints
# survive a Colab runtime dying -- the local disk does not.
CHECKPOINT_DIR = pathlib.Path(os.environ.get("CHECKPOINT_DIR", DATA_DIR.parent))


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
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()

    for i, prompt in enumerate(prompts):
        n = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
        # If truncation cut a row at or before the end of its prompt, masking the
        # whole prompt would leave the row with zero supervised tokens -- and
        # cross-entropy over zero elements is nan, which the next backward()
        # writes into every LoRA weight. Always leave one token scored.
        n = min(n, input_ids.shape[1] - 1)
        labels[i, :n] = -100
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def lora_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def save_adapter(model, path: pathlib.Path) -> None:
    torch.save(lora_state_dict(model), path)
    print(f"saved adapter weights to {path}")


def save_train_state(model, optimizer, epoch: int, path: pathlib.Path) -> None:
    torch.save(
        {
            "epoch": epoch,
            "lora_state": lora_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    print(f"saved training checkpoint (epoch {epoch}) to {path}")


def load_train_state(model, optimizer, path: pathlib.Path) -> int:
    """Load a checkpoint saved by save_train_state and return the epoch to resume at."""
    state = torch.load(path, map_location=device)
    result = model.load_state_dict(state["lora_state"], strict=False)
    assert not result.unexpected_keys, (
        f"checkpoint has lora keys the model doesn't: {result.unexpected_keys}"
    )
    # optimizer.load_state_dict matches saved state to param_groups by position, not
    # name -- only correct because apply_lora walks model.modules() deterministically
    # and the LoRA config is identical between the run that saved this and this one.
    optimizer.load_state_dict(state["optimizer_state"])
    epoch = state["epoch"]
    print(f"resumed from {path}: epoch {epoch} complete, optimizer state restored")
    return epoch + 1


def assert_adapters_moved(model) -> None:
    """Fail loudly if one optimizer step left every adapter untouched.

    lora_B starts as zeros, so any non-zero weight in it proves gradients
    reached the adapters and the optimizer applied them. Checking .grad instead
    would not work here -- zero_grad() has already set it back to None.

    This catches the quiet failure mode of gradient checkpointing: if no input
    to a checkpointed block requires grad, the recomputed block has no graph,
    every LoRA grad is None, and the loss prints happily while nothing trains.
    """
    moved = sum(
        1
        for name, p in model.named_parameters()
        if "lora_B" in name and p.count_nonzero() > 0
    )
    assert moved > 0, (
        "no lora_B weight moved off its zero init after an optimizer step -- "
        "gradients are not reaching the adapters"
    )
    print(f"gradient check: {moved} lora_B matrices updated")


def run_epoch(
    model, loader, optimizer=None, checkpoint_path=None, accum_steps=ACCUM_STEPS
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(training):
            loss = model(**batch).loss
        if training:
            # Accumulated gradients sum, so scale down to keep this equivalent
            # to one backward over a batch of BATCH_SIZE * accum_steps. Without
            # the division the effective learning rate is accum_steps times too
            # high. Report the unscaled loss so the numbers stay comparable.
            (loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                opt_step = (step + 1) // accum_steps
                if checkpoint_path is not None and opt_step % SAVE_EVERY == 0:
                    save_adapter(model, checkpoint_path)
        total_loss += loss.item()
        if step % 20 == 0:
            print(f"  step {step}/{len(loader)}: loss {loss.item():.4f}")

    # len(loader) is not divisible by accum_steps, so the last few micro-batches
    # leave gradients sitting in .grad with no step behind them.
    if training and len(loader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B", dtype=torch.bfloat16
    ).to(device)

    n = apply_lora(model)
    assert n > 0
    print(trainable_fraction(model))

    # A LoRALinear sits inside all 28 attention blocks, so autograd would
    # otherwise retain the forward activations of the entire network. Recompute
    # them in the backward pass instead: ~30% slower, several GB cheaper.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    # Print the memory-relevant settings up front. A stale checkout otherwise
    # looks exactly like a fresh one until it OOMs several minutes later.
    print(
        f"config: batch {BATCH_SIZE} x accum {ACCUM_STEPS} "
        f"(effective {BATCH_SIZE * ACCUM_STEPS}), max_length {MAX_LENGTH}, "
        f"grad checkpointing {model.is_gradient_checkpointing}"
    )

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
            # accum_steps=1: this loop is one batch, and stepping every batch
            # keeps it at the full lr rather than a quarter of it.
            loss = run_epoch(model, sanity_loader, optimizer, accum_steps=1)
            if step == 0:
                assert_adapters_moved(model)
            if step % 50 == 0:
                print(f"sanity step {step}: loss {loss:.4f}")
        return

    checkpoint_path = CHECKPOINT_DIR / "lora_weights.pt"
    train_state_path = CHECKPOINT_DIR / "train_state.pt"
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    start_epoch = 0
    if train_state_path.exists():
        start_epoch = load_train_state(model, optimizer, train_state_path)

    epochs = 3
    if start_epoch >= epochs:
        print(f"train_state.pt already at epoch {start_epoch - 1}; nothing to resume ({epochs=})")
    else:
        for epoch in range(start_epoch, epochs):
            train_loss = run_epoch(model, train_loader, optimizer, checkpoint_path)
            val_loss = run_epoch(model, val_loader)
            print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")
            save_adapter(model, checkpoint_path)
            save_train_state(model, optimizer, epoch, train_state_path)


if __name__ == "__main__":
    main()
