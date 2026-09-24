import inspect
import logging
import pathlib
import re

import torch

from docgen.astutils import FuncDef, has_section, raises_exception, sections_match
from docgen.prompt import build_prompt

logger = logging.getLogger(__name__)

# Same budget training/evaluate.py generates with.
MAX_NEW_TOKENS = 200

_FENCE = re.compile(r"^\s*```")

# Anchored at column 0 with no indent allowance on purpose: Google sections are
# indented, and a `>>> def f()` inside an Example: block is prefixed. These only
# fire when the model has genuinely stopped writing a docstring and resumed
# emitting top-level code -- which is what it does when it misses EOS.
_STOP = re.compile(r"^(?:```|@|def |async def |class |if __name__|Write a Google-style)")

_QUOTES = ('"""', "'''")


def _strip_quotes(text: str) -> str:
    """Remove a wrapping triple-quoted literal the model emitted anyway.

    The prompt says "no quotes", but the model was trained on code and reaches
    for them regardless.
    """
    text = text.strip()
    for quote in _QUOTES:
        if text.startswith(quote):
            text = text[len(quote):]
            end = text.find(quote)
            return (text[:end] if end != -1 else text).strip()
    for quote in _QUOTES:
        if text.endswith(quote):
            return text[: -len(quote)].strip()
    return text


def postprocess(raw: str) -> str:
    """Turn a raw model continuation into docstring text.

    Args:
        raw: The decoded text generated after the prompt.

    Returns:
        str: Cleaned docstring text, without quotes or a common indent.
    """
    lines = raw.split("\n")

    # Drop an opening code fence BEFORE the stop scan. If the scan ran first, a
    # leading ```python would match _STOP at index 0 and truncate to nothing.
    if lines and _FENCE.match(lines[0]):
        lines = lines[1:]

    kept = []
    for line in lines:
        if _STOP.match(line):
            break
        kept.append(line)

    # Quote stripping after the stop scan: when the model does close its
    # literal, that closing quote is the better truncation point.
    text = _strip_quotes("\n".join(kept))

    # Last, and the same normalization the training targets went through, so
    # the gate below judges text in the representation the model was taught.
    return inspect.cleandoc(text)


def is_usable(text: str, fn: FuncDef) -> bool:
    """True if text is safe to insert and consistent with fn's signature.

    Gates on sections_match rather than is_google_style: requiring at least one
    Args/Returns/Yields section is a training-corpus rule, and a real no-argument
    void function could never satisfy it.
    """
    if not text:
        return False
    # The inserter cannot quote these; reject here so it is reported as a skip
    # with a reason rather than silently dropped later.
    if any(quote in text for quote in _QUOTES) or text.endswith("\\"):
        return False

    # Raises is checked one way only: documenting an exception a function has
    # no raise statement for is invented, but omitting Raises from a function
    # that does raise is a normal, common docstring. Observed failure this
    # catches: an __init__ with no arguments, no return and no raise -- which
    # the corpus filter excluded, so the model never saw one -- came back with
    # a confident "Raises: ValueError: if the total is not a positive integer."
    if has_section(text, "Raises") and not raises_exception(fn):
        return False

    return sections_match(text, fn)


def _argsort_by_length(lengths: list[int]) -> list[int]:
    """Return indices that sort lengths ascending.

    order[i] is the original index of the i-th shortest item -- the same
    convention argsort uses.
    """
    return sorted(range(len(lengths)), key=lambda i: lengths[i])


def _unsort(sorted_items: list, order: list[int]) -> list:
    """Undo _argsort_by_length: place each sorted item back at its original index."""
    items = [None] * len(order)
    for pos, orig_idx in enumerate(order):
        items[orig_idx] = sorted_items[pos]
    return items


class Generator:
    """Wraps a loaded model and tokenizer to produce docstrings in a batch."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Left padding is required for batched generation on a decoder-only
        # model: generation continues from the right edge of each row, so
        # right-padding would insert padding tokens between prompt and
        # continuation instead of before the prompt.
        self.tokenizer.padding_side = "left"

    def generate_batch(self, codes: list[str]) -> list[str]:
        """Generate raw docstring continuations for a batch of functions.

        One model.generate() call for the whole batch, instead of one per
        function -- the sequential version left the CPU/GPU underused between
        calls.

        Args:
            codes: Each function's source, dedented to column 0.

        Returns:
            list[str]: One decoded continuation per input, in the same order.
            An entry is "" if that generation ran past its token budget
            without stopping.
        """
        prompts = [build_prompt(code) for code in codes]

        # Tokenize once, unpadded, to get exact per-item lengths, then sort so
        # the batch pads to a much smaller max length than a random ordering
        # would. Unsorted back to input order below -- callers never see this.
        encoded = self.tokenizer(prompts, padding=False)
        order = _argsort_by_length([len(ids) for ids in encoded["input_ids"]])
        sorted_encoded = {k: [v[i] for i in order] for k, v in encoded.items()}

        inputs = self.tokenizer.pad(sorted_encoded, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        eos_id = self.tokenizer.eos_token_id

        results_sorted = []
        for row in out[:, input_len:]:
            # pad_token_id == eos_token_id (set in model.py), so a finished
            # row's trailing padding is also eos -- the first eos in a row is
            # always its real stop, and skip_special_tokens strips all of it.
            # A row with no eos anywhere never stopped on its own: it ran the
            # full budget and is a half-written docstring: refuse it rather
            # than insert a truncated one.
            if not bool((row == eos_id).any()):
                logger.warning("generation hit the %d-token cap", MAX_NEW_TOKENS)
                results_sorted.append("")
                continue
            results_sorted.append(self.tokenizer.decode(row, skip_special_tokens=True))

        return _unsort(results_sorted, order)


def load_generator(adapter: pathlib.Path | None = None) -> Generator:
    """Load the fine-tuned model, downloading the adapter if it is not cached."""
    from docgen.model import load_model, resolve_adapter

    adapter_path = resolve_adapter(adapter)
    print("loading model (first run downloads ~3 GB from HuggingFace)...")
    return Generator(*load_model(adapter_path))
