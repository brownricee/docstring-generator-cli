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


class Generator:
    """Wraps a loaded model and tokenizer to produce one docstring at a time."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, code: str) -> str:
        """Generate a raw docstring continuation for one function.

        Args:
            code: The function's source, dedented to column 0.

        Returns:
            str: The decoded continuation, or "" if generation ran past its
            token budget without stopping.
        """
        prompt = build_prompt(code)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        # skip_special_tokens erases EOS from the text, so "did it stop on its
        # own?" has to be asked of the token count. A run-on generation is a
        # half-written docstring; refuse it rather than insert a truncated one.
        if out.shape[1] - input_len >= MAX_NEW_TOKENS:
            logger.warning("generation hit the %d-token cap", MAX_NEW_TOKENS)
            return ""

        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def load_generator(adapter: pathlib.Path | None = None) -> Generator:
    """Load the fine-tuned model, downloading the adapter if it is not cached."""
    from docgen.model import load_model, resolve_adapter

    adapter_path = resolve_adapter(adapter)
    print("loading model (first run downloads ~3 GB from HuggingFace)...")
    return Generator(*load_model(adapter_path))
