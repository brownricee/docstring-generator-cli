import inspect
import logging
import os
import pathlib
import re
import shutil
import urllib.request

from docgen.astutils import FuncDef, has_section, raises_exception, sections_match
from docgen.prompt import build_prompt

logger = logging.getLogger(__name__)

# Same budget training/evaluate.py generates with.
MAX_NEW_TOKENS = 200

# llama-cpp-python defaults to 512, which a function near scanner.MAX_CODE_CHARS
# (2000 chars) plus the prompt plus MAX_NEW_TOKENS can overflow.
N_CTX = 2048

MODEL_URL = (
    "https://github.com/brownricee/docstring-generator-cli/releases/download/"
    "gguf-v1/docgen-q8_0.gguf"
)
CACHE_DIR = pathlib.Path(
    os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
) / "docgen"

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
    """Wraps a loaded llama_cpp.Llama to produce docstrings for a batch."""

    def __init__(self, llm):
        self.llm = llm

    def generate_batch(self, codes: list[str]) -> list[str]:
        """Generate raw docstring continuations for a batch of functions.

        llama-cpp-python's completion API runs one sequence at a time, so this
        loops; the batch interface stays so cli.py's flush logic is unchanged.

        Args:
            codes: Each function's source, dedented to column 0.

        Returns:
            list[str]: One decoded continuation per input, in the same order.
            An entry is "" if that generation ran past its token budget
            without stopping.
        """
        results = []
        for code in codes:
            # temperature=0 is greedy, matching the do_sample=False the model
            # was evaluated with. repeat_penalty=1.0 is required: llama-cpp's
            # default of 1.1 is not greedy decoding and changes the output.
            out = self.llm(
                build_prompt(code),
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                repeat_penalty=1.0,
            )
            choice = out["choices"][0]
            # "length" means it never emitted eos: a half-written docstring.
            # Refuse it rather than insert a truncated one.
            if choice["finish_reason"] == "length":
                logger.warning("generation hit the %d-token cap", MAX_NEW_TOKENS)
                results.append("")
                continue
            results.append(choice["text"])
        return results


def resolve_model(override: pathlib.Path | None = None) -> pathlib.Path:
    """Return a local path to the GGUF model, downloading it on first use."""
    if override is not None:
        return override

    dest = CACHE_DIR / MODEL_URL.rsplit("/", 1)[1]
    if dest.exists():
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading model (1.6 GB, first run only) -> {dest}")

    # Download to .part and rename. os.replace is atomic, so a Ctrl-C partway
    # through leaves no half-written file that later runs would trust as cached.
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp, dest)

    return dest


def load_generator(model_path: pathlib.Path | None = None) -> Generator:
    """Load the fine-tuned GGUF model, downloading it if it is not cached."""
    from llama_cpp import Llama

    path = resolve_model(model_path)
    print("loading model...")
    return Generator(Llama(model_path=str(path), n_ctx=N_CTX, verbose=False))
