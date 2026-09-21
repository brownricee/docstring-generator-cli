import ast
import logging
import textwrap
from pathlib import Path

from docgen.astutils import FuncDef, scan_source

logger = logging.getLogger(__name__)

# is_reasonable_length's own bound, reused deliberately: the model provably
# never saw a function longer than this, so prompting with one is out of
# distribution (and slow).
MAX_CODE_CHARS = 2000


class Unsupported(Exception):
    """A function the tool cannot handle. The message is the reason to report."""


def iter_py_files(path: Path) -> list[Path]:
    """Return the .py files under path, or path itself if it is a file."""
    return [path] if path.is_file() else sorted(path.rglob("*.py"))


def read_source(path: Path) -> str:
    """Read a file without translating its line endings.

    newline="" keeps CRLF as CRLF. Path.read_text would collapse it to \\n and
    the matching write would then rewrite every line ending in the file, turning
    a three-line docstring insertion into a whole-file diff.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write_source(path: Path, source: str) -> None:
    """Write a file without translating its line endings (see read_source)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(source)


def scan_file(path: Path) -> tuple[str, list[dict], int] | None:
    """Read and scan one file.

    Args:
        path: The .py file to read.

    Returns:
        tuple: The source text, the records for functions with no docstring,
        and the total number of functions found. None if the file could not be
        read or parsed.
    """
    try:
        source = read_source(path)
    except (UnicodeDecodeError, OSError):
        logger.debug("unreadable file: %s", path)
        return None

    try:
        records = scan_source(source, str(path))
    except SyntaxError:
        logger.debug("unparseable file: %s", path)
        return None

    return source, [r for r in records if not r["has_docstring"]], len(records)


def function_source(source: str, record: dict) -> str:
    """Return the function's own source text, dedented to column 0.

    Slices from the `def` line, so decorators are excluded -- matching
    CodeSearchNet's func_code_string, which is what the model was trained on.
    """
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[record["lineno"] - 1 : record["end_lineno"]]))


def prepare(source: str, record: dict) -> tuple[str, FuncDef]:
    """Return (code, parsed function) ready to prompt with.

    Raises:
        Unsupported: If this function should be skipped, with the reason.
    """
    code = function_source(source, record)

    if len(code) >= MAX_CODE_CHARS:
        raise Unsupported("function too long")

    try:
        fn = ast.parse(code).body[0]
    except SyntaxError:
        # IndentationError subclasses SyntaxError, and is what dedent leaves
        # behind when a nested function's snippet has no common indent to strip
        # (e.g. it contains a multi-line string starting at column 0).
        raise Unsupported("could not parse in isolation") from None

    if not isinstance(fn, FuncDef):
        raise Unsupported("not a function definition")

    # `def f(): return 1` -- the body shares the def's line, so there is no
    # line of its own to insert a docstring into without reformatting the code.
    if fn.body[0].lineno == fn.lineno:
        raise Unsupported("one-line body")

    return code, fn
