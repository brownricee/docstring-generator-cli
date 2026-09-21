import logging
import re

import libcst as cst
from libcst.metadata import PositionProvider

logger = logging.getLogger(__name__)

_LEADING_WS = re.compile(r"[ \t]*")


def _literal(text: str, indent: str, newline: str) -> str:
    """Render docstring text as a triple-quoted literal for a block at indent.

    libcst emits a SimpleStatementLine's own indent, but the string token
    itself is written out verbatim -- it has no idea there are lines inside it.
    So the first line is indented for us and every line after it, plus the
    closing quotes, has to carry the indent explicitly.
    """
    # A backslash in the text is meant literally (regex, LaTeX, escape docs).
    # Without the r prefix it would be interpreted at insertion time and the
    # docstring would no longer say what the model wrote.
    prefix = "r" if "\\" in text else ""

    # `"""ends in a quote""""` is a SyntaxError. Switching the quote character
    # is the fix; escaping is not available to us, since the literal may be raw.
    # Text containing ''' is refused upstream, so this is always safe.
    quote = "'''" if text.endswith('"') else '"""'

    lines = text.split("\n")
    if len(lines) == 1:
        return f"{prefix}{quote}{text}{quote}"

    body = lines[0]
    for line in lines[1:]:
        # Blank lines stay genuinely blank. Padding them to the indent would
        # ship trailing whitespace (W293) into someone else's linted repo.
        body += newline + (indent + line if line.strip() else "")
    return f"{prefix}{quote}{body}{newline}{indent}{quote}"


class _Inserter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, docstrings: dict[int, str], lines: list[str], newline: str):
        self.docstrings = docstrings
        self.lines = lines
        self.newline = newline
        self.inserted: set[int] = set()

    def leave_FunctionDef(self, original_node, updated_node):
        # Take the line from the name, not the FunctionDef: libcst owns
        # decorators as a field of the function, so the node's range can begin
        # at the `@`, while scan_source's lineno is always the `def` line. The
        # name token is on the def line under every reading.
        line = self.get_metadata(PositionProvider, original_node.name).start.line
        text = self.docstrings.get(line)
        if text is None:
            return updated_node

        # Both quote styles are refused: _literal picks between them, so it
        # needs one of the two to be absent. A trailing backslash is refused
        # because it cannot end a raw string and we cannot escape inside one.
        if '"""' in text or "'''" in text or text.endswith("\\"):
            logger.warning("line %d: docstring is not safely quotable, skipping", line)
            return updated_node

        # `def f(): return 1` is a SimpleStatementSuite, not an IndentedBlock.
        # Inserting would mean reformatting the user's line, so leave it.
        if not isinstance(original_node.body, cst.IndentedBlock):
            logger.warning("line %d: one-line body, skipping", line)
            return updated_node

        # Read the indent off the source line of the block's existing first
        # statement rather than computing it from the column number: a column
        # is a character count, so a tab-indented file would yield one space.
        first_line = self.get_metadata(
            PositionProvider, original_node.body.body[0]
        ).start.line
        indent = _LEADING_WS.match(self.lines[first_line - 1]).group(0)
        if not indent:
            logger.warning("line %d: could not determine indent, skipping", line)
            return updated_node

        try:
            node = cst.SimpleStatementLine(
                [cst.Expr(cst.SimpleString(_literal(text, indent, self.newline)))]
            )
        except cst.CSTValidationError:
            logger.warning("line %d: could not build a string literal, skipping", line)
            return updated_node

        self.inserted.add(line)
        # Build on updated_node.body.body, not the original: leave_* is
        # post-order, so a nested function's insertion is already in there.
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=[node, *updated_node.body.body])
        )


def insert_docstrings(source: str, docstrings: dict[int, str]) -> tuple[str, set[int]]:
    """Insert docstrings into source, keyed by each function's `def` line.

    Values are plain text without quotes or indentation -- the same form the
    training targets are stored in. This function owns the quoting and the
    re-indentation.

    Every insertion for a file happens in one pass. That is what makes the line
    numbers safe to use as keys: libcst computes positions once over the
    original tree and materializes the new source only at the end, so nothing
    shifts underneath us mid-edit.

    Args:
        source: The file's text.
        docstrings: Maps a `def` line number to the docstring text to insert.

    Returns:
        tuple: The new source, and the set of lines actually inserted at.
    """
    if not docstrings:
        return source, set()

    module = cst.parse_module(source)
    transformer = _Inserter(
        docstrings, source.splitlines(), module.default_newline
    )
    new_module = cst.MetadataWrapper(module, unsafe_skip_copy=True).visit(transformer)
    return new_module.code, transformer.inserted
