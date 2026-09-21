import ast
import logging
import re
from pathlib import Path

# Silent unless the caller configures a handler, so a 412k-example pipeline run
# stays quiet, but `logging.basicConfig(level=logging.DEBUG)` in a REPL turns
# every drop reason back on when you are debugging a bad yield.
logger = logging.getLogger(__name__)

# `async def` parses to a different node type than `def`. Naming the union once
# here means you can't forget the async half in any of the signatures below.
FuncDef = ast.FunctionDef | ast.AsyncFunctionDef

def find_docstring_node(func: FuncDef) -> ast.Expr | None:
    """Return the ast.Expr node holding func's docstring, or None.

    The shared primitive. Everything else calls this.
    Apply the rule from the notes: body[0] is an Expr whose .value is a
    Constant whose .value is a str.
    """

    if not func.body:
        return None

    first = func.body[0]

    # Is the first statement a bare expression?
    if not isinstance(first, ast.Expr):
        return None

    # Is that expression a literal?
    if not isinstance(first.value, ast.Constant):
        return None

    # Is the literal a string?
    if not isinstance(first.value.value, str):
        return None

    return first

def strip_docstring(source: str) -> str | None:
    """Return source with its function's docstring removed, or None if unusable.

    Used to build clean training pairs: the model must not see the docstring
    it is being asked to write.

    Return None (caller drops the example) when:
      - source does not parse
      - top-level node is not a function def
      - there is no docstring
      - the docstring is the entire body (nothing left to describe)
    """

    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("unparseable source (probably Python 2)")
        return None

    if not tree.body or not isinstance(tree.body[0], FuncDef):
        return None

    fn = tree.body[0]

    doc_node = find_docstring_node(fn)

    # Must come before any doc_node.<attr> access below -- without it, the
    # first two-statement function with no docstring raises AttributeError.
    if doc_node is None:
        logger.debug("no docstring")
        return None

    # Docstring-only body: strip it and `def f():` has no body left.
    # Ordered after the None check so it reads as what it means -- the body is
    # nothing but the docstring -- rather than "any one-statement function".
    if len(fn.body) == 1:
        logger.debug("docstring is the entire body")
        return None

    # Line surgery is only safe if the docstring occupies whole lines of its
    # own. Two ways it can share a line with real code, both of which would
    # make the slice below delete something it must not.
    if doc_node.lineno == fn.lineno or fn.body[1].lineno <= doc_node.end_lineno:
        logger.debug("docstring shares a line with code")
        return None

    # Whole-line delete. NOT ast.unparse -- that would throw away every comment
    # in the function and reformat what is left, and this text is training data.
    lines = source.splitlines(keepends=True)
    return "".join(lines[: doc_node.lineno - 1] + lines[doc_node.end_lineno :])


def has_params(func: FuncDef) -> bool:
    """True if func takes at least one non-self/cls parameter."""
    args_list = []
    func_args = func.args

    # Positionals kept separate so the receiver check below can look at "first
    # positional parameter" specifically - position in the merged list is not
    # the same thing.
    positional = [*func_args.posonlyargs, *func_args.args]

    # `self`/`cls` is the receiver only as the FIRST POSITIONAL parameter.
    if positional and positional[0].arg in ("self", "cls"):
        positional.pop(0)

    args_list.extend(positional)
    args_list.extend(func_args.kwonlyargs)

    if func_args.vararg is not None:
        args_list.append(func_args.vararg)

    if func_args.kwarg is not None:
        args_list.append(func_args.kwarg)

    return len(args_list) > 0


def _iter_own_nodes(func: FuncDef):
    """Yield every node inside func's own body, not descending into nested
    scopes.

    Shared by returns_value and is_generator - both need the same "this
    function's statements, but not an inner function's statements" walk, and
    getting it wrong is the subtle failure in both.
    """
    # The API you need is ast.iter_child_nodes(node) -- direct children only,
    # one level down. Build the recursion yourself so you control where it
    # stops.
    scope_openers = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    stack = [node for node in func.body if not isinstance(node, scope_openers)]

    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, scope_openers):
                stack.append(child)


def returns_value(func: FuncDef) -> bool:
    """True if func returns something worth a "Returns:" section.

    Bare `return` and explicit `return None` both count as void.
    """
    # An explicit `return None` does NOT count. Such a function is
    # void, and a "Returns:" section describing None is noise the model would
    # learn to imitate. Note this is per-Return, not per-function: a function
    # with `return None` on one branch and `return x` on another still has a
    # real return value, and the loop below reports True on the second one.
    #
    # Written as an explicit loop rather than any(), because each of the three
    # skips means something different and deserves its own line.
    for node in _iter_own_nodes(func):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            continue  # bare `return`
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            continue  # explicit `return None`
        return True
    return False


def is_generator(func: FuncDef) -> bool:
    """True if func contains a yield / yield from (same nesting caveat)."""
    # No edge cases here, so any() reads better than a loop. Both node types
    # are needed: `yield x` and `yield from xs` do not parse to the same node.
    return any(
        isinstance(node, (ast.Yield, ast.YieldFrom))
        for node in _iter_own_nodes(func)
    )


# --- Google-style section checks ---------------------------------------------

# `\r` is in the TRAILING class only. Under re.M, `$` matches just before a
# `\n`, so in a CRLF docstring the `\r` sits between "Args:" and that position
# and has to be consumable -- without it, "Args:\r\n" reads as not-a-header and
# ~408 real training pairs get silently dropped. The leading class needs no
# `\r`, since in CRLF the `\r` ends the previous line rather than starting this
# one.
#
# "Raises" is in the table but is NOT part of is_google_style -- adding it
# there would change the corpus filter. It exists for the CLI's inference-time
# check; see sections_match's note about one-directional claims.
_SECTIONS = {n: re.compile(rf"^[ \t]*{n}:[ \t\r]*$", re.M)
             for n in ("Args", "Returns", "Yields", "Raises")}


def has_section(docstring: str, section_name: str) -> bool:
    # True if docstring contains a Google section header on a line of its own.
    return _SECTIONS[section_name].search(docstring) is not None


def raises_exception(func: FuncDef) -> bool:
    """True if func contains a raise statement in its own scope."""
    return any(isinstance(node, ast.Raise) for node in _iter_own_nodes(func))


def sections_match(docstring: str, fn: FuncDef) -> bool:
    """True if Args/Returns/Yields is present exactly when the signature calls for it.

    Equality in both directions: a missing section is undocumented, and a
    section the signature does not support is wrong documentation.

    This is the check the CLI gates generated docstrings on. It deliberately
    does NOT require a section to be present at all -- see is_google_style --
    and it says nothing about Raises, which is one-directional and belongs to
    the CLI alone (see docgen.generator.is_usable).
    """
    return (
        has_section(docstring, "Args") == has_params(fn)
        and has_section(docstring, "Returns") == returns_value(fn)
        and has_section(docstring, "Yields") == is_generator(fn)
    )


def is_google_style(docstring: str, fn: FuncDef) -> bool:
    """The training-data filter: sections_match, plus at least one section.

    The extra clause is a corpus-purity requirement -- it keeps the training
    set teaching sections at all, rather than filling up with bare one-line
    summaries. A real no-parameter, void, non-generator function can never
    satisfy it, which is why inference gates on sections_match alone.

    Keep this defined in terms of sections_match so the two cannot drift.
    """
    return sections_match(docstring, fn) and any(
        has_section(docstring, n) for n in ("Args", "Returns", "Yields")
    )


# --- scanner half (the Week-5 CLI's core, useful now for eval targets) ---

def scan_source(source: str, path: str) -> list[dict]:
    """Return one record per function found: name, qualified name, line span,
    and whether it already has a docstring.

    Use ast.NodeVisitor here (not ast.walk) so you can track the enclosing
    class and report 'MyClass.my_method'.
    """

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.records = []

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            self.records.append(
                {
                    "name": node.name,
                    "qualname": ".".join(self.scope + [node.name]),
                    # lineno is the `def` line, NOT the first decorator -- so
                    # slicing [lineno-1:end_lineno] yields the function without
                    # its decorators, matching CodeSearchNet's func_code_string.
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "has_docstring": ast.get_docstring(node) is not None,
                    "path": path,
                }
            )
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    tree = ast.parse(source)
    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.records


def scan_path(root: Path) -> list[dict]:
    """Walk root for *.py and scan each. Skip files that fail to parse."""
    records = []
    for path in root.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.debug("skipping undecodable file: %s", path)
            continue
        try:
            records.extend(scan_source(source, str(path)))
        except SyntaxError:
            logger.debug("skipping unparseable file: %s", path)
            continue
    return records


if __name__ == "__main__":
    for record in scan_path(Path(__file__).parent):
        print(
            f"{record['path']}:{record['lineno']} {record['qualname']} "
            f"has_docstring={record['has_docstring']}"
        )
