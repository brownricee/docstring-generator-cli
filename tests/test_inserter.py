import ast

import pytest

from docgen.inserter import insert_docstrings


def docstring_at(source: str, lineno: int) -> str | None:
    """Return the docstring ast reads back for the function defined at lineno."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno == lineno:
                return ast.get_docstring(node, clean=True)
    raise AssertionError(f"no function at line {lineno}")


def check(source: str, docstrings: dict[int, str]) -> str:
    """Insert, assert every requested line was applied and the result parses.

    The text itself is re-indented on the way in, so it is deliberately NOT
    compared verbatim here -- each test asserts the round-trip through
    ast.get_docstring instead, which is the property that actually matters.
    """
    new, inserted = insert_docstrings(source, docstrings)
    assert inserted == set(docstrings), "not every requested line was inserted"
    ast.parse(new)  # must be valid Python
    return new


def test_empty_is_byte_identical_noop():
    source = "def f(a):\n    return a\n"
    assert insert_docstrings(source, {}) == (source, set())


def test_module_level_multiline():
    source = "def add(a, b):\n    return a + b\n"
    text = "Add two numbers.\n\nArgs:\n    a: First.\n    b: Second.\n\nReturns:\n    int: Sum."
    new = check(source, {1: text})
    assert new == (
        'def add(a, b):\n'
        '    """Add two numbers.\n'
        '\n'
        '    Args:\n'
        '        a: First.\n'
        '        b: Second.\n'
        '\n'
        '    Returns:\n'
        '        int: Sum.\n'
        '    """\n'
        '    return a + b\n'
    )
    assert docstring_at(new, 1) == text


def test_method_indent_is_four_not_zero():
    source = "class A:\n    def m(self):\n        return 1\n"
    text = "Do it.\n\nReturns:\n    int: One."
    new = check(source, {2: text})
    assert '        """Do it.' in new
    assert "\n        Returns:\n" in new
    assert new.endswith('        """\n        return 1\n')
    assert docstring_at(new, 2) == text


def test_nested_function_indent_is_eight():
    source = "def outer():\n    def inner(x):\n        return x\n    return inner\n"
    text = "Inner.\n\nArgs:\n    x: Thing."
    new = check(source, {2: text})
    assert '        """Inner.' in new
    assert "\n            x: Thing.\n" in new
    assert docstring_at(new, 2) == text


def test_tab_indented_file_gets_tab_indent():
    source = "def f(a):\n\treturn a\n"
    text = "Sum.\n\nArgs:\n    a: Thing."
    new = check(source, {1: text})
    assert '\t"""Sum.' in new
    assert "\n\t\"\"\"\n" in new
    assert "    \"\"\"\n" not in new  # not space-indented
    assert docstring_at(new, 1) == text


def test_decorated_function_matched_by_def_line():
    source = "class A:\n    @property\n    def v(self):\n        return self._v\n"
    text = "The value.\n\nReturns:\n    int: It."
    new = check(source, {3: text})
    assert "@property\n    def v(self):\n" in new  # decorator untouched
    assert docstring_at(new, 3) == text


def test_same_qualname_pair_only_targets_requested_line():
    source = (
        "class A:\n"
        "    @property\n"
        "    def v(self):\n"
        "        return self._v\n"
        "\n"
        "    @v.setter\n"
        "    def v(self, value):\n"
        "        self._v = value\n"
    )
    text = "Set the value.\n\nArgs:\n    value: New value."
    new = check(source, {7: text})
    # Only the setter gained a docstring; the getter is untouched.
    assert docstring_at(new, 3) is None
    assert docstring_at(new, 7) == text


def test_async_def():
    source = "async def fetch(url):\n    return await get(url)\n"
    text = "Fetch it.\n\nArgs:\n    url: Where.\n\nReturns:\n    bytes: Body."
    new = check(source, {1: text})
    assert docstring_at(new, 1) == text


def test_three_functions_in_one_pass():
    source = "def a(x):\n    return x\n\n\ndef b(y):\n    return y\n\n\ndef c(z):\n    return z\n"
    new = check(
        source,
        {
            1: "A.\n\nArgs:\n    x: X.",
            5: "B.\n\nArgs:\n    y: Y.",
            9: "C.\n\nArgs:\n    z: Z.",
        },
    )
    assert new.count('"""') == 6
    # Later functions landed correctly despite earlier insertions shifting each
    # one down by the 5 lines of the docstring inserted above it.
    assert docstring_at(new, 1).startswith("A.")
    assert docstring_at(new, 10).startswith("B.")
    assert docstring_at(new, 19).startswith("C.")


def test_one_line_body_is_left_alone():
    source = "def f(): return 1\n"
    new, inserted = insert_docstrings(source, {1: "Return one."})
    assert new == source
    assert inserted == set()


def test_backslash_text_uses_raw_string():
    source = "def f(p):\n    return p\n"
    text = "Match \\d+ digits.\n\nArgs:\n    p: Pattern.\n\nReturns:\n    bool: Matched."
    new = check(source, {1: text})
    assert 'r"""' in new
    assert docstring_at(new, 1) == text  # backslash survived verbatim


@pytest.mark.parametrize("text", ['Uses """ quotes.', "Uses ''' quotes.", "Trails \\"])
def test_unquotable_text_is_refused(text):
    source = "def f():\n    return 1\n"
    new, inserted = insert_docstrings(source, {1: text})
    assert new == source
    assert inserted == set()


def test_text_ending_in_quote_switches_quote_style():
    source = "def f(a):\n    return a\n"
    text = 'Handles the "quoted"'
    new = check(source, {1: text})
    # """Handles the "quoted"""" would be a SyntaxError.
    assert "'''" in new
    assert docstring_at(new, 1) == text


def test_multiline_text_ending_in_quote():
    source = "def f(a):\n    return a\n"
    text = 'Sum.\n\nArgs:\n    a: The "thing"'
    new = check(source, {1: text})
    assert docstring_at(new, 1) == text


def test_comment_before_first_statement():
    source = "def f(a):\n    # a note\n    return a\n"
    text = "Sum.\n\nArgs:\n    a: Thing."
    new = check(source, {1: text})
    # Docstring goes above the comment, and the comment survives at its indent.
    assert new.index('"""Sum.') < new.index("# a note")
    assert "    # a note\n" in new
    assert docstring_at(new, 1) == text


def test_crlf_source_stays_crlf():
    source = "def f(a):\r\n    return a\r\n"
    text = "Sum.\n\nArgs:\n    a: Thing."
    new, _ = insert_docstrings(source, {1: text})
    ast.parse(new)
    assert "\r\n" in new
    # No lone \n anywhere: every newline is part of a \r\n pair.
    assert new.replace("\r\n", "") .count("\n") == 0
    assert docstring_at(new, 1) == text


def test_blank_line_has_no_trailing_whitespace():
    source = "class A:\n    def m(self, a):\n        return a\n"
    text = "Sum.\n\nArgs:\n    a: Thing."
    new = check(source, {2: text})
    for line in new.split("\n"):
        assert line == line.rstrip(), f"trailing whitespace on {line!r}"


def test_single_line_docstring():
    source = "def reset():\n    global x\n    x = 0\n"
    text = "Reset the counter."
    new = check(source, {1: text})
    assert '    """Reset the counter."""\n' in new
    assert docstring_at(new, 1) == text


@pytest.mark.parametrize("line", [2, 99])
def test_unmatched_line_is_a_noop(line):
    source = "def f(a):\n    return a\n"
    new, inserted = insert_docstrings(source, {line: "Nope."})
    assert new == source
    assert inserted == set()
