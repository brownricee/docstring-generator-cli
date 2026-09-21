import ast

import pytest

from docgen.astutils import is_google_style, scan_source, sections_match

# (source, docstring-with-every-section, which sections the signature requires)
CASES = [
    ("def f(a):\n    return a\n", {"Args", "Returns"}),
    ("def f(a):\n    print(a)\n", {"Args"}),
    ("def f():\n    return 1\n", {"Returns"}),
    ("def f(a):\n    yield a\n", {"Args", "Yields"}),
    ("def f():\n    yield 1\n", {"Yields"}),
    ("def f():\n    pass\n", set()),
    ("def f():\n    return None\n", set()),  # explicit `return None` is void
    ("class A:\n    def m(self):\n        pass\n", set()),  # self is not a param
]


def parse(source: str):
    node = ast.parse(source).body[0]
    return node.body[0] if isinstance(node, ast.ClassDef) else node


def docstring_with(sections: set[str]) -> str:
    body = "Summary."
    for name in ("Args", "Returns", "Yields"):
        if name in sections:
            body += f"\n\n{name}:\n    thing: A thing."
    return body


@pytest.mark.parametrize("source,required", CASES)
def test_sections_match_requires_exactly_the_right_sections(source, required):
    fn = parse(source)
    assert sections_match(docstring_with(required), fn)

    # Any other combination of sections contradicts the signature.
    for name in ("Args", "Returns", "Yields"):
        wrong = required ^ {name}
        assert not sections_match(docstring_with(wrong), fn)


@pytest.mark.parametrize("source,required", CASES)
def test_is_google_style_is_sections_match_plus_one_section(source, required):
    """The corpus filter must not have changed when it moved out of data_pipeline."""
    fn = parse(source)
    doc = docstring_with(required)
    assert is_google_style(doc, fn) == (sections_match(doc, fn) and bool(required))


def test_no_arg_void_function_is_the_gates_one_disagreement():
    """The reason inference uses sections_match instead of is_google_style."""
    fn = parse("def reset():\n    global x\n    x = 0\n")
    doc = "Reset the counter to zero."
    assert sections_match(doc, fn) is True  # safe to insert
    assert is_google_style(doc, fn) is False  # but too bare for training data


def test_crlf_section_headers_are_recognized():
    """The \\r in the section regex is load-bearing -- see the comment there."""
    fn = parse("def f(a):\r\n    return a\r\n")
    assert sections_match("Summary.\r\n\r\nArgs:\r\n    a: A.\r\n\r\nReturns:\r\n    int: B.", fn)


def test_scan_source_reports_line_span_and_docstring_state():
    source = (
        "def documented(a):\n"
        '    """Has one.\n\n    Args:\n        a: A.\n    """\n'
        "    return a\n"
        "\n"
        "class A:\n"
        "    async def m(self):\n"
        "        pass\n"
    )
    records = {r["qualname"]: r for r in scan_source(source, "x.py")}

    assert records["documented"]["has_docstring"] is True
    assert records["documented"]["lineno"] == 1
    assert records["documented"]["end_lineno"] == 7

    assert records["A.m"]["has_docstring"] is False
    assert records["A.m"]["lineno"] == 10
    assert records["A.m"]["end_lineno"] == 11


def test_scan_source_lineno_is_the_def_not_the_decorator():
    source = "class A:\n    @property\n    def v(self):\n        return 1\n"
    (record,) = scan_source(source, "x.py")
    assert record["lineno"] == 3
