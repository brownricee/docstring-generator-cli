import ast

import pytest

from docgen.generator import is_usable, postprocess

ARGS_RETURNS = "Sum two numbers.\n\nArgs:\n    a: First.\n\nReturns:\n    int: Sum."


def parse(source: str):
    return ast.parse(source).body[0]


def test_plain_output_is_unchanged():
    assert postprocess(ARGS_RETURNS) == ARGS_RETURNS


def test_run_on_into_new_code_is_truncated():
    raw = ARGS_RETURNS + "\n\n\ndef something_else(x):\n    return x\n"
    assert postprocess(raw) == ARGS_RETURNS


def test_run_on_into_a_decorator_or_class_is_truncated():
    assert postprocess("Summary.\n\n@property\ndef v(self): ...") == "Summary."
    assert postprocess("Summary.\n\nclass Other:\n    pass") == "Summary."
    assert postprocess("Summary.\n\nif __name__ == '__main__':\n    main()") == "Summary."


def test_repeated_prompt_is_cut():
    raw = "Summary.\n\nWrite a Google-style docstring for the following Python function."
    assert postprocess(raw) == "Summary."


def test_leading_code_fence_is_stripped_not_truncated():
    raw = "```python\n" + ARGS_RETURNS + "\n```"
    assert postprocess(raw) == ARGS_RETURNS


def test_wrapping_triple_quotes_are_removed():
    assert postprocess('"""' + ARGS_RETURNS + '"""') == ARGS_RETURNS
    assert postprocess("'''Summary.'''") == "Summary."


def test_trailing_quotes_without_an_opener_are_removed():
    assert postprocess('Summary.\n"""') == "Summary."


def test_indented_section_content_survives():
    """The stop patterns must not fire on indented lines inside a section."""
    raw = "Summary.\n\nExample:\n    def inner():\n        pass\n\nReturns:\n    int: A."
    assert "def inner():" in postprocess(raw)


def test_common_indent_is_normalized():
    raw = "Summary.\n\n    Args:\n        a: First."
    assert postprocess(raw) == "Summary.\n\nArgs:\n    a: First."


def test_whitespace_only_output_is_empty():
    assert postprocess("   \n\n  ") == ""


def test_is_usable_accepts_a_no_arg_void_summary():
    """sections_match, not is_google_style -- this is the whole point."""
    assert is_usable("Reset the counter.", parse("def reset():\n    global x\n    x = 0"))


def test_is_usable_rejects_contradictory_sections():
    fn = parse("def f(a):\n    return a")
    assert is_usable(ARGS_RETURNS, fn)
    assert not is_usable("Summary.\n\nReturns:\n    int: A.", fn)  # missing Args
    assert not is_usable("Summary.\n\nArgs:\n    a: A.", fn)  # missing Returns


def test_is_usable_rejects_a_generator_without_yields():
    fn = parse("def f(a):\n    yield a")
    assert not is_usable("Summary.\n\nArgs:\n    a: A.", fn)
    assert is_usable("Summary.\n\nArgs:\n    a: A.\n\nYields:\n    int: A.", fn)


@pytest.mark.parametrize("text", ["", 'Has """ inside', "Has ''' inside", "Trails \\"])
def test_is_usable_rejects_unquotable_text(text):
    assert not is_usable(text, parse("def f():\n    pass"))


def test_is_usable_rejects_an_invented_raises_section():
    """The real failure this caught: a hallucinated Raises: on an __init__."""
    fn = parse("class A:\n    def __init__(self):\n        self.total = 0").body[0]
    invented = "Constructor.\n\nRaises:\n    ValueError: if the total is negative."
    assert not is_usable(invented, fn)
    assert is_usable("Constructor.", fn)


def test_is_usable_allows_raises_when_the_function_raises():
    fn = parse("def f(a):\n    if a < 0:\n        raise ValueError(a)\n    return a")
    assert is_usable(
        "Check a.\n\nArgs:\n    a: A.\n\nReturns:\n    int: A.\n\nRaises:\n    ValueError: bad.",
        fn,
    )


def test_is_usable_allows_omitting_raises():
    """One-directional: documenting Raises is optional, inventing it is not."""
    fn = parse("def f(a):\n    if a < 0:\n        raise ValueError(a)\n    return a")
    assert is_usable("Check a.\n\nArgs:\n    a: A.\n\nReturns:\n    int: A.", fn)
