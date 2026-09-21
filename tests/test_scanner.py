import ast

import pytest

from docgen.scanner import (
    Unsupported,
    function_source,
    iter_py_files,
    prepare,
    read_source,
    scan_file,
    write_source,
)

SOURCE = (
    'def documented(a):\n'
    '    """Has one.\n\n    Args:\n        a: A.\n    """\n'
    "    return a\n"
    "\n"
    "def bare(x):\n"
    "    return x * 2\n"
    "\n"
    "class Thing:\n"
    "    @staticmethod\n"
    "    def method(y):\n"
    "        return y\n"
    "\n"
    "    async def fetch(self):\n"
    "        return 1\n"
    "\n"
    "def one_liner(): return 1\n"
)


def write(tmp_path, source, name="m.py"):
    path = tmp_path / name
    write_source(path, source)
    return path


def test_scan_file_reports_missing_and_total(tmp_path):
    source, missing, total = scan_file(write(tmp_path, SOURCE))
    assert total == 5
    assert {r["qualname"] for r in missing} == {
        "bare",
        "Thing.method",
        "Thing.fetch",
        "one_liner",
    }


def test_scan_file_returns_none_for_unparseable(tmp_path):
    assert scan_file(write(tmp_path, "def f(:\n")) is None


def test_function_source_dedents_a_method(tmp_path):
    source, missing, _ = scan_file(write(tmp_path, SOURCE))
    record = next(r for r in missing if r["qualname"] == "Thing.method")
    code = function_source(source, record)
    assert code.startswith("def method(y):")
    assert ast.parse(code)  # parses standalone only because it was dedented


def test_function_source_excludes_the_decorator(tmp_path):
    source, missing, _ = scan_file(write(tmp_path, SOURCE))
    record = next(r for r in missing if r["qualname"] == "Thing.method")
    assert "@staticmethod" not in function_source(source, record)


def test_prepare_rejects_one_line_body(tmp_path):
    source, missing, _ = scan_file(write(tmp_path, SOURCE))
    record = next(r for r in missing if r["qualname"] == "one_liner")
    with pytest.raises(Unsupported, match="one-line body"):
        prepare(source, record)


def test_prepare_rejects_a_long_function(tmp_path):
    body = "\n".join(f"    x{i} = {i}" for i in range(400))
    source = f"def big(a):\n{body}\n    return a\n"
    src, missing, _ = scan_file(write(tmp_path, source))
    with pytest.raises(Unsupported, match="too long"):
        prepare(src, missing[0])


def test_prepare_returns_code_and_node(tmp_path):
    source, missing, _ = scan_file(write(tmp_path, SOURCE))
    record = next(r for r in missing if r["qualname"] == "bare")
    code, fn = prepare(source, record)
    assert code == "def bare(x):\n    return x * 2\n"
    assert fn.name == "bare"


def test_iter_py_files_accepts_a_file_or_a_directory(tmp_path):
    a = write(tmp_path, "def f():\n    pass\n", "a.py")
    (tmp_path / "pkg").mkdir()
    b = write(tmp_path / "pkg", "def g():\n    pass\n", "b.py")
    (tmp_path / "notes.txt").write_text("ignored")

    assert iter_py_files(a) == [a]
    assert set(iter_py_files(tmp_path)) == {a, b}


def test_read_write_source_preserve_crlf(tmp_path):
    path = write(tmp_path, "def f(a):\r\n    return a\r\n")
    source = read_source(path)
    assert source == "def f(a):\r\n    return a\r\n"
    write_source(path, source)
    assert path.read_bytes() == b"def f(a):\r\n    return a\r\n"
