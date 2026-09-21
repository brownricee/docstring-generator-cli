import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docgen.cli import app
from docgen.scanner import read_source, write_source

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE = (
    "def double(x):\n"
    '    """Double a number.\n'
    "\n"
    "    Args:\n"
    "        x: The number.\n"
    "\n"
    "    Returns:\n"
    "        int: Twice x.\n"
    '    """\n'
    "    return x * 2\n"
    "\n"
    "\n"
    "def word_count(text):\n"
    "    counts = {}\n"
    "    for word in text.split():\n"
    "        counts[word] = counts.get(word, 0) + 1\n"
    "    return counts\n"
    "\n"
    "\n"
    "class Accumulator:\n"
    '    """Running total."""\n'
    "\n"
    "    def add(self, value):\n"
    "        self.total += value\n"
    "        return self.total\n"
)

GOOD = "Summary line.\n\nArgs:\n    thing: A thing.\n\nReturns:\n    int: A number."

runner = CliRunner()


class StubGenerator:
    """Stands in for the real model: returns canned text, records its calls."""

    def __init__(self, text=GOOD):
        self.text = text
        self.calls = []

    def generate(self, code):
        self.calls.append(code)
        return self.text


@pytest.fixture
def stub(monkeypatch):
    """Patch load_generator so no model is ever loaded."""
    from docgen import generator

    engine = StubGenerator()
    monkeypatch.setattr(generator, "load_generator", lambda adapter=None: engine)
    return engine


@pytest.fixture
def module(tmp_path):
    path = tmp_path / "m.py"
    write_source(path, SOURCE)
    return path


def test_scan_lists_gaps(module):
    result = runner.invoke(app, ["scan", str(module)])
    assert result.exit_code == 0
    assert "word_count" in result.output
    assert "Accumulator.add" in result.output
    assert "2 of 3 functions missing docstrings" in result.output


def test_scan_does_not_import_torch(module):
    """`scan` must stay instant -- a model import would cost seconds.

    Runs in a subprocess: other tests in this session import torch, so
    checking sys.modules in-process would prove nothing.
    """
    program = textwrap.dedent(
        f"""
        import sys
        from typer.testing import CliRunner
        from docgen.cli import app
        result = CliRunner().invoke(app, ["scan", {str(module)!r}])
        assert result.exit_code == 0, result.output
        print("TORCH" if "torch" in sys.modules else "CLEAN")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "CLEAN" in proc.stdout


def test_fill_dry_run_prints_diff_and_changes_nothing(module, stub):
    before = module.read_bytes()
    result = runner.invoke(app, ["fill", str(module)])

    assert result.exit_code == 0
    assert "+++" in result.output and '+    """Summary line.' in result.output
    assert "dry run" in result.output
    assert module.read_bytes() == before


def test_fill_write_applies_and_output_parses(module, stub):
    result = runner.invoke(app, ["fill", str(module), "--write"])
    assert result.exit_code == 0

    new = read_source(module)
    docs = {
        n.name: ast.get_docstring(n, clean=True)
        for n in ast.walk(ast.parse(new))
        if isinstance(n, ast.FunctionDef)
    }
    assert docs["word_count"] == GOOD
    assert docs["add"] == GOOD
    # The already-documented function was left exactly as it was.
    assert docs["double"].startswith("Double a number.")
    assert "wrote 2 docstring(s)" in result.output


def test_fill_write_then_scan_reports_no_gaps(module, stub):
    runner.invoke(app, ["fill", str(module), "--write"])
    result = runner.invoke(app, ["scan", str(module)])
    assert "0 of 3 functions missing docstrings" in result.output


def test_limit_stops_after_n_generations(module, stub):
    result = runner.invoke(app, ["fill", str(module), "--write", "--limit", "1"])
    assert result.exit_code == 0
    assert len(stub.calls) == 1
    assert "wrote 1 docstring(s)" in result.output


def test_prompt_is_built_from_dedented_source(module, stub):
    runner.invoke(app, ["fill", str(module)])
    method_call = next(c for c in stub.calls if "def add" in c)
    assert method_call.startswith("def add(self, value):")


def test_unusable_generation_is_skipped_not_inserted(module, monkeypatch):
    """A docstring contradicting the signature must never reach the file."""
    from docgen import generator

    # word_count takes an arg and returns a value; a bare summary fails the gate.
    engine = StubGenerator("Just a summary with no sections.")
    monkeypatch.setattr(generator, "load_generator", lambda adapter=None: engine)

    before = module.read_bytes()
    result = runner.invoke(app, ["fill", str(module), "--write"])

    assert result.exit_code == 0
    assert "failed validation" in result.output
    assert module.read_bytes() == before


def test_nothing_to_do_on_a_fully_documented_file(tmp_path, stub):
    path = tmp_path / "done.py"
    write_source(path, 'def f():\n    """Done."""\n    return 1\n')
    result = runner.invoke(app, ["fill", str(path)])
    assert "nothing to do" in result.output
    assert stub.calls == []
