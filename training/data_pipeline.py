import ast
import inspect
import json
import pathlib
import collections
import re
from datasets import load_dataset

# Package-qualified so this resolves from the repo root rather than only from
# inside training
from training.ast_extractor import (
    FuncDef,
    has_params,
    is_generator,
    returns_value,
    strip_docstring,
)

# `\r` is in the TRAILING class only. Under re.M, `$` matches just before a
# `\n`, so in a CRLF docstring the `\r` sits between "Args:" and that position
# and has to be consumable -- without it, "Args:\r\n" reads as not-a-header and
# ~408 real training pairs get silently dropped. The leading class needs no
# `\r`, since in CRLF the `\r` ends the previous line rather than starting this
# one.
_SECTIONS = {n: re.compile(rf"^[ \t]*{n}:[ \t\r]*$", re.M)
             for n in ("Args", "Returns", "Yields")}

def load_raw():
    ds = load_dataset("code_search_net", "python")
    return ds

def is_google_style(docstring: str, fn: FuncDef) -> bool:
    args_ok = has_section(docstring, "Args") == has_params(fn)
    returns_ok = has_section(docstring, "Returns") == returns_value(fn)
    yields_ok = has_section(docstring, "Yields") == is_generator(fn)

    return args_ok and returns_ok and yields_ok and any(has_section(docstring, n) for n in ("Args", "Returns", "Yields"))

def has_section(docstring: str, section_name: str) -> bool:
    # True if docstring contains a Google section header on a line of its own.
    return _SECTIONS[section_name].search(docstring) is not None



def is_reasonable_length(docstring: str, code: str) -> bool:
    doc_lines = docstring.strip().count("\n") + 1
    return 2 <= doc_lines <= 30 and len(code) < 2000

def build_split(ds) -> tuple:
    # One pass, one `continue` per rejection, one counter bump per survivor.
    # Order matters: each stage must be a strict subset of the one above it, so
    # the printed funnel reads top to bottom and a broken filter is obvious.
    records = []
    counts = collections.Counter()

    for ex in ds:
        counts["raw"] += 1

        # Read the CodeSearchNet field names ONCE, here. Below this point
        # everything is plain strings, so nothing else in the file has to know
        # what the dataset calls its columns.
        doc = ex["func_documentation_string"]
        src = ex["func_code_string"]

        # try wraps ONLY the parse. Anything wider risks swallowing a
        # SyntaxError raised by something that isn't the input.
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # Python 2 and friends; ~1% of the corpus, expected

        # `not tree.body` guards the index: ast.parse("") returns a Module with
        # an empty body, and tree.body[0] would raise IndexError -- which the
        # except above does not catch, so it would crash the run.
        if not tree.body or not isinstance(tree.body[0], FuncDef):
            continue
        fn = tree.body[0]
        counts["parsed"] += 1

        if not is_google_style(doc, fn):
            continue
        counts["google_style"] += 1

        code = strip_docstring(src)
        if code is None:
            continue
        counts["stripped"] += 1

        # cleans up indentation from multi-line strings
        doc = inspect.cleandoc(doc)

        if not is_reasonable_length(doc, code):
            continue
        counts["length_ok"] += 1

        records.append({"code": code, "docstring": doc})

    return records, counts

def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    """Write one JSON object per line to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # encoding="utf-8" -- Windows would otherwise default to cp1252 and raise
    #                     UnicodeEncodeError on the many non-ASCII docstrings.
    # newline=""       -- stops Python translating each "\n" we write into
    #                     "\r\n" on Windows, which would put a stray \r inside
    #                     every JSON string and change the data.
    with open(path, "w", encoding="utf-8", newline="") as f:
        for rec in records:
            # ensure_ascii=False keeps unicode as real characters instead of
            # \uXXXX escapes: smaller file, and readable when you eyeball it.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

_FUNNEL = ("raw", "parsed", "google_style", "stripped", "length_ok")

# HuggingFace calls the middle split "validation"
_SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))


def main():
    ds = load_raw()

    # Regression guard, not a fix: the native splits are already repo-disjoint
    # (measured: 0 shared repos). This fails loudly if that ever stops being
    # true -- e.g. if someone re-splits ds["train"] by hand later.
    train_repos = set(ds["train"]["repository_name"])
    for split in ("validation", "test"):
        overlap = train_repos & set(ds[split]["repository_name"])
        assert not overlap, f"{split} shares {len(overlap)} repos with train"

    # Anchored to this file, not the working directory, so output lands in
    # <repo>/data/ regardless of where the command is run from.
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "data"

    for hf_name, out_name in _SPLITS:
        records, counts = build_split(ds[hf_name])
        path = out_dir / f"{out_name}.jsonl"
        write_jsonl(records, path)

        print(f"\n{hf_name} -> {path}")
        for stage in _FUNNEL:
            print(f"  {stage:<14}{counts[stage]:>8}")

if __name__ == "__main__":
    main()