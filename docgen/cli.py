import ast
import difflib
import pathlib
from typing import Optional

import typer

from docgen.inserter import insert_docstrings
from docgen.scanner import (
    Unsupported,
    iter_py_files,
    prepare,
    scan_file,
    write_source,
)

app = typer.Typer(
    help="Find Python functions missing docstrings and fill them in.",
    add_completion=False,
)

PathArg = typer.Argument(..., exists=True, help="File or directory to process.")


@app.command()
def scan(path: pathlib.Path = PathArg):
    """List functions that have no docstring. Loads no model."""
    total = 0
    gaps = 0
    for file in iter_py_files(path):
        result = scan_file(file)
        if result is None:
            continue
        _, missing, count = result
        total += count
        for record in missing:
            gaps += 1
            typer.echo(f"{file}:{record['lineno']}  {record['qualname']}")

    typer.echo(f"\n{gaps} of {total} functions missing docstrings")


@app.command()
def fill(
    path: pathlib.Path = PathArg,
    write: bool = typer.Option(False, "--write", help="Apply changes in place."),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Stop after generating this many docstrings."
    ),
    adapter: Optional[pathlib.Path] = typer.Option(
        None, "--adapter", help="Path to lora_weights.pt (default: cached download)."
    ),
):
    """Generate docstrings for undocumented functions and insert them.

    Prints a diff by default; pass --write to modify files.
    """
    targets = []
    for file in iter_py_files(path):
        result = scan_file(file)
        if result is None:
            continue
        source, missing, _ = result
        if missing:
            targets.append((file, source, missing))

    if not targets:
        typer.echo("nothing to do: no undocumented functions found")
        return

    # Imported here, not at module scope, so `scan` never pays for torch.
    from docgen import generator as gen

    engine = gen.load_generator(adapter)

    written = 0
    generated = 0
    for file, source, missing in targets:
        docstrings = {}
        for record in missing:
            if limit is not None and generated >= limit:
                break
            where = f"{file}:{record['lineno']} {record['qualname']}"
            try:
                code, fn = prepare(source, record)
            except Unsupported as exc:
                typer.echo(f"  skip {where}: {exc}")
                continue

            generated += 1
            text = gen.postprocess(engine.generate(code))
            if not gen.is_usable(text, fn):
                typer.echo(f"  skip {where}: generated docstring failed validation")
                continue
            docstrings[record["lineno"]] = text

        if not docstrings:
            continue

        new_source, inserted = insert_docstrings(source, docstrings)
        if not inserted:
            continue

        # The one check that catches any quoting bug the inserter did not
        # anticipate. Never hand back a file that stopped being valid Python.
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            typer.echo(f"  skip {file}: insertion produced invalid Python ({exc})")
            continue

        if write:
            write_source(file, new_source)
            written += len(inserted)
            typer.echo(f"  wrote {len(inserted)} docstring(s) to {file}")
        else:
            diff = difflib.unified_diff(
                source.splitlines(keepends=True),
                new_source.splitlines(keepends=True),
                fromfile=str(file),
                tofile=str(file),
            )
            typer.echo("".join(diff))

    if write:
        typer.echo(f"\nwrote {written} docstring(s)")
    else:
        typer.echo("\ndry run -- pass --write to apply")
