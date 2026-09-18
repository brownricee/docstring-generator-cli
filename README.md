# docstring-generator-cli

A pip-installable CLI that scans a Python codebase for functions missing
docstrings and generates them with a fine-tuned language model.

Most open-source repos aren't fully undocumented — they have a handful of
gaps. This tool finds and fills exactly those gaps rather than assuming a repo
needs docs from scratch.

## How it works

1. **Data** — (Google-style, function, docstring) pairs sourced from
   CodeSearchNet Python, filtered to a single consistent docstring style and
   cleaned aggressively for quality over volume.
2. **Model** — a small open code model (Qwen2.5-Coder) fine-tuned with LoRA,
   implemented from scratch in PyTorch: a custom `LoRALinear` module wraps the
   frozen attention projection layers with trainable low-rank matrices, so
   only a small fraction of parameters are trained.

## Project structure

```
docstring-generator-cli/
├── training/
│   ├── data_pipeline.py   # Filters CodeSearchNet down to clean, Google-style (code, docstring) pairs
│   ├── ast_extractor.py   # AST helpers: strip docstrings, detect params/returns/yields
│   ├── lora.py            # LoRALinear module (LoRA adapter for an nn.Linear layer)
│   └── train.py           # Training loop (in progress)
├── data/                  # Generated train/val/test JSONL (committed, so Colab clones get it)
├── requirements.txt       # Training deps
├── requirements-data.txt  # Dataset-building deps (data_pipeline.py only)
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Rebuilding the dataset additionally needs `datasets`, which is kept in a
separate file so training runs don't install it for nothing:

```bash
pip install -r requirements-data.txt
```

Everything runs as a module from the repo root, e.g.:

```bash
python -m training.data_pipeline
```

## Building the dataset

`training/data_pipeline.py` downloads CodeSearchNet's Python split and
filters it down to clean training pairs:

- Parses each function with `ast`; drops anything that fails to parse.
- Keeps only docstrings that follow a strict Google-style convention: an
  `Args:` section iff the function takes parameters, a `Returns:` section iff
  it returns a value, a `Yields:` section iff it's a generator.
- Strips the docstring out of the source (via line slicing, not
  `ast.unparse`, to preserve comments and formatting) so the model is never
  trained on an input that already contains the answer.
- Writes `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl`, one
  `{"code": ..., "docstring": ...}` record per line.

Current output sizes: ~24.7k train / ~1k val / ~1.7k test pairs.

## License

MIT — see [LICENSE](LICENSE).
