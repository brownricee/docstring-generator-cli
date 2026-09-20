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
│   ├── train.py           # Training loop
│   ├── load_adapter.py    # Loads the base model + trained LoRA adapter for inference
│   └── evaluate.py        # Base-vs-fine-tuned comparison on held-out test.jsonl
├── data/                  # Generated train/val/test JSONL (committed, so Colab clones get it)
├── checkpoints/           # Trained LoRA weights (gitignored -- see "Getting the trained weights")
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

## Getting the trained weights

Training runs on Colab and takes several hours, so the LoRA adapter isn't
committed to git. It's attached to a [GitHub Release](https://github.com/brownricee/docstring-generator-cli/releases) instead:

```bash
gh release download adapter-v1 --pattern lora_weights.pt --dir checkpoints
```

(or download `lora_weights.pt` from the Releases page and place it at
`checkpoints/lora_weights.pt` yourself)

Then verify it loads:

```bash
python -m training.load_adapter
```

## Evaluation

`training/evaluate.py` compares the base model against the fine-tuned model on
a fixed random sample of `data/test.jsonl`, scoring each generated docstring
with the same conditional-section style filter (`is_google_style`) used to
build the training data in the first place:

```bash
python -m training.evaluate          # 30 samples by default
python -m training.evaluate --n 100  # larger sample, slower on CPU
```

It also prints the LoRA trainable-parameter fraction and a handful of
side-by-side (code, reference docstring, base output, fine-tuned output)
examples for a manual read.

| Run | r | alpha | lr | epochs | trainable % | base pass-rate | fine-tuned pass-rate |
|---|---|---|---|---|---|---|---|
| baseline | 16 | 32 | 2e-4 | 3 | 0.282% | *(pending)* | *(pending)* |

## License

MIT — see [LICENSE](LICENSE).
