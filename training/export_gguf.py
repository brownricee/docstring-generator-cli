from docgen.model import load_model
from training.load_adapter import CHECKPOINT_PATH

OUT_DIR = CHECKPOINT_PATH.parent / "merged"


def main():
    """Save the base model with the adapter fused in, as a plain HF checkpoint.

    llama.cpp's convert_hf_to_gguf.py reads this directory; see the README for
    the conversion and quantization commands. load_model fuses in float32 on
    CPU, so the small LoRA delta is not rounded away by bfloat16 before saving.
    """
    model, tokenizer = load_model(CHECKPOINT_PATH)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved merged model to {OUT_DIR}")


if __name__ == "__main__":
    main()
