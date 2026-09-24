import logging
import os
import pathlib
import shutil
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from docgen.lora import LoRALinear

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"

# Every one of Qwen's layers contains these four nn.Linear modules -- they are
# the attention part of the layer. Wrapping only these, and leaving the much
# bigger feed-forward layers beside them alone, is what keeps the trainable
# count near 0.1-1% of the model.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

ADAPTER_URL = (
    "https://github.com/brownricee/docstring-generator-cli/releases/download/"
    "adapter-v1/lora_weights.pt"
)
CACHE_DIR = pathlib.Path(
    os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
) / "docgen"


def apply_lora(model, r: int = 16, alpha: int = 32) -> int:
    """Replace every targeted nn.Linear with a LoRALinear wrapping it.

    Returns how many were replaced. Check it is not zero: wrapping nothing
    trains nothing, and that looks exactly like a bad learning rate.
    """
    to_replace = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if child_name in TARGET_MODULES:
                to_replace.append((parent, child_name, child))

    for parent, child_name, child in to_replace:
        setattr(parent, child_name, LoRALinear(child, r, alpha))

    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    return len(to_replace)


def fuse_lora(model) -> int:
    """Replace every LoRALinear with its fused nn.Linear equivalent.

    Returns how many were fused. Call only after the adapter's weights are
    loaded and finalized -- fusing before that bakes in the untrained init
    instead of the trained correction.
    """
    to_fuse = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, LoRALinear):
                to_fuse.append((parent, child_name, child))

    for parent, child_name, child in to_fuse:
        setattr(parent, child_name, child.fuse())

    return len(to_fuse)


def resolve_adapter(override: pathlib.Path | None = None) -> pathlib.Path:
    """Return a local path to the adapter, downloading it on first use.

    An installed CLI has no repo checkout, so the checkpoints/ directory the
    training scripts use does not exist. Cache the release asset instead.
    """
    if override is not None:
        return override

    dest = CACHE_DIR / "lora_weights.pt"
    if dest.exists():
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading adapter (8.4 MB) -> {dest}")

    # Download to .part and rename. os.replace is atomic, so a Ctrl-C partway
    # through leaves no half-written file that later runs would trust as cached.
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(ADAPTER_URL) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp, dest)

    return dest


def load_model(adapter_path: pathlib.Path):
    """Load the base model wrapped with LoRA and restore trained adapter weights.

    Args:
        adapter_path: Path to a lora_weights.pt saved by training/train.py.

    Returns:
        A (model, tokenizer) pair, with the model in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # bfloat16 matmul on a CPU without AVX512-BF16 is emulated and crawls.
    # float32 costs ~6 GB of RAM but is the only usable CPU option; the cuda
    # branch is unchanged, so Colab eval numbers stay reproducible.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # torch_dtype=, not dtype=. The `dtype` alias only exists from transformers
    # 4.56; on 4.51 it is swallowed into config kwargs and the requested dtype
    # is silently ignored. torch_dtype= is understood by every version in our
    # supported range.
    #
    # attn_implementation="sdpa" asks for PyTorch's fused scaled_dot_product_attention
    # kernel instead of transformers' eager attention loop. Verified after load,
    # not just requested: an unsupported kwarg has silently no-op'd before (see
    # the torch_dtype note above), and scaled_dot_product_attention itself has
    # existed unconditionally since torch 2.0, so a hard crash here is unlikely --
    # the real risk is a silent fallback, which only checking model.config catches.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    actual_attn = getattr(model.config, "_attn_implementation", None)
    if actual_attn != "sdpa":
        logger.warning("requested sdpa attention but got %r", actual_attn)
    apply_lora(model)

    state_dict = torch.load(adapter_path, map_location=device)
    result = model.load_state_dict(state_dict, strict=False)
    # missing_keys is every frozen base-model weight, which never appears in an
    # adapter-only checkpoint -- only unexpected_keys indicates an actual mismatch.
    assert not result.unexpected_keys, (
        f"checkpoint has keys the model doesn't: {result.unexpected_keys}"
    )

    # Fuse now, not before load_state_dict: fusing must see the trained
    # lora_A/lora_B values, not their kaiming/zero init.
    fused = fuse_lora(model)
    assert fused > 0, "fused nothing -- LoRA wrapping must have failed silently"

    model.eval()
    return model, tokenizer
