import torch
import torch.nn as nn

from docgen.lora import LoRALinear
from docgen.model import apply_lora, fuse_lora


class TinyAttn(nn.Module):
    """Stands in for a Qwen decoder layer: real nn.Linear children named
    like the ones apply_lora/fuse_lora target, without loading the real model.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)  # not a target -- must be left alone


def test_fuse_lora_removes_wrappers_and_preserves_output():
    model = TinyAttn()
    replaced = apply_lora(model, r=4, alpha=8)
    assert replaced == 2  # q_proj, o_proj -- not mlp

    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_A.weight)
            nn.init.normal_(module.lora_B.weight)

    x = torch.randn(3, 8)
    before = model.o_proj(model.q_proj(x))

    fused = fuse_lora(model)

    assert fused == 2
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert isinstance(model.q_proj, nn.Linear)
    assert isinstance(model.o_proj, nn.Linear)

    after = model.o_proj(model.q_proj(x))
    assert torch.allclose(after, before, atol=1e-5)
