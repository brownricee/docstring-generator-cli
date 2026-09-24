import torch
import torch.nn as nn

from docgen.lora import LoRALinear


def test_fuse_matches_forward_before_fusing():
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=4, alpha=8)

    # Zero-init lora_B would make the delta trivially zero and hide bugs.
    nn.init.normal_(wrapped.lora_A.weight)
    nn.init.normal_(wrapped.lora_B.weight)

    x = torch.randn(3, 8)
    before = wrapped(x)

    fused = wrapped.fuse()

    assert isinstance(fused, nn.Linear)
    assert not isinstance(fused, LoRALinear)
    assert torch.allclose(fused(x), before, atol=1e-5)
