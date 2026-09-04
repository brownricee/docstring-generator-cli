import math

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = base

        # Freeze the WHOLE base module, not just .weight. Iterating
        # parameters() covers weight and bias together and sidesteps the
        # bias=None case for free
        for p in self.linear.parameters():
            p.requires_grad = False

        factory = {"device": base.weight.device, "dtype": base.weight.dtype}

        # bias=False on both: LoRA learns a low-rank *correction*, and the base
        # layer already carries whatever bias the pretrained model needs.
        self.lora_A = nn.Linear(base.in_features, r, bias=False, **factory)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, **factory)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # alpha/r decouples rank from effective adapter learning rate: doubling
        # r halves the scale, so r is a capacity knob rather than a second,
        # hidden lr knob.
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base model output
        base_out = self.linear(x)

        return base_out + self.scale * self.lora_B(self.lora_A(x))

def main():
    # Smoke test only: construct a wrapper and confirm it builds. Loss and
    # optimizer moved to train.py -- they belong to the training run, not to
    # the layer definition.
    base = nn.Linear(64, 64)
    model = LoRALinear(base, r=16, alpha=32)
    model.to(device)

if __name__ == "__main__":
    main()