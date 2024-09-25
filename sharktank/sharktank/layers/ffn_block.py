# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Theta, ThetaLayer
from .linear import LinearLayer

__all__ = [
    "FFN",
]

def compare_baseline(t, name, baseline, replace=False):
    if not baseline:
        return t
    shp = [int(d) for d in t.shape]
    shp[1] = -1

    t_baseline = torch.tensor(baseline[name].reshape(*shp))
    t_cut = torch.tensor(t[:, :t_baseline.shape[1], :])

    diff = torch.abs(t_cut - t_baseline)
    diff = diff.flatten()

    atol = torch.max(diff)
    print (f"Tensor {name:<40} {atol.item():10.6f}")

    # Substitute in values
    if replace:
        return torch.concat((t_baseline, t[:, t_baseline.shape[1]:, :]), dim=1)

    return t

class FFN(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        baseline: dict = None,
    ):
        super().__init__(theta)
        self.baseline = baseline
        self.add_module("ffn_gate", LinearLayer(theta("ffn_gate")))
        self.add_module("ffn_up", LinearLayer(theta("ffn_up")))
        self.add_module("ffn_down", LinearLayer(theta("ffn_down")))

    def forward(
        self,
        h: torch.Tensor,
    ):
        ffn_gate = self.ffn_gate(h)
        ffn_gate = compare_baseline(ffn_gate, "ffn_gate", self.baseline, replace=True)

        ffn_gate = F.silu(ffn_gate)
        ffn_gate = compare_baseline(ffn_gate, "ffn_silu", self.baseline, replace=True)

        ffn_up = self.ffn_up(h)
        ffn_up = compare_baseline(ffn_up, "ffn_up", self.baseline, replace=True)

        ffn_down = self.ffn_down(ffn_gate * ffn_up)
        return ffn_down
