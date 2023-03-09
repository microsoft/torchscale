# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy

import torch
import torch.nn as nn


def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1