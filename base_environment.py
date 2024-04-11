import typing

import torch
from torch import Tensor


class Environment:
    """
    Defines the transition dynamics of the environment.
    """
    def __init__(self):
        self.x: Tensor = None
        self.y: Tensor = None

    def oracle(self) -> Tensor:
        # never use this anywhere except to generate x for logging/viz purposes
        return torch.clone(self.x)

    def observe(self) -> Tensor:
        return torch.clone(self.y)

    def step(self, u: Tensor):
        raise NotImplementedError

    def loss(self, u: Tensor) -> Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


__all__ = ["Environment"]
