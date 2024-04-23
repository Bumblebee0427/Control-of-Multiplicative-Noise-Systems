import typing

import torch.nn as nn
from torch import Tensor
import torch


class M1P1LinearModule(nn.Module):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t.
    """
    def forward(self, y_history: typing.List[Tensor]):  # List[()] -> ()
        y_t: Tensor = y_history[-1]  # ()
        u_t: Tensor = self.theta * y_t  # ()
        return u_t


class FixedWeightM1P1LinearModule(M1P1LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is fixed at initialization.
    """
    def __init__(self, theta: float):
        super(FixedWeightM1P1LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.tensor(theta), requires_grad=False)


class LearnableWeightM1P1LinearModule(M1P1LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is trainable.
    Initializes theta_0 = 0.
    """
    def __init__(self):
        super(LearnableWeightM1P1LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)


__all__ = ["FixedWeightM1P1LinearModule", "LearnableWeightM1P1LinearModule"]
