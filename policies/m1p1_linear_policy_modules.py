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

class LinearMemory1Period1(nn.Module):
    """
    Implements the linear memory-1 period-1 control policy F(Y_(t); θ) = θ_0 * Y_t.
    """
    def __init__(self, theta_0: float):
        super(LinearMemory1Period1, self).__init__()
        self.theta_0 = nn.Parameter(torch.tensor([theta_0]), requires_grad=True)
    
    def forward(self, y_t: Tensor) -> Tensor:
        return self.theta_0 * y_t

class AffineMemory2Period1(nn.Module):
    """
    Implements the affine memory-2 period-1 control policy described by the equations:
    F(Y_(t); θ) = { θ_0 + θ_1 * Y_t,     if t = 0
                   { θ_0 + θ_1 * Y_t + θ_2 * Y_t-1,  if t >= 1.
    """
    def __init__(self, theta_0: float, theta_1: float, theta_2: float):
        super(AffineMemory2Period1, self).__init__()
        self.theta_0 = nn.Parameter(torch.tensor([theta_0]), requires_grad=True)
        self.theta_1 = nn.Parameter(torch.tensor([theta_1]), requires_grad=True)
        self.theta_2 = nn.Parameter(torch.tensor([theta_2]), requires_grad=True)
    
    def forward(self, y_history: typing.List[Tensor]) -> Tensor:
        y_t = y_history[-1]
        if len(y_history) == 1:  # t = 0
            return self.theta_0 + self.theta_1 * y_t
        else:  # t >= 1
            y_t_minus_1 = y_history[-2]
            return self.theta_0 + self.theta_1 * y_t + self.theta_2 * y_t_minus_1

class AffineMemory1Period2(nn.Module):
    """
    Implements the affine memory-1 period-2 control policy described by the equations:
    F(Y_(t); θ) = { θ_0 + θ_1 * Y_t,     if t is even
                   { θ_2 + θ_3 * Y_t,     if t is odd.
    """
    def __init__(self, theta_0: float, theta_1: float, theta_2: float, theta_3: float):
        super(AffineMemory1Period2, self).__init__()
        self.theta_0 = nn.Parameter(torch.tensor([theta_0]), requires_grad=True)
        self.theta_1 = nn.Parameter(torch.tensor([theta_1]), requires_grad=True)
        self.theta_2 = nn.Parameter(torch.tensor([theta_2]), requires_grad=True)
        self.theta_3 = nn.Parameter(torch.tensor([theta_3]), requires_grad=True)
    
    def forward(self, y_t: Tensor, t: int) -> Tensor:
        if t % 2 == 0:  # t is even
            return self.theta_0 + self.theta_1 * y_t
        else:  # t is odd
            return self.theta_2 + self.theta_3 * y_t

__all__ = ["FixedWeightM1P1LinearModule", "LearnableWeightM1P1LinearModule"]
