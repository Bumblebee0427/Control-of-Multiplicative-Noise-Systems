import math
import typing

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from policies.base_policy import StochasticPolicy


class AdditiveGaussianNoiseStochasticPolicy(StochasticPolicy):
    """
    Stochastic policy which implements the additive Gaussian noise stochasticity u_t = F(y_(t)) + N(0, omega^2).
    """
    def __init__(self, F: nn.Module, F_optim: optim.Optimizer, omega: float):
        super(AdditiveGaussianNoiseStochasticPolicy, self).__init__(F, F_optim)
        self.omega: float = omega

    def noisy_action(self, y_history: typing.List[Tensor]) -> Tensor:  # List[()] -> ()
        """
        Given Y_(t), computes U_t = F_t(Y_(t)) then adds noise W_t to get \tilde{U}_t.
        """
        u_t: Tensor = self.F(y_history)
        w_t: Tensor = self.omega * torch.randn(size=())
        return u_t + w_t

    def logits(self, y_histories: Tensor, u_histories: Tensor) -> Tensor:  # (N, T), (N, T) -> ()
        """
        Given an observation trajectory [Y_0, ..., Y_T-1] and the noisy actions [\\tilde{U}_0, ..., \\tilde{U}_T-1]
        where \\tilde{U}_t = F_t(Y_(t)) + N(0, omega^2), computes the log probabilities [q_1, ..., q_T] where
        q_t = log(pi_theta(\\tilde{U}_t | Y_(t))).

        For convenience, computes these log probabilities over size N _batches_ of disjoint trajectories.

        Hint: You can compute log(pi_theta) in closed form in terms of \\tilde{U}_t and Y_(t).
        """
        N, T = y_histories.shape[0], y_histories.shape[1]
        assert u_histories.shape[0] == N and u_histories.shape[1] == T

        mu_histories: Tensor = torch.zeros(size=(N, T))
        for i in range(N):
            for t in range(T):
                mu_histories[i][t] = self.F(list(y_histories[i][:t + 1]))
        logits: Tensor = - 1/2 * math.log(2 * torch.pi * (self.omega ** 2)) \
                         - ((u_histories - mu_histories) ** 2) / (2 * (self.omega ** 2))
        return logits


__all__ = ["AdditiveGaussianNoiseStochasticPolicy"]
