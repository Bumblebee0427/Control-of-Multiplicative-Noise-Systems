from pathlib import Path
import typing

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class Policy:
    """
    Abstract class that any policy should inherit.
    """
    def __init__(self, F: nn.Module):
        self.F: nn.Module = F

    def save(self, savefile_dir: Path):
        torch.save(self.F.state_dict(), savefile_dir / "F_weights.pt")

    def load(self, savefile_dir: Path):
        self.F.load_state_dict(torch.load(savefile_dir / "F_weights.pt"))

    def action(self, y_history: typing.List[Tensor]) -> Tensor:  # List[()] -> ()
        return self.F(y_history)


class StochasticPolicy(Policy):
    """
    Abstract class that any policy which is usable for policy gradient should inherit.
    """
    def __init__(self, F: nn.Module, F_optim: optim.Optimizer):
        super(StochasticPolicy, self).__init__(F)
        self.F_optim: optim.Optimizer = F_optim

    def save(self, savefile_dir: Path):
        super(StochasticPolicy, self).save(savefile_dir)
        torch.save(self.F_optim.state_dict(), savefile_dir / "F_optim_weights.pt")

    def load(self, savefile_dir: Path):
        super(StochasticPolicy, self).load(savefile_dir)
        self.F_optim.load_state_dict(torch.load(savefile_dir / "F_optim_weights.pt"))

    def noisy_action(self, y_history: typing.List[Tensor]) -> Tensor:  # List[()] -> ()
        raise NotImplementedError

    def logits(self, y_histories: Tensor, u_histories: Tensor) -> Tensor:  # (N, T), (N, T) -> (N, T)
        raise NotImplementedError

    def optimization_step(self,
                          y_histories: Tensor,  # (N, T)
                          u_histories: Tensor,  # (N, T)
                          l_histories: Tensor  # (N, T)
                          ):
        """
        Implements the REINFORCE algorithm given collected trajectories.
        """
        N, T = y_histories.shape[0], y_histories.shape[1]
        assert u_histories.shape[0] == N and u_histories.shape[1] == T
        assert l_histories.shape[0] == N and l_histories.shape[1] == T
        logits: Tensor = self.logits(y_histories=y_histories, u_histories=u_histories)  # (N, T)
        loss: Tensor = torch.mean(torch.sum(l_histories, dim=1) * torch.sum(logits, dim=1))  # ()
        self.F_optim.zero_grad()
        loss.backward()
        self.F_optim.step()


__all__ = ["Policy", "StochasticPolicy"]
