from typing import List

from torch import Tensor

from policies.base_policy import Policy, StochasticPolicy


class Agent:
    """
    Wrapper for policy that interacts with the environment.
    """

    def __init__(self, policy: Policy):
        self.policy: Policy = policy

    def action(self, y_history: List[Tensor]) -> Tensor:  # List[()] -> ()
        return self.policy.action(y_history)


class PolicyGradientAgent(Agent):
    """
    Wrapper for policy that interacts with the environment and can run policy gradient.
    """

    def __init__(self, policy: StochasticPolicy):
        super(PolicyGradientAgent, self).__init__(policy)

    def noisy_action(self, y_history: List[Tensor]) -> Tensor:  # List[()] -> ()
        return self.policy.noisy_action(y_history)

    def optimization_step(self,
                          y_histories: Tensor,  # (N, T)
                          u_histories: Tensor,  # (N, T)
                          l_histories: Tensor  # (N, T)
                          ):
        self.policy.optimization_step(y_histories=y_histories,
                                      u_histories=u_histories,
                                      l_histories=l_histories)


__all__ = ["Agent", "PolicyGradientAgent"]
