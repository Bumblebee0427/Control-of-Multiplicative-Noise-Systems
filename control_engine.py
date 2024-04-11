from tqdm import tqdm
from typing import List, Tuple

import torch
from torch import Tensor

from infrastructure.pytorch_utils import extract_copy_tensors_from_iterable
from infrastructure.visualization import plot_parameter_trajectory, plot_average_second_moment_trajectory
from environments.base_environment import Environment
from agent import Agent, PolicyGradientAgent


class ControlEngine:
    def __init__(
            self,
            agent: Agent,
            environment: Environment,
    ):
        self.agent: Agent = agent
        self.environment: Environment = environment

    def train_agent(self,
                    M: int,  # number of training iterations
                    N: int,  # number of trajectories sampled per training iteration
                    T: int,  # length of each trajectory,
                    ):
        """
        Trains agent using policy gradient and the REINFORCE algorithm.
        Then plots the theta trajectory.
        """
        assert isinstance(self.agent, PolicyGradientAgent)
        theta_history: List = []

        for i in tqdm(range(M)):
            _, y_histories, u_histories, l_histories = self.collect_trajectories(N, T, use_stochastic_policy=True)
            self.agent.optimization_step(y_histories=y_histories, u_histories=u_histories, l_histories=l_histories)
            theta_history.append(extract_copy_tensors_from_iterable(self.agent.policy.F.parameters()))

        theta_history: Tensor = torch.stack(tensors=theta_history)
        plot_parameter_trajectory(theta_history=theta_history)

    def evaluate_agent(self,
                       N: int,  # number of trajectories sampled to average over
                       T: int  # length of each trajectory
                       ):
        """
        Evaluates the given policy by plotting the average second moment trajectory for samples collected
        using this policy.
        """
        x_histories, y_histories, u_histories, r_histories = self.collect_trajectories(N, T, use_stochastic_policy=False)
        plot_average_second_moment_trajectory(x_histories)

    def collect_trajectories(self,
                             N: int,  # number of trajectories sampled to average over
                             T: int,  # length of each trajectory
                             use_stochastic_policy: bool  # whether to collect noisy actions
                             ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # [(N, T), (N, T), (N, T), (N, T)]
        """
        Collects a set of trajectories of a given length using the current policy.
        """
        with torch.no_grad():
            x_histories: List[Tensor] = []  # List[tensor of shape (T, )]
            y_histories: List[Tensor] = []  # List[tensor of shape (T, )]
            u_histories: List[Tensor] = []  # List[tensor of shape (T, )]
            l_histories: List[Tensor] = []  # List[tensor of shape (T, )]

            for j in range(N):
                self.environment.reset()

                x_history: List[Tensor] = []  # List[tensor of shape ()]
                y_history: List[Tensor] = []  # List[tensor of shape ()]
                u_history: List[Tensor] = []  # List[tensor of shape ()]
                l_history: List[Tensor] = []  # List[tensor of shape ()]

                for t in range(T):
                    x_t: Tensor = self.environment.oracle()  # ()
                    y_t: Tensor = self.environment.observe()  # ()

                    x_history.append(x_t)
                    y_history.append(y_t)

                    if use_stochastic_policy:
                        assert isinstance(self.agent, PolicyGradientAgent)
                        u_t = self.agent.noisy_action(y_history)  # ()
                    else:
                        u_t = self.agent.action(y_history)  # ()

                    u_history.append(u_t)

                    self.environment.step(u_t)

                    l_t: Tensor = self.environment.loss(u_t)
                    l_history.append(l_t)

                x_history: Tensor = torch.stack(tensors=x_history)  # (T, )
                y_history: Tensor = torch.stack(tensors=y_history)  # (T, )
                u_history: Tensor = torch.stack(tensors=u_history)  # (T, )
                l_history: Tensor = torch.stack(tensors=l_history)  # (T, )

                x_histories.append(x_history)
                y_histories.append(y_history)
                u_histories.append(u_history)
                l_histories.append(l_history)

            x_histories: Tensor = torch.stack(tensors=x_histories)  # (N, T)
            y_histories: Tensor = torch.stack(tensors=y_histories)  # (N, T)
            u_histories: Tensor = torch.stack(tensors=u_histories)  # (N, T)
            l_histories: Tensor = torch.stack(tensors=l_histories)  # (N, T)

            return x_histories, y_histories, u_histories, l_histories


__all__ = ["ControlEngine"]
