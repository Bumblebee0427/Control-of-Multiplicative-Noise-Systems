import typing

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import ndarray
from torch import Tensor

from infrastructure.pytorch_utils import to_numpy


def plot_average_second_moment_trajectory(x_histories: Tensor, a_value: float = None):  # (N, T)
    if torch.isnan(x_histories).any():
        print("X trajectory **diverged**; cannot visualize.")
        return
    x_sq_histories: Tensor = x_histories ** 2  # (N, T)
    mean_x_sq_history: Tensor = torch.mean(x_sq_histories, dim=0)  # (T, )
    mean_x_sq_history: ndarray = to_numpy(mean_x_sq_history)  # (T, )
    plt.plot(mean_x_sq_history, '-o')
    plt.title("$\mathbb{E}[X_t^2]$" + (f", a={a_value}" if a_value else ""))
    plt.yscale("log")
    plt.xlabel("$t$")
    plt.show()
    plt.close()


def plot_parameter_trajectory(theta_history: Tensor):  # (M, N_params)
    if torch.isnan(theta_history).any():
        print("Theta trajectory **diverged**; cannot visualize.")
        return
    M: int = theta_history.shape[0]
    N_p: int = theta_history.shape[1]
    for j in range(N_p):
        param_j_history: Tensor = theta_history[:, j]  # (M, )
        param_j_history: ndarray = to_numpy(param_j_history)  # (M, )
        plt.plot(param_j_history, '-o')
    plt.title("$\\theta$ parameters as function of # optimization iterations $i$")
    plt.xlabel("$i$")
    plt.show()
    plt.close()


def plot_empirical_average_second_moment_and_partial_loss(states: Tensor, losses: Tensor, env_type: str):
    states = np.array(states)
    losses = np.array(losses)

    mean_second_moments = np.power(states, 2).mean(axis=0)
    mean_cumsum_losses = losses.cumsum(axis=1).mean(0)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9,12))

    ax[0].plot(mean_second_moments)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\mathbb{{E}}[X_t^2]$')
    ax[0].set_title(f'Empirical $\mathbb{{E}}[X_t^2]$ for {env_type} System')

    ax[1].plot(mean_cumsum_losses)
    ax[1].set_xlabel(r'$\tilde{t}$')
    ax[1].set_ylabel(r'$L(F; \tilde{t})$')
    ax[1].set_title(f'Empirical Partial Loss for {env_type} System')
    plt.show()


def plot_empirical_loss_over_a(a_list: typing.List, losses: typing.List):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(a_list, losses)
    ax.set_yscale('log')
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$\overline{L}(F)$')
    ax.set_title('Empirical Loss for Optimal Policy')
    plt.show()


__all__ = ["plot_average_second_moment_trajectory", "plot_parameter_trajectory",
           "plot_empirical_average_second_moment_and_partial_loss", "plot_empirical_loss_over_a"]
