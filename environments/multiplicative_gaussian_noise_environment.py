import typing

import torch
from torch import Tensor

from environments.base_environment import Environment


class MultiplicativeGaussianNoiseEnvironment(Environment):
    """
    Defines the transition dynamics of the environment with multiplicative noise according to the equations:
    x_(t+1) = A_t x_t + B_t u_t
    y_t     = C_t x_t
    l_t     = x_(t+1)^2 + lambda * u_t
    where A_t ~ N(a, alpha^2), B_t ~ N(b, beta^2), C_t ~ N(c, gamma^2), and X_0 ~ N(mu, sigma^2).
    """
    def __init__(self, a: float, b: float, c: float, mu: float, alpha: float, beta: float, gamma: float,
                 sigma: float, lmbda: float):
        super(MultiplicativeGaussianNoiseEnvironment, self).__init__()
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.mu: float = mu
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.sigma: float = sigma
        self.lmbda: float = lmbda

        self.x: Tensor = self.mu + self.sigma * torch.randn(size=())

        C_0: Tensor = self.c + self.gamma * torch.randn(size=())
        self.y: Tensor = C_0 * self.x

    def step(self, u: Tensor):
        """
        Given an input u, executes the transition dynamics once.
        """
        A_t: Tensor = self.a + self.alpha * torch.randn(size=())
        B_t: Tensor = self.b + self.beta * torch.randn(size=())
        C_t: Tensor = self.c + self.gamma * torch.randn(size=())
        self.y: Tensor = C_t * self.x  # TODO
        self.x: Tensor = A_t * self.x + B_t * u # TODO

    def loss(self, u: Tensor) -> Tensor:
        """
        Given an input u, computes the loss l = x^2 + lambda u^2.
        """
        return (self.x ** 2) + self.lmbda * (u ** 2)

    def reset(self):
        """
        Resets the state and observation to their initial distributions.
        """
        self.x: Tensor = self.mu + self.sigma * torch.randn(size=()) # TODO

        C_0: Tensor = self.c + self.gamma * torch.randn(size=()) #TODO
        self.y: Tensor = C_0 * self.x # TODO


class MultiplicativeGaussianControlNoiseEnvironment(MultiplicativeGaussianNoiseEnvironment):
    """
    Defines the transition dynamics of the environment with multiplicative input noise according to the equations:
    x_(t+1) = a x_t + B_t u_t
    y_t     = x_t
    l_t     = x_(t+1)^2 + lambda * u_t
    where B_t ~ N(b, beta^2) and X_0 = 1.
    """
    def __init__(self, a: float, b: float, beta: float, lmbda: float):
        super(MultiplicativeGaussianControlNoiseEnvironment, self).__init__(
            a=a, b=b, c=1.0, mu=1.0, alpha=0.0, beta=beta, gamma=0.0, sigma=0.0, lmbda=lmbda
        )


class MultiplicativeGaussianStateControlNoiseEnvironment(MultiplicativeGaussianNoiseEnvironment):
    """
    Defines the transition dynamics of the environment with multiplicative state and input noise according to the equations:
    x_(t+1) = A_t x_t + B_t u_t
    y_t     = x_t
    l_t     = x_(t+1)^2 + lambda * u_t
    where A_t ~ N(a, alpha^2), B_t ~ N(b, beta^2), and X_0 = 1.
    """
    def __init__(self, a: float, b: float, alpha: float, beta: float, lmbda: float):
        super(MultiplicativeGaussianStateControlNoiseEnvironment, self).__init__(
            a=a, b=b, c=1.0, mu=1.0, alpha=alpha, beta=beta, gamma=0.0, sigma=0.0, lmbda=lmbda
        )


class MultiplicativeGaussianObservationNoiseEnvironment(MultiplicativeGaussianNoiseEnvironment):
    """
    Defines the transition dynamics of the environment with multiplicative observation noise according to the equations:
    x_(t+1) = a x_t + u_t
    y_t     = C_t x_t
    l_t     = x_(t+1)^2 + lambda * u_t
    where C_t ~ N(c, gamma^2) and X_0 = 1.
    """
    def __init__(self, a: float, c: float, gamma: float, lmbda: float):
        super(MultiplicativeGaussianObservationNoiseEnvironment, self).__init__(
            a=a, b=1.0, c=c, mu=1.0, alpha=0.0, beta=0.0, gamma=gamma, sigma=0.0, lmbda=lmbda
        )


__all__ = [
    "MultiplicativeGaussianNoiseEnvironment", "MultiplicativeGaussianControlNoiseEnvironment",
    "MultiplicativeGaussianStateControlNoiseEnvironment", "MultiplicativeGaussianObservationNoiseEnvironment"
]
