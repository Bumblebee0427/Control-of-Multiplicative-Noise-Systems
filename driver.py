import torch

from environments.multiplicative_gaussian_noise_environment import *
from policies.base_policy import *
from policies.additive_gaussian_policy import *
from policies.m1p1_linear_policy_modules import *
from agent import *
from control_engine import *

M = 100
N_train = 1000
N_eval = 100
T_train = 10
T_eval = 100
a = 2
b = 1.0
c = 1.0
mu = 1.0
alpha = 0.0
beta = 0.0
gamma = 1.0
sigma = 0.0
lmbda = 0.0
omega = 1.0
lr = 1e-1

env = MultiplicativeGaussianNoiseEnvironment(a, b, c, mu, alpha, beta, gamma, sigma, lmbda)
U = LearnableWeightM1P1LinearModule()
U_optim = torch.optim.Adam(U.parameters(), lr=lr)  # Why are we using Adam instead of SGD? See below
policy = AdditiveGaussianNoiseStochasticPolicy(U, U_optim, omega)
agent = PolicyGradientAgent(policy)
engine = ControlEngine(agent, env)
engine.train_agent(M, N_train, T_train)
engine.evaluate_agent(N_eval, T_eval)

"""
Debugging tip: if your trajectory diverges, first re-run a few times -- especially on the "edge of stability", 
you may have had bad draws of random noise that screwed up your gradients. If this doesn't help,
try modifying your learning rate or increasing M. Your learning rate could either be too high, 
which has the usual effects of high learning rate that we saw when analyzing convergence of gradient descent 
-- or it could be too low, in which case theta could not converge, and so either you should raise the learning 
rate or raise the number of training iterations.

Why do we use Adam instead of SGD? Something really interesting happens with SGD. The gradient at the first
iteration is likely to be large if the best policy isn't just F_t(Y_(t)) = 0, and so your first theta iterate 
will also be really large. This means that the gradient at the second iteration will be gigantic, and when you update
theta, you will get numerical overflow. If you set your learning rate to make the first 
theta iterate of a reasonable size, then in SGD the learning rate will be too low to move your theta iterate 
significantly in subsequent iterates, and so you will not converge to a good policy in reasonable time. This 
could be fixed by initializing at a "good" policy, but that's cheating :< especially since we don't always know 
good policies (if we did, then what's the point of learning them?)

Instead, we use Adam, which essentially uses adaptive learning rates and thus bypasses the issue of 
needing different scales of learning rates at different times. For more information about Adam,
see here: https://ruder.io/optimizing-gradient-descent/index.html#adam
"""