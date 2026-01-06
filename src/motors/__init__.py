from typing import Callable

import numpy as np
import pyro
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
from pyro.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Normal,
    Uniform,
)
import scipy.stats as st
import torch


def grid_approx(
    trials: int,
    num_successes: int,
    prior: np.ndarray = None,
    num_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    # Define the grid
    p_grid = np.linspace(0, 1, num_points)

    # Define the prior distribution
    _prior = prior or np.ones(num_points)

    # Define the liklihood at each p; the (conditional) PDF
    liklihood = st.binom(n=trials, p=p_grid).pmf(num_successes)

    # Compute un-standardized posterior
    posterior = liklihood * _prior

    # Standardize posterior; divide PDF by expectation
    posterior /= sum(posterior)

    return p_grid, posterior


def get_map_using_grad(
    model: Callable,
    weights: list[float],
    iters: int = 1000,
    learning_rate: float = 0.01,
) -> tuple[float, np.ndarray]:
    """
    Finds the Maximum a Posteriori (MAP) estimate using manual gradient ascent.

    Args:
        model (Callable): A pyro model that takes a parameter tensor `p`
                         as its first argument, followed by other arguments (`weights`).
        weights (list[float]): Additional arguments to the model.
        iters (int): Number of optimization iterations.
        learning_rate (float): The learning rate for gradient ascent.

    Returns:
        A tuple containing the MAP estimate and the history of estimates.
    """
    # Initial guess for p, requires gradient for optimization
    p = torch.tensor(0.5, requires_grad=True)
    values = []

    for _ in range(iters):
        # A fresh trace is needed for each gradient calculation
        traced_model = pyro.poutine.trace(model)
        # Assumes the model's first argument is the parameter to optimize
        log_L = traced_model.get_trace(p, *weights).log_prob_sum()
        log_L.backward()

        # Manual gradient ascent step
        p.data += learning_rate * p.grad
        p.grad.zero_()
        values.append(p.item())

    # MAP estimate for p and p's distribution
    return p.detach().item(), np.array(values)


def get_quad_approx(
    model: Callable,
    weights: list[float],
    iters: int = 1000,
) -> tuple[float, float, np.ndarray]:
    """
    Calculates the quadratic approximation of a posterior distribution.

    Args:
        model (Callable): The pyro model, with the parameter to be optimized
                         as the first argument.
        weights (list[float]): Arguments to the model.
        iters (int): Number of optimization iterations.

    Returns:
        A tuple containing the mean (mu), standard deviation (std_dev),
        and the history of estimates during MAP optimization.
    """
    # 1. Find the Maximum a Posteriori (MAP) estimate for p
    mu_val, estimates = get_map_using_grad(model, weights, iters)
    mu_tensor = torch.tensor(mu_val, requires_grad=True)

    # 2. Calculate the curvature at the MAP to get the standard deviation
    traced_model = pyro.poutine.trace(model)
    log_L_at_map = traced_model.get_trace(mu_tensor, *weights).log_prob_sum()

    # Calculate the second-derivative to find the curvature/std_dev
    first_derivative = torch.autograd.grad(log_L_at_map, mu_tensor, create_graph=True)[0]
    second_derivative = torch.autograd.grad(first_derivative, mu_tensor)[0]

    std_dev = (1 / torch.sqrt(-second_derivative)).item()

    return mu_val, std_dev, estimates


def find_map_with_svi(
    model: Callable,
    weights: list[float],
    iters: int = 1000,
) -> torch.Tensor:
    """
    Finds the Maximum a Posteriori (MAP) estimate using Pyro's SVI with an AutoDelta guide.
    This is the recommended approach for finding point estimates in Pyro.
    Note: This function does not return the history of estimates during optimization.

    Args:
        model (Callable): The pyro model.
        weights (list[float]): Arguments to the model.
        iters (int): Number of optimization iterations.

    Returns:
        torch.Tensor: The MAP estimate for the parameter 'p'.
    """
    pyro.clear_param_store()
    guide = AutoDelta(model)
    optimizer = optim.Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for _ in range(iters):
        svi.step(*weights)

    # The guide contains the MAP estimates of the latent variables
    map_estimates = guide()
    return map_estimates['p']

