import numpy as np
import pymc as pm


def generate_ou_paths(current_price, dt, steps, n_paths, theta, mu, sigma):
    """Generates paths based on the discretized OU formula."""
    paths = np.zeros((steps, n_paths))
    paths[0] = current_price

    for t in range(1, steps):
        # The OU Discretized Step
        drift = paths[t - 1] * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
        diffusion = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
        paths[t] = drift + diffusion * np.random.normal(size=n_paths)

    return paths


def generate_ar_paths(
    current_price,
    dt,
    steps,
    n_paths,
    phi,
    m_log,
    sigma,
    p_jump=0.0,
    mu_jump=0.0,
    sigma_jump=0.0,
):
    x = np.zeros((steps, n_paths))
    x[0] = np.log(current_price)
    rng = np.random.default_rng(73)

    if p_jump > 0.0:
        do_jump = rng.random(n_paths) < p_jump
        t_jump = np.where(
            do_jump & (steps > 0), rng.integers(0, steps, size=n_paths), -1
        )
        J = np.where(do_jump, rng.normal(mu_jump, sigma_jump, size=n_paths), 0.0)
    else:
        t_jump = np.full(n_paths, -1)
        J = np.zeros(n_paths)

    # Pre-generate noise for speed
    diffusion_noise = rng.normal(0, sigma, (steps, n_paths))

    for t in range(1, steps):
        # 1. Standard AR(1) Step (Diffusion)
        # x_t = m + phi * (x_{t-1} - m) + noise
        x[t] = m_log + phi * (x[t - 1] - m_log) + diffusion_noise[t]

        # 2. Optional Jump Component (Poisson/Bernoulli style)
        mask = t == t_jump
        if mask.any():
            x[t, mask] += J[mask]

    # 3. Convert from Log-Space back to Price-Space
    price_paths = np.exp(x)

    return price_paths


def get_jump_prior_stats(prices):
    log_returns = np.diff(np.log(prices))

    # Use MAD for a robust estimate of 'normal' volatility
    # This prevents the jumps from inflating the 'normal' sigma
    median = np.median(log_returns)
    mad = np.median(np.abs(log_returns - median))
    robust_sigma = 1.4826 * mad

    # Identify jumps (outliers)
    outliers = log_returns[np.abs(log_returns) > 1 * robust_sigma]

    prior_p = len(outliers) / len(log_returns)
    prior_mu = np.percentile(outliers, 0.85) if len(outliers) > 0 else 0
    prior_sigma = np.std(outliers) if len(outliers) > 0 else robust_sigma * 2

    return prior_p, prior_mu, prior_sigma


def estimate_ou_params(observed_prices, n_samples, dt):
    with pm.Model() as robust_ou_model:  # noqa: F841
        # Priors
        theta = pm.Exponential("theta", lam=0.95)
        mu = pm.Normal(
            "mu", mu=np.median(observed_prices), sigma=np.std(observed_prices)
        )
        sigma = pm.HalfStudentT(
            "sigma", nu=3, sigma=np.std(np.diff(np.log(observed_prices)))
        )

        # Degrees of freedom for Student-T (nu)
        # Small nu (e.g., 3) allows for fat tails/outliers
        nu = pm.Exponential("nu", lam=0.01) + 2

        drift = observed_prices[:-1] * pm.math.exp(-theta * dt) + mu * (
            1 - pm.math.exp(-theta * dt)
        )

        # Likelihood using Student-T instead of Normal
        obs = pm.StudentT(  # noqa: F841
            "obs", nu=nu, mu=drift, sigma=sigma, observed=observed_prices[1:]
        )

        trace = pm.sample(n_samples, tune=2000, target_accept=0.95)
        return trace


def estimate_ar1_params(
    observed_log_prices, p_est, mu_est, sigma_est, n_samples, set_jump=True
):
    with pm.Model() as ar1_model:  # noqa: F841
        # --- Priors ---
        # m: The long-term log-mean
        m_mu = np.percentile(observed_log_prices, 0.5)
        m_sigma = np.std(observed_log_prices)
        m = pm.Normal(
            "m",
            mu=m_mu,
            sigma=m_sigma,
        )

        # phi: Persistence (must be between 0 and 1 for mean reversion)
        # We use a Beta distribution because it is naturally bounded [0, 1]
        phi = pm.Beta("phi", alpha=4, beta=4)

        # sigma: The noise level (sigma_bps / 10,000)
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(observed_log_prices))

        # --- Likelihood ---
        # The 'mu' of our normal distribution follows the AR(1) logic
        # x_t = m + phi * (x_{t-1} - m)
        x_prev = observed_log_prices[:-1]
        mu_t = m + phi * (x_prev - m)

        if set_jump:
            # Jump Priors
            p_jump = pm.Beta(
                "p_jump", alpha=1, beta=1 / p_est if p_est > 0 else 1e2
            )  # Prior: jumps are rare
            mu_jump = pm.Normal("mu_jump", mu=mu_est, sigma=0.1)
            sigma_jump = pm.Exponential("sigma_jump", lam=1 / sigma_est)

            # Components
            # 0: Standard move
            # 1: Standard move + Jump
            comp_dists = [
                pm.Normal.dist(mu=mu_t, sigma=sigma),
                pm.Normal.dist(
                    mu=mu_t + mu_jump, sigma=pm.math.sqrt(sigma**2 + sigma_jump**2)
                ),
            ]

            # Likelihood
            pm.Mixture(
                "obs",
                w=[1 - p_jump, p_jump],
                comp_dists=comp_dists,
                observed=observed_log_prices[1:],
            )
        else:
            # Observed data (x_t)
            obs = pm.Normal(  # noqa: F841
                "obs", mu=mu_t, sigma=sigma, observed=observed_log_prices[1:]
            )

            # --- Inference ---
        trace = pm.sample(n_samples, tune=1000, target_accept=0.95)

    return trace, ar1_model, m_mu, m_sigma
