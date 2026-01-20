import numpy as np

def derive_jump_parameters(prices: np.ndarray):
    """
    Estimates jump probability, mean, and standard deviation from a price series.
    """
    log_returns = np.diff(np.log(prices))

    median = np.median(log_returns)
    mad = np.median(np.abs(log_returns - median))
    robust_sigma = 1.4826 * mad

    outliers = log_returns[np.abs(log_returns) > 3 * robust_sigma]

    p_jump = len(outliers) / len(log_returns) if len(log_returns) > 0 else 0
    mu_jump = np.mean(outliers) if len(outliers) > 0 else 0
    sigma_jump = np.std(outliers) if len(outliers) > 0 else 0
    
    return {"p_jump": p_jump, "mu_jump": mu_jump, "sigma_jump": sigma_jump}
