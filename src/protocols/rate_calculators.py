import numpy as np

# Constants from AdaptiveCurveIrm, scaled for floating-point math
WAD = 1e18
SECONDS_PER_YEAR = 365 * 24 * 60 * 60

# --- Replicating Solidity Math in NumPy ---

def w_mul_to_zero(a, b):
    # Solidity: (a * b) / WAD
    return (a * b) / WAD

def w_div_to_zero(a, b):
    # Solidity: (a * WAD) / b
    if b == 0:
        return 0
    return (a * WAD) / b

def _w_exp(x):
    # A simplified, vectorized version of the w_exp function for NumPy arrays.
    # The original uses fixed-point math and complex bit-shifting for precision.
    # For a distribution of forecasted inputs, np.exp is a reasonable approximation
    # of the behavior, as we are more interested in the resulting distribution shape.
    # x is linear_adaptation = speed * elapsed
    return np.exp(x / WAD)


def calculate_morpho_rate_stateless(
    utilization,
    current_time,
    last_update_time=0,
    start_rate_at_target=None,
    target_utilization=0.9,
    adjustment_speed_per_year=50.0,
    curve_steepness=4.0,
    initial_rate_at_target_per_year=0.04,
    min_rate_at_target_per_year=0.001,
    max_rate_at_target_per_year=2.0,
):
    """
    A stateless, vectorized implementation of the Morpho AdaptiveCurveIrm borrow_rate.
    """
    # Convert annual rates to per-second rates, scaled by WAD
    adjustment_speed = (adjustment_speed_per_year * WAD) / SECONDS_PER_YEAR
    initial_rate_at_target = (initial_rate_at_target_per_year * WAD) / SECONDS_PER_YEAR
    min_rate_at_target = (min_rate_at_target_per_year * WAD) / SECONDS_PER_YEAR
    max_rate_at_target = (max_rate_at_target_per_year * WAD) / SECONDS_PER_YEAR

    if start_rate_at_target is None:
        start_rate_at_target = initial_rate_at_target

    # --- Calculations from borrow_rate ---
    err_norm_factor = np.where(
        utilization > target_utilization,
        (1.0 - target_utilization),
        target_utilization,
    )
    err = (utilization - target_utilization) / err_norm_factor

    speed = adjustment_speed * err
    elapsed = current_time - last_update_time
    linear_adaptation = speed * elapsed

    # --- From _new_rate_at_target ---
    exp_adaptation = _w_exp(linear_adaptation)
    new_rate = w_mul_to_zero(start_rate_at_target, exp_adaptation)
    end_rate_at_target = np.clip(new_rate, min_rate_at_target, max_rate_at_target)

    # Simplified avg_rate_at_target for forecasting (using end rate)
    avg_rate_at_target = end_rate_at_target

    # --- From _curve ---
    coeff = np.where(
        err < 0,
        1.0 - (1.0 / curve_steepness),
        curve_steepness - 1.0,
    )
    rate_per_second = w_mul_to_zero(
        w_mul_to_zero(coeff, err) + 1.0, avg_rate_at_target
    )

    # Convert back to an annualized percentage rate
    annual_rate_pct = (rate_per_second * SECONDS_PER_YEAR * 100)
    return annual_rate_pct


def calculate_morpho_rate(utilization, kink=0.8, base=0.0, slope1=0.04, slope2=1.00):
    """
    (This function is now deprecated in favor of the more accurate stateless model but kept for reference)
    Calculates the borrow interest rate based on the Morpho Blue model.
    Vectorized to handle arrays of samples.
    """
    # If U < kink: rate = base + (U / kink) * slope1
    # If U > kink: rate = base + slope1 + ((U - kink) / (1 - kink)) * slope2
    rates = np.where(
        utilization <= kink,
        base + (utilization / kink) * slope1,
        base + slope1 + ((utilization - kink) / (1 - kink)) * slope2
    )
    return rates

