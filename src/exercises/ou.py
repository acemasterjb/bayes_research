import argparse
import asyncio
import os
from datetime import datetime, timedelta
import arviz as az
from arch import arch_model
import scipy.stats as stats
from src.motors.garch_model import run_garch_model, forecast_garch_model
from src.motors.ewma_model import run_ewma_model
from src.motors.gbm_model import run_gbm_model, generate_gbm_price_paths
from src.plotting.garch_plots import (
    plot_garch_posteriors,
    plot_garch_volatility,
    plot_garch_forecast,
    plot_garch_in_sample_fit,
    plot_garch_ppc,
    plot_garch_simple_residuals,
    plot_garch_standardized_residuals_and_acf,
)
from src.plotting.ewma_plots import (
    plot_ewma_posteriors,
    plot_ewma_volatility,
    plot_ewma_ppc,
    plot_ewma_standardized_residuals_and_acf,
    plot_ewma_in_sample_fit,
)
from src.plotting.gbm_plots import (
    plot_gbm_posteriors,
    plot_gbm_in_sample_fit,
    plot_gbm_ppc,
    plot_gbm_standardized_residuals_and_acf,
)


import cloudpickle
import numpy as np
import pandas as pd

from src.mobula.client import MobulaClient
from src.motors.ou_model import (
    estimate_ar1_params,
    get_jump_prior_stats,
    generate_ar_paths,
)
from src.plotting.ou_plots import (
    plot_posterior,
    plot_ar1_in_sample_fit,
    plot_priors_vs_posteriors,
    plot_posterior_predictive_check,
    plot_residuals,
)

from src.validation.stationarity_tests import run_stationarity_analysis
from src.validation.risk_metrics import (
    get_ar1_risk_distribution,
    get_garch_risk_distribution,
)
from src.plotting.risk_plots import plot_risk_distributions
import plotly.graph_objects as go


def get_cached_model_results(cache_path):
    """
    Loads model results from a cache file if it exists.
    """
    if os.path.exists(cache_path):
        print("Loading model results from cache...")
        with open(cache_path, "rb") as f:
            return cloudpickle.load(f)
    return None


def cache_model_results(cache_path, results):
    """
    Saves model results to a cache file.
    """
    with open(cache_path, "wb") as f:
        cloudpickle.dump(results, f)


def compare_arch_and_bayesian(trace, log_returns):
    # --- 1. ARCH Point Estimate ---
    arch_m = arch_model(log_returns, vol="Garch", p=1, q=1, dist="t")
    res = arch_m.fit(disp="off")
    print("\\n--- ARCH vs. Bayesian GARCH Results ---")
    print(res.summary())

    # --- 2. Bayesian Parameter Estimates ---
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    bayesian_params = {
        "mu": posterior["mu"].mean().item(),
        "omega": posterior["omega"].mean().item(),
        "alpha": posterior["alpha"].mean().item(),
        "beta": posterior["beta"].mean().item(),
    }

    print("\\n--- Bayesian Mean Posterior Estimates ---")
    for param, value in bayesian_params.items():
        hdi = np.array(az.hdi(trace.posterior[param], hdi_prob=0.95)[param])
        print(f"  - {param}: {value:.6f} (95% HDI: [{hdi[0]}, {hdi[1]}])")

    # --- 3. Residual Analysis ---
    # Get standardized residuals from ARCH
    arch_std_resid = res.std_resid

    # Calculate standardized residuals from Bayesian model
    mu_mean = bayesian_params["mu"]
    omega_mean = bayesian_params["omega"]
    alpha_mean = bayesian_params["alpha"]
    beta_mean = bayesian_params["beta"]
    initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

    vol_sq_series = np.zeros(len(log_returns))
    vol_sq_series[0] = initial_vol_sq_mean
    for t in range(1, len(log_returns)):
        err_tm1 = log_returns[t - 1] - mu_mean
        vol_sq_series[t] = (
            omega_mean + alpha_mean * err_tm1**2 + beta_mean * vol_sq_series[t - 1]
        )

    bayesian_std_resid = (log_returns - mu_mean) / np.sqrt(vol_sq_series)

    # --- 4. Plotting Residuals ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=arch_std_resid, mode="lines", name="ARCH Std. Residuals")
    )
    fig.add_trace(
        go.Scatter(
            y=bayesian_std_resid,
            mode="lines",
            name="Bayesian Std. Residuals",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Standardized Residuals Comparison",
        xaxis_title="Time",
        yaxis_title="Std. Residual",
    )
    fig.show()

    # --- 5. QQ Plots ---
    fig_qq = go.Figure()

    # ARCH QQ Plot
    qq_arch = stats.probplot(arch_std_resid, dist="t", sparams=(20,))
    fig_qq.add_trace(
        go.Scatter(
            x=qq_arch[0][0], y=qq_arch[0][1], mode="markers", name="ARCH Residuals"
        )
    )

    # Bayesian QQ Plot
    qq_bayesian = stats.probplot(bayesian_std_resid, dist="t", sparams=(20,))
    fig_qq.add_trace(
        go.Scatter(
            x=qq_bayesian[0][0],
            y=qq_bayesian[0][1],
            mode="markers",
            name="Bayesian Residuals",
        )
    )

    # Add reference line
    x = np.linspace(
        min(qq_arch[0][0].min(), qq_bayesian[0][0].min()),
        max(qq_arch[0][0].max(), qq_bayesian[0][0].max()),
        100,
    )
    fig_qq.add_trace(
        go.Scatter(
            x=x, y=x, mode="lines", name="Normal", line=dict(color="black", dash="dash")
        )
    )

    fig_qq.update_layout(
        title="QQ Plot of Standardized Residuals vs. Normal Distribution",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    fig_qq.show()


async def run(
    asset_address: str,
    chain_id: str,
    percentile: float = 0.5,
    current_price: float = 1.0,
    n_paths: int = 10,
    plot_jumps: bool = False,
    run_stationarity_tests: bool = False,
    test_returns: bool = False,
    model_type: str = "ar1",
    forecast_days: int = 0,
    trim_lower: float = 0.1,
    trim_upper: float = 0.9,
    run_risk_analysis: bool = False,
    sim_days: int = 30,
    sim_paths: int = 1000,
):
    """
    Main function to run the Ornstein-Uhlenbeck process analysis.
    """
    # --- Caching Setup ---
    today = datetime.now().strftime("%Y-%m-%d")
    cache_dir = "./data"
    os.makedirs(cache_dir, exist_ok=True)

    prices_cache_path = os.path.join(
        cache_dir, f"prices_{chain_id}_{asset_address}_{today}.npy"
    )
    results_cache_path = os.path.join(
        cache_dir, f"model_results_{chain_id}_{asset_address}_{today}.pkl"
    )

    # --- Step 1: Load or Fetch Prices ---
    if os.path.exists(prices_cache_path):
        print("Loading prices from cache...")
        prices = np.load(prices_cache_path)
    else:
        print("Fetching prices from Mobula API...")
        mobula_client = MobulaClient()
        to_timestamp = int(datetime.now().timestamp() * 1000)
        from_timestamp = int((datetime.now() - timedelta(days=334)).timestamp() * 1000)
        market_history = await mobula_client.get_market_history(
            asset_address=asset_address,
            chain_id=chain_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        prices = np.array([item[1] for item in market_history["data"]["priceHistory"]])

        # Handle NaN values and save to cache
        prices_series = pd.Series(prices)
        prices_series.interpolate(method="linear", inplace=True)
        prices_series.bfill(inplace=True)
        prices_series.ffill(inplace=True)
        prices = prices_series.to_numpy()
        np.save(prices_cache_path, prices)

    if np.isnan(prices).all():
        raise ValueError("All price data is NaN after cleaning. Cannot proceed.")

    if run_stationarity_tests:
        run_stationarity_analysis(prices, use_returns=test_returns)
        return

    # --- Step 2: Model Selection and Execution ---
    model_cache_path = os.path.join(
        cache_dir, f"{model_type}_model_results_{chain_id}_{asset_address}_{today}.pkl"
    )
    cached_results = get_cached_model_results(model_cache_path)

    if cached_results:
        trace = cached_results["trace"]
        model = cached_results["model"]
    else:
        print(f"Running MCMC simulation for {model_type.upper()} model...")
        log_returns = np.diff(np.log(prices))
        if model_type.lower() == "ewma":
            trace, model = run_ewma_model(log_returns)
        elif model_type.lower() == "gbm":
            trace, model = run_gbm_model(prices)
        elif model_type.lower() == "garch":
            trace, model = run_garch_model(log_returns)
        elif model_type.lower() == "ar1":
            trace, model, m_mu, m_sigma = estimate_ar1_params(
                np.log(prices), 0, 0, 0, int(1e4), False
            )
            cached_results = {
                "trace": trace,
                "model": model,
                "m_mu": m_mu,
                "m_sigma": m_sigma,
            }

        if model_type.lower() != "ar1":
            cached_results = {"trace": trace, "model": model}

        cache_model_results(model_cache_path, cached_results)

    if model_type.lower() == "ewma":
        print("Visualizing EWMA results...")
        log_returns = np.diff(np.log(prices))
        plot_ewma_posteriors(trace)
        plot_ewma_volatility(trace, log_returns)
        plot_ewma_ppc(model, trace)
        plot_ewma_in_sample_fit(trace, prices)
        plot_ewma_standardized_residuals_and_acf(trace, log_returns)
        return

    if model_type.lower() == "gbm":
        print("Visualizing GBM results...")
        plot_gbm_posteriors(trace)
        plot_gbm_in_sample_fit(trace, prices)
        plot_gbm_ppc(model, trace)
        plot_gbm_standardized_residuals_and_acf(trace, prices)
        return

    if model_type.lower() == "garch":
        log_returns = np.diff(np.log(prices))
        p_est, mu_est, sigma_est = get_jump_prior_stats(prices)

        print("Visualizing GARCH results...")
        # plot_garch_posteriors(trace)
        # plot_garch_volatility(trace, log_returns)
        # plot_garch_in_sample_fit(
        #     trace,
        #     prices,
        #     p_est,
        #     mu_est,
        #     sigma_est,
        #     percentile=percentile,
        #     trim_percentile_lower=trim_lower,
        #     trim_percentile_upper=trim_upper,
        # )
        # plot_garch_ppc(model, trace)
        # plot_garch_simple_residuals(trace, log_returns)
        # plot_garch_standardized_residuals_and_acf(trace, log_returns)

        # --- Compare with ARCH ---
        compare_arch_and_bayesian(trace, log_returns)

        # Get optimal parameters
        omega_opt = np.quantile(trace.posterior["omega"], percentile)
        alpha_opt = np.quantile(trace.posterior["alpha"], percentile)
        beta_opt = np.quantile(trace.posterior["beta"], percentile)

        print("Optimal OU Process Parameters:")
        print(f"    Omega: {omega_opt}")
        print(f"    Alpha: {alpha_opt}")
        print(f"    Beta: {beta_opt}")
        print("Estimated Jump Params:")
        print(f"    p_jump: {p_est}, mu: {mu_est}, sigma: {sigma_est}")

        # --- GARCH Forecasting ---
        if forecast_days > 0:
            print(f"\\n--- Generating GARCH Forecast for {forecast_days} days ---")
            # We need the last return and the last inferred volatility to start the forecast
            posterior = trace.posterior.stack(sample=("chain", "draw"))
            mu_mean = posterior["mu"].mean().item()
            omega_mean = posterior["omega"].mean().item()
            alpha_mean = posterior["alpha"].mean().item()
            beta_mean = posterior["beta"].mean().item()
            initial_vol_sq_mean = posterior["initial_vol_sq"].mean().item()

            # Reconstruct the last volatility
            vol_sq_series = np.zeros(len(log_returns))
            vol_sq_series[0] = initial_vol_sq_mean
            for t in range(1, len(log_returns)):
                err_tm1 = log_returns[t - 1] - mu_mean
                vol_sq_series[t] = (
                    omega_mean
                    + alpha_mean * err_tm1**2
                    + beta_mean * vol_sq_series[t - 1]
                )

            last_vol_sq = vol_sq_series[-1]
            last_return = log_returns[-1]

            forecasted_returns = forecast_garch_model(
                trace, last_return, last_vol_sq, n_forecast_steps=forecast_days
            )
            # plot_garch_forecast(prices, forecasted_returns)

        # --- GARCH Risk Analysis ---
        if run_risk_analysis:
            print("\\n--- Running Bayesian Risk Analysis for GARCH Model ---")
            var_dist, cvar_dist = get_garch_risk_distribution(
                trace,
                returns=log_returns,
                s0=prices[-1],
                T=sim_days,
                N_per_sample=sim_paths,
                p_jump=p_est,
                mu_jump=mu_est,
                sigma_jump=sigma_est,
                alpha=0.05,
            )

            print(f"  - Mean VaR (95%): {np.mean(var_dist):.2%}")
            print(f"  - Std Dev of VaR: {np.std(var_dist):.2%}")
            print(f"  - Mean CVaR (95%): {np.mean(cvar_dist):.2%}")
            print(f"  - Std Dev of CVaR: {np.std(cvar_dist):.2%}")

            plot_risk_distributions(var_dist, cvar_dist)

        return

    # --- Step 3: AR(1) Model Execution ---
    if "m_mu" not in cached_results:
        # This block is for backwards compatibility with old cache files
        print("Running MCMC simulation for AR(1) model...")
        p_est, mu_est, sigma_est = 0, 0, 0
        trace, model, m_mu, m_sigma = estimate_ar1_params(
            np.log(prices), p_est, mu_est, sigma_est, int(1e4), False
        )
        cache_model_results(
            model_cache_path,
            {"trace": trace, "model": model, "m_mu": m_mu, "m_sigma": m_sigma},
        )
    else:
        m_mu = cached_results["m_mu"]
        m_sigma = cached_results["m_sigma"]

    # --- Step 4: Analysis and Plotting for AR(1) ---
    print("Visualizing AR(1) results...")
    p_est, mu_est, sigma_est = get_jump_prior_stats(prices)

    # Visualize the posteriors
    # plot_posterior(trace)
    # plot_priors_vs_posteriors(trace, m_mu, m_sigma)
    plot_posterior_predictive_check(model, trace)
    plot_residuals(trace, np.log(prices))

    # --- Step 5: Generate and Plot In-Sample Fit ---
    print("Generating AR(1) in-sample fit plot...")
    plot_ar1_in_sample_fit(
        trace,
        prices,
        p_jump=p_est,
        mu_jump=abs(mu_est),
        sigma_jump=sigma_est,
        n_paths=100,
        percentile=percentile,
        trim_percentile_lower=trim_lower,
        trim_percentile_upper=trim_upper,
    )

    # --- Step 6: Optional: Run Bayesian Risk Analysis ---
    if run_risk_analysis:
        print("\\n--- Running Bayesian Risk Analysis for AR(1) Model ---")
        var_dist, cvar_dist = get_ar1_risk_distribution(
            trace, s0=prices[-1], T=sim_days, N_per_sample=sim_paths, alpha=0.05
        )

        print(f"  - Mean VaR (95%): {np.mean(var_dist):.2%}")
        print(f"  - Std Dev of VaR: {np.std(var_dist):.2%}")
        print(f"  - Mean CVaR (95%): {np.mean(cvar_dist):.2%}")
        print(f"  - Std Dev of CVaR: {np.std(cvar_dist):.2%}")

        plot_risk_distributions(var_dist, cvar_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ornstein-Uhlenbeck process analysis."
    )
    parser.add_argument(
        "--asset_address", type=str, required=True, help="Asset address"
    )
    parser.add_argument("--chain_id", type=str, required=True, help="Chain ID")
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.5,
        help="Percentile for optimal parameters",
    )
    parser.add_argument(
        "--current_price", type=float, default=1.0, help="Current price for simulation"
    )
    parser.add_argument(
        "--n_paths", type=int, default=10, help="Number of simulated paths"
    )
    parser.add_argument(
        "--plot_jumps",
        action="store_true",
        help="Whether or not to plot path with jumps",
    )
    parser.add_argument(
        "--run-stationarity-tests",
        action="store_true",
        help="Run stationarity tests on the price series and exit.",
    )
    parser.add_argument(
        "--test-returns",
        action="store_true",
        help="If running stationarity tests, use log-returns instead of log-prices.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ar1",
        choices=["ar1", "garch", "ewma", "gbm"],
        help="Specify the model type to run ('ar1', 'garch', 'ewma', or 'gbm').",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=30,
        help="Number of days to forecast into the future for the GARCH model.",
    )
    parser.add_argument(
        "--trim-lower",
        type=float,
        default=0.1,
        help="Lower percentile for trimmed mean calculation in in-sample plot.",
    )
    parser.add_argument(
        "--trim-upper",
        type=float,
        default=0.9,
        help="Upper percentile for trimmed mean calculation in in-sample plot.",
    )
    parser.add_argument(
        "--run-risk-analysis",
        action="store_true",
        help="Run Bayesian VaR/CVaR analysis.",
    )
    parser.add_argument(
        "--sim-days",
        type=int,
        default=30,
        help="Simulation horizon (in days) for risk analysis.",
    )
    parser.add_argument(
        "--sim-paths",
        type=int,
        default=1000,
        help="Number of simulation paths per posterior sample for risk analysis.",
    )

    args = parser.parse_args()
    asyncio.run(
        run(
            asset_address=args.asset_address,
            chain_id=args.chain_id,
            percentile=args.percentile,
            current_price=args.current_price,
            n_paths=args.n_paths,
            plot_jumps=args.plot_jumps,
            run_stationarity_tests=args.run_stationarity_tests,
            test_returns=args.test_returns,
            model_type=args.model_type,
            forecast_days=args.forecast_days,
            trim_lower=args.trim_lower,
            trim_upper=args.trim_upper,
            run_risk_analysis=args.run_risk_analysis,
            sim_days=args.sim_days,
            sim_paths=args.sim_paths,
        )
    )
