# src/exercises/2.py

import os
import sys

# Add the root of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import plotly.graph_objects as go
from src.motors import grid_approx

def main():
    """
    Runs grid approximation for three different scenarios and plots the
    posterior distributions on a single figure.
    """
    # Define the scenarios
    scenarios = [
        {"trials": 3, "num_successes": 3},
        {"trials": 4, "num_successes": 3},
        {"trials": 7, "num_successes": 5},
    ]

    num_points = 100

    # Create a single figure to hold all plots
    fig = go.Figure()

    # Iterate through scenarios and add traces to the figure
    for scenario in scenarios:
        trials = scenario["trials"]
        num_successes = scenario["num_successes"]

        p_grid, posterior = grid_approx(
            trials=trials,
            num_successes=num_successes,
            num_points=num_points
        )

        # Add a trace for the current scenario
        fig.add_trace(go.Scatter(
            x=p_grid,
            y=posterior,
            mode='lines',
            name=f"Trials={trials}, Successes={num_successes}"
        ))

    # Update the layout for the combined figure
    fig.update_layout(
        title="Grid Approximation for Multiple Scenarios",
        xaxis_title="Parameter Value (p)",
        yaxis_title="Posterior Probability",
        legend_title="Scenarios"
    )

    fig.show()

if __name__ == "__main__":
    main()
