# src/examples/2_4.py

import os
import sys

# Add the root of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.motors import grid_approx
from src.plotting import plot_grid_approx

def main():
    """
    Runs the grid approximation for two different numbers of grid points
    and displays the resulting posterior distributions.
    """
    # Define parameters
    trials = 9
    num_successes = 6

    # --- Case 1: 5 grid points ---
    num_points_5 = 5
    p_grid_5, posterior_5 = grid_approx(
        trials=trials,
        num_successes=num_successes,
        num_points=num_points_5
    )
    fig_5 = plot_grid_approx(p_grid_5, posterior_5)
    fig_5.update_layout(title=f"Grid Approximation (Points = {num_points_5})")
    fig_5.show()

    # --- Case 2: 20 grid points ---
    num_points_20 = 20
    p_grid_20, posterior_20 = grid_approx(
        trials=trials,
        num_successes=num_successes,
        num_points=num_points_20
    )
    fig_20 = plot_grid_approx(p_grid_20, posterior_20)
    fig_20.update_layout(title=f"Grid Approximation (Points = {num_points_20})")
    fig_20.show()

if __name__ == "__main__":
    main()
