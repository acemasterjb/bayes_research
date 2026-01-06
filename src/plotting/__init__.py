import plotly.graph_objects as go
import numpy as np

def plot_grid_approx(p_grid: np.ndarray, posterior: np.ndarray, **kwargs) -> go.Figure:
    """
    Plots the grid approximation of a posterior distribution using plotly.

    Args:
        p_grid (np.ndarray): The grid of parameter values.
        posterior (np.ndarray): The posterior probabilities.
        **kwargs: Additional keyword arguments to be passed to go.Figure().

    Returns:
        go.Figure: The plotly figure object.
    """

    fig = go.Figure(**kwargs)
    fig.add_trace(go.Scatter(x=p_grid, y=posterior, mode='markers', name='Posterior Points'))
    fig.add_trace(go.Scatter(x=p_grid, y=posterior, mode='lines', name='Posterior Line'))
    fig.update_layout(
        title="Grid Approximation of Posterior",
        xaxis_title="Parameter Value (p)",
        yaxis_title="Posterior Probability"
    )

    return fig
