import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable




def scatter_with_marginal_histograms(ax, x_values, y_values,
                                     x_bins=None, y_bins=None,
                                     x_label="", y_label="",
                                     title="",
                                     log_x=False):
    """
    Creates a scatter plot with marginal histograms in a specified axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object where the scatter plot will be drawn.
    x_values (array-like): Values for the x-axis.
    y_values (array-like): Values for the y-axis.
    x_bins (array-like or int, optional): Bin edges or number of bins for the x-axis histogram.
    y_bins (array-like or int, optional): Bin edges or number of bins for the y-axis histogram.
    """
    # Create a GridSpec layout within the provided axis
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # Main scatter plot
    ax.scatter(x_values, y_values)
    ax.set_yticks(y_bins)
    # ax.set_xlim(left=x_bins.min()-20*(1/sampling_rate), right=x_bins.max()+20*(1/sampling_rate))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Set x-axis to logarithmic scale if requested
    if log_x:
        ax.set_xscale('log')

    # Histogram for x values
    ax_histx.hist(x_values, bins=x_bins, edgecolor='black')
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['bottom'].set_visible(False)
    ax_histx.get_xaxis().set_visible(False)
    ax_histx.set_ylabel('Counts')
    ax_histx.set_title(title)

    # Histogram for y values

    hist_y_bins = np.sort(np.concatenate(
        ((y_bins - 0.5 * np.diff(y_bins).mean()), np.array([y_bins.max() + 0.5 * np.diff(y_bins).mean()]))))  # sorry
    ax.set_ylim(hist_y_bins.min() - 0.1, hist_y_bins.max() + 0.1)
    ax_histy.hist(y_values, bins=hist_y_bins, edgecolor='black', orientation='horizontal')
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['left'].set_visible(False)
    ax_histy.get_yaxis().set_visible(False)
    ax_histy.set_xlabel('Counts')

    # Adjust layout to prevent overlap
    plt.tight_layout()


def bar_scatter_plot_meanbars(df, title='Plot Title', ylabel='Y Axis', colorscheme='viridis'):
    """
    Function to plot a DataFrame's columns as scatter points, with the mean represented by a horizontal line.

    Parameters:
    df (pd.DataFrame): The input DataFrame. Each column is treated as a separate set of points.
    title (str): The title of the plot.
    ylabel (str): The label of the Y-axis.
    colorscheme (str): The matplotlib colorscheme to be used for the points.

    Returns:
    fig, ax: The matplotlib figure and axis objects (for use in subplots).
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the colormap based on the colorscheme provided
    cmap = plt.get_cmap(colorscheme)

    # Number of columns
    num_cols = len(df.columns)

    # Create a scatter plot for each column
    for i, col in enumerate(df.columns):
        # Drop NaN values from the column
        values = df[col].dropna()

        # Scatter plot for each column's values, centered on the column's iloc index
        scatter_x = np.full(len(values), i) + np.random.uniform(-0.05, 0.05, len(values))  # Small random jitter
        ax.scatter(scatter_x, values, color=cmap(i / num_cols), label=col, alpha=0.6)

        # Plot the mean as a horizontal line, centered on the column's iloc index
        mean_value = values.mean()
        ax.hlines(mean_value, i - 0.2, i + 0.2, colors='red', linewidth=2)

    # Set x-ticks to the column names
    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')

    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Return figure and axis for use in subplots
    return fig, ax



