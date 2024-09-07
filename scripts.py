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