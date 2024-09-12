import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scripts import bar_scatter_plot_meanbars
from matplotlib.gridspec import GridSpec

def bar_scatter_plot_meanbars(df, ax, title='Plot Title', ylabel='Y Axis', colorscheme='viridis',
                              plot_error_bars = True, plot_mean_bars = True, error_type = "bar", plot_line = False):
    """
    Function to plot a DataFrame's columns as scatter points, with the mean represented by a black horizontal line,
    and error bars (quartiles) as grey horizontal lines.

    Parameters:
    df (pd.DataFrame): The input DataFrame. Each column is treated as a separate set of points.
    ax (matplotlib axis): The axis object to plot on (for use in subplots).
    title (str): The title of the plot.
    ylabel (str): The label of the Y-axis.
    colorscheme (str): The matplotlib colorscheme to be used for the points.
    """

    # Get the colormap based on the colorscheme provided
    cmap = plt.get_cmap(colorscheme)

    df = df.dropna(axis=1, how="all")

    # Number of columns
    num_cols = len(df.columns)

    # Array to store the mean values for the line plot
    mean_values = []

    # Create a scatter plot for each column
    for i, col in enumerate(df.columns):
        # Drop NaN values from the column
        values = df[col].dropna()

        # Scatter plot for each column's values, centered on the column's iloc index
        scatter_x = np.full(len(values), i) + np.random.uniform(-0.05, 0.05, len(values))  # Small random jitter
        ax.scatter(scatter_x, values, color=cmap(i / num_cols), label=col, alpha=0.6)

        # Calculate mean and quartiles (for error bars)
        mean_value = values.mean()
        if error_type == 'bar':
            q1 = np.percentile(values, 25)  # First quartile (25th percentile)
            q3 = np.percentile(values, 75)  # Third quartile (75th percentile)
        elif error_type == 'std':
            q1 = mean_value + values.std()
            q3 = mean_value - values.std()
        if plot_error_bars:
            # Plot the quartiles (error bars) as grey horizontal lines
            ax.hlines([q1, q3], i - 0.1, i + 0.1, colors='grey', linewidth=1, linestyle='--')
            ax.vlines(i, q1, q3, colors='grey', linewidth=1, linestyle= "--")
        if plot_mean_bars:
            # Plot the mean as a black horizontal line, centered on the column's iloc index
            ax.hlines(mean_value, i - 0.2, i + 0.2, colors='black', linewidth=2)

    if plot_line:
        ax.plot(np.arange(num_cols), mean_values, color='blue', linestyle='-', linewidth=2, marker='o')

    # Set x-ticks to the column names
    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')

    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_title(title)


filepaths = {'A2': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A2-0908.xlsx',
             'A4': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A4-0910.xlsx',
             'A4G2': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A4G2-2224.xlsx'}

"""
500ms
"""

plot500 = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                          names=("condition","holding_potential")),
                       columns= ["Rise time (ms)", "Peak (pA)", "Steady state %","weighted tau (ms)","weighted tau (ms) decay"])


for key in filepaths:
    ms500 = pd.read_excel(filepaths[key], sheet_name='500ms_ex')

    for variable in plot500.columns:
        plot500[variable][key, 50] = (np.array(ms500[ms500["Holding potential (mV)"] == 50][variable].dropna()))
        plot500[variable][key, -60] = (np.array(ms500[ms500["Holding potential (mV)"] == -60][variable].dropna()))

plot500["Peak (pA)"] = plot500["Peak (pA)"].abs()

"""
2ms
"""

plot2 =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                          names=("condition","holding_potential")),
                       columns= ["Rise time (ms)", "Peak (pA)","weighted tau (ms)"])

for key in filepaths:
    ms2 = pd.read_excel(filepaths[key], sheet_name='2ms_ex')

    for variable in plot2.columns:
        plot2[variable][key, 50] = (np.array(ms2[ms2["Holding potential (mV)"] == 50][variable].dropna()))
        plot2[variable][key, -60] = (np.array(ms2[ms2["Holding potential (mV)"] == -60][variable].dropna()))

plot2["Peak (pA)"] = plot2["Peak (pA)"].abs()

"""
increment
"""
plot_increment = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                  names=("condition","holding_potential")),
                              columns = [0,25,100,225,400,625,900,1225,1600,2025,2500,3025])
for key in filepaths:
    increment = pd.read_excel(filepaths[key], sheet_name='increment_ex').set_index('holding v').T
    increment.index = increment.index.astype(float).astype(int)

    for variable in plot_increment.columns:
        plot_increment[variable][key, 50] = (np.array(increment[increment.index == 50][variable].dropna()))
        plot_increment[variable][key, -60] = (np.array(increment[increment.index == -60][variable].dropna()))

"""
trains
"""

plot_10hz =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                  names=("condition","holding_potential")),
                              columns = np.arange(0,5000,100))
for key in filepaths:
    hz10 = pd.read_excel(filepaths[key], sheet_name='10Hz_ex').set_index('holding v').T
    hz10.index = hz10.index.astype(float).astype(int)

    for variable in plot_10hz.columns:
        plot_10hz[variable][key, 50] = (np.array(hz10[hz10.index == 50][variable].dropna()))
        plot_10hz[variable][key, -60] = (np.array(hz10[hz10.index == -60][variable].dropna()))

plot_20hz =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                  names=("condition","holding_potential")),
                              columns = np.arange(0,5000,50))
for key in ['A4','A4G2']:
    hz20 = pd.read_excel(filepaths[key], sheet_name='20Hz_ex').set_index('holding v').T
    hz20.index = hz20.index.astype(float).astype(int)

    for variable in plot_20hz.columns:
        plot_20hz[variable][key, 50] = (np.array(hz20[hz20.index == 50][variable].dropna()))
        plot_20hz[variable][key, -60] = (np.array(hz20[hz20.index == -60][variable].dropna()))

plot_50hz =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                  names=("condition","holding_potential")),
                              columns = np.arange(0,5000,20))

for key in ['A4', 'A4G2']:
    hz50 = pd.read_excel(filepaths[key], sheet_name='50Hz_ex').set_index('holding v').T
    hz50.index = hz50.index.astype(float).astype(int)

    for variable in plot_10hz.columns:
        plot_50hz[variable][key, 50] = (np.array(hz50[hz50.index == 50][variable].dropna()))
        plot_50hz[variable][key, -60] = (np.array(hz50[hz50.index == -60][variable].dropna()))

"""
iv
"""
plot_iv = plot_increment = pd.DataFrame(index = filepaths.keys(),
                                        columns = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100])
for key in filepaths:
    iv = pd.read_excel(filepaths[key], sheet_name='IV_ex').set_index("Recording number").T

    for variable in plot_iv.columns:
        plot_iv[variable][key] = (np.array(iv[variable].dropna()))



for i, condition in enumerate(plot_iv.index):
    x = np.array(len(plot_iv.loc[condition].to_list()[1]) * plot_iv.columns.to_list()).astype(float)
    x.sort()
    x += np.random.normal(loc = 10, scale = 1, size = x.shape[0])
    plt.scatter(x = x, y = plot_iv.loc[condition].to_list())


"""

    # kinetics
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(1, 3, figure=fig)

# decay and desensitization fits
fig1 = plt.figure(figsize=(15, 5))
gs1 = GridSpec(1, 3, figure=fig1)

subplots500 = [fig.add_subplot(gs[0, 0]),
               fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[0, 2]),
               fig1.add_subplot(gs[0]),
               fig1.add_subplot(gs[1]),
               ]

subplots2 = [fig.add_subplot(gs[1, 0]),
             fig.add_subplot(gs[1, 1]),
             fig.add_subplot(gs[1, 2]),
             fig1.add_subplot(gs[2]),
             ]

for i, variable in enumerate(plot500.columns):
    bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot500[variable].to_list(),
                                                index = [f"{i[0]}: {i[1]} mV" for i in plot500.index]).T,
                              ax = subplots500[i],
                              title = "",
                              ylabel = variable
    )



for i, variable in enumerate(plot2.columns):
    bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot2[variable].to_list(),
                                                index = [f"{i[0]}: {i[1]} mV" for i in plot2.index]).T,
                              ax = subplots2[i],
                              title = None,
                              ylabel = variable
    )
"""
