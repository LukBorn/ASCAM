import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scripts import bar_scatter_plot_meanbars
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


# Get the viridis colormap
cmap = cm.get_cmap('viridis')



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

plot_500 = False
if plot_500:
    fig, ax = plt.subplots(1,3)
    for i, variable in enumerate(plot500.columns[:3]):
        bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot500[variable].to_list(),
                                                    index = [f"{i[0]}: {i[1]} mV" for i in plot500.index]).T,
                                  ax = ax[i],
                                  title = "",
                                  ylabel = variable
        )
    plt.tight_layout()

plot_tau = True
if plot_tau:
    fig, ax = plt.subplots(1,3)
    for i, variable in enumerate(plot500.columns[3:5]):
        bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot500[variable].to_list(),
                                                    index = [f"{i[0]}: {i[1]} mV" for i in plot500.index]).T,
                                  ax = ax[i],
                                  title = "",
                                  ylabel = variable + " 500ms"
        )
    bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot2['weighted tau (ms)'].to_list(),
                                                    index = [f"{i[0]}: {i[1]} mV" for i in plot500.index]).T,
                                  ax = ax[2],
                                  title = "",
                                  ylabel = variable + " 2ms")
    plt.tight_layout()



"""
increment
"""
plot_increment = False
if plot_increment:
    plot_increment = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                      names=("condition","holding_potential")),
                                  columns = [0,25,100,225,400,625,900,1225,1600,2025,2500,3025])
    for key in filepaths:
        increment = pd.read_excel(filepaths[key], sheet_name='increment_ex').set_index('holding v').T
        increment.index = increment.index.astype(float).astype(int)

        for variable in plot_increment.columns:
            plot_increment[variable][key, 50] = (np.array(increment[increment.index == 50][variable].dropna()))
            plot_increment[variable][key, -60] = (np.array(increment[increment.index == -60][variable].dropna()))

    plot_increment.index = [f"{i[0]}: {i[1]} mV" for i in plot_increment.index]
    plot_increment.drop(0, axis = 1, inplace = True)
    #plot_increment.columns = [0]+(plot_increment.columns[1:]-plot_increment.columns[:-1] -25).to_list()

    # Filter data for holding potential 50 mV and -60 mV
    plot_increment_50 = plot_increment.loc[plot_increment.index.str.contains("50 mV")]
    plot_increment_60 = plot_increment.loc[plot_increment.index.str.contains("-60 mV")]

    # Create two vertically stacked subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    # Normalize the number of conditions to match the colormap range
    norm = mcolors.Normalize(vmin=0, vmax=len(plot_increment_50.index) - 1)
    # Plot for holding potential = 50 mV
    for i, condition in enumerate(plot_increment_50.index):
        means = plot_increment_50.loc[condition].apply(np.mean)
        error = plot_increment_50.loc[condition].apply(np.std)

        ax[0].plot(plot_increment.columns, means, label=f'{condition}', marker='o',)
        ax[0].fill_between(plot_increment.columns, means - error, means + error, alpha=0.3)
    ax[0].set_xlabel("Time since last pulse (ms)", fontsize=12)
    ax[0].set_ylabel('Current (normalized to first activation)',fontsize=12)
    ax[0].set_title('Holding Potential = 50 mV',fontsize=14)
    ax[0].legend()

    # Plot for holding potential = -60 mV
    for condition in plot_increment_60.index:
        means = plot_increment_60.loc[condition].apply(np.mean)
        error = plot_increment_60.loc[condition].apply(np.std)
        ax[1].plot(plot_increment.columns, means, label=f'{condition}', marker='o')
        ax[1].fill_between(plot_increment.columns, means - error, means + error, alpha=0.3)


    ax[1].set_xlabel("Time since last pulse (ms)",fontsize=12)
    ax[1].set_title('Holding Potential = -60 mV',fontsize=14)
    ax[1].legend()

    # Set x-axis ticks to be the same for both subplots
    ax[1].set_xscale('log')
    #ax[1].set_xticks(plot_increment.columns)
    #ax[1].set_xticklabels([0]+(plot_increment.columns[1:]-plot_increment.columns[:-1] -25).to_list())

    # Adjust layout
    plt.tight_layout()
    plt.show()

"""
trains
"""
plot_10hz=False
if plot_10hz:
    plot_10hz =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                                      names=("condition","holding_potential")),
                                  columns = np.arange(0,5000,100))
    for key in filepaths:
        hz10 = pd.read_excel(filepaths[key], sheet_name='10Hz_ex').set_index('holding v').T
        hz10.index = hz10.index.astype(float).astype(int)

        for variable in plot_10hz.columns:
            plot_10hz[variable][key, 50] = (np.array(hz10[hz10.index == 50][variable].dropna()))
            plot_10hz[variable][key, -60] = (np.array(hz10[hz10.index == -60][variable].dropna()))

    plot_10hz.index = [f"{i[0]}: {i[1]} mV" for i in plot_10hz.index]

    # Filter data for holding potential 50 mV and -60 mV
    plot_10hz_50 = plot_10hz.loc[plot_10hz.index.str.contains("50 mV")]
    plot_10hz_60 = plot_10hz.loc[plot_10hz.index.str.contains("-60 mV")]
    fig, ax = plt.subplots(2, 1, figsize=(12,9), sharex=True)
    # Normalize the number of conditions to match the colormap range
    norm = mcolors.Normalize(vmin=0, vmax=len(plot_10hz_50.index) - 1)

    # Plot for holding potential = 50 mV
    for condition in plot_10hz_50.index:
        means = plot_10hz_50.loc[condition].apply(np.mean)
        error = plot_10hz_50.loc[condition].apply(np.std)
        ax[0].plot(plot_10hz.columns, means, label=f'{condition}', marker='o',color = cmap(norm(i)))
        ax[0].fill_between(plot_10hz.columns, means - error, means + error, alpha=0.3,color = cmap(norm(i)))

    ax[0].set_ylabel('Current (normalized to first activation)',fontsize=12)
    ax[0].set_title('Holding Potential = 50 mV',fontsize=14)
    ax[0].legend()

    # Plot for holding potential = -60 mV
    for condition in plot_10hz_60.index:
        means = plot_10hz_60.loc[condition].apply(np.mean)
        error = plot_10hz_60.loc[condition].apply(np.std)
        ax[1].plot(plot_10hz.columns, means, label=f'{condition}', marker='o',color = cmap(norm(i)))
        ax[1].fill_between(plot_10hz.columns, means - error, means + error, alpha=0.3,color = cmap(norm(i)))

    ax[1].set_ylabel('Current (normalized to first activation)',fontsize=12)
    ax[1].set_xlabel("Time (ms)", fontsize=12)
    ax[1].set_title('Holding Potential = -60 mV', fontsize=14)
    ax[1].legend()

    # Set x-axis ticks to be the same for both subplots
    ax[1].set_xticks(plot_10hz.columns[::2])

    # Adjust layout
    plt.tight_layout()
    plt.show()

plot_20hz=False
if plot_20hz:
    plot_20hz =  pd.DataFrame(index = pd.MultiIndex.from_product((('A4','A4G2'),(50,-60)),
                                                                      names=("condition","holding_potential")),
                                  columns = np.arange(0,5000,50))
    for key in ['A4','A4G2']:
        hz20 = pd.read_excel(filepaths[key], sheet_name='20Hz_ex').set_index('holding v').T
        hz20.index = hz20.index.astype(float).astype(int)

        for variable in plot_20hz.columns:
            plot_20hz[variable].loc[key, 50] = (np.array(hz20[hz20.index == 50][variable].dropna()))
            plot_20hz[variable].loc[key, -60] = (np.array(hz20[hz20.index == -60][variable].dropna()))

    plot_20hz.index = [f"{i[0]}: {i[1]} mV" for i in plot_20hz.index]

    plot_20hz.dropna(axis =0, how="all", inplace = True)

    # Filter data for holding potential 50 mV and -60 mV
    plot_20hz_50 = plot_20hz.loc[plot_20hz.index.str.contains("50 mV")]
    plot_20hz_60 = plot_20hz.loc[plot_20hz.index.str.contains("-60 mV")]
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    # Normalize the number of conditions to match the colormap range
    norm = mcolors.Normalize(vmin=0, vmax=len(plot_20hz_50.index) - 1)

    # Plot for holding potential = 50 mV
    for condition in plot_20hz_50.index:
        means = plot_20hz_50.loc[condition].apply(np.mean)
        error = plot_20hz_50.loc[condition].apply(np.std)
        ax[0].plot(plot_20hz.columns, means, label=f'{condition}', marker='o',color = cmap(norm(i)))
        ax[0].fill_between(plot_20hz.columns, means - error, means + error, alpha=0.3,color = cmap(norm(i)))

    ax[0].set_ylabel('Current (normalized to first activation)', fontsize=12)
    ax[0].set_title('Holding Potential = 50 mV', fontsize=14)
    ax[0].legend()

    # Plot for holding potential = -60 mV
    for condition in plot_20hz_60.index:
        means = plot_20hz_60.loc[condition].apply(np.mean)
        error = plot_20hz_60.loc[condition].apply(np.std)
        ax[1].plot(plot_20hz.columns, means, label=f'{condition}', marker='o')
        ax[1].fill_between(plot_20hz.columns, means - error, means + error, alpha=0.3)

    ax[1].set_ylabel('Current (normalized to first activation)', fontsize=12)
    ax[1].set_xlabel("Time (ms)", fontsize=12)
    ax[1].set_title('Holding Potential = -60 mV', fontsize=14)
    ax[1].legend()

    # Set x-axis ticks to be the same for both subplots
    ax[1].set_xticks(plot_20hz.columns[::4])

    # Adjust layout
    plt.tight_layout()
    plt.show()


plot_50hz = False #"""Doesnt work"""
if plot_50hz:
    plot_50hz =  pd.DataFrame(index = pd.MultiIndex.from_product((['A4','A4G2'],(50,-60)),
                                                                      names=("condition","holding_potential")),
                                  columns = np.arange(0,5000,20))

    for key in ['A4', 'A4G2']:
        hz50 = pd.read_excel(filepaths[key], sheet_name='50Hz_ex').set_index('holding v').T
        hz50.index = hz50.index.astype(float).astype(int)

        for variable in plot_50hz.columns:
            plot_50hz[variable][key, 50] = (np.array(hz50[hz50.index == 50][variable].dropna()))
            plot_50hz[variable][key, -60] = (np.array(hz50[hz50.index == -60][variable].dropna()))
    plot_50hz = plot_50hz[~plot_50hz.applymap(lambda x: x == []).all(axis=1)]
    plot_50hz.index = [f"{i[0]}: {i[1]} mV" for i in plot_50hz.index]
    plot_50hz.drop(0, axis=1, inplace=True)

    fig, ax = plt.subplots()

    for condition in plot_50hz.index:
        means = plot_50hz.loc[condition].apply(np.mean)
        error = plot_50hz.loc[condition].apply(np.std)
        ax.plot(plot_50hz.columns, means, label=f'{condition}', marker='o')
        ax.fill_between(plot_50hz.columns, means - error, means + error, alpha=0.3)

    ax.set_label('Current (normalized to first activation)')
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(plot_50hz.columns)
    ax.legend()
    plt.show()

"""
iv
"""
plot_iv = False
if plot_iv:
    plot_iv = pd.DataFrame(index = filepaths.keys(),
                                            columns = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100])
    for key in filepaths:
        iv = pd.read_excel(filepaths[key], sheet_name='IV_ex').set_index("Recording number").T

        for variable in plot_iv.columns:
            plot_iv[variable][key] = (np.array(iv[variable].dropna()))

    fig, ax = plt.subplots(1,3)

    ax[0].set_ylabel('Current (normalized to -80mv)')

    for i, condition in enumerate(plot_iv.index):
        x = np.array(len(plot_iv.loc[condition].to_list()[1]) * plot_iv.columns.to_list()).astype(float)
        x.sort()
        x += np.random.normal(loc = 0, scale = 1, size = x.shape[0])
        means = plot_iv.loc[condition].apply(np.mean)
        error = plot_iv.loc[condition].apply(np.std)

        ax[i].scatter(x = x, y = plot_iv.loc[condition].to_list(), marker='o',facecolors='none', edgecolors='grey' )
        ax[i].plot(plot_iv.columns, means, color = 'black')
        ax[i].errorbar(x = plot_iv.columns, y = means, yerr= error,color = 'darkslategrey')
        ax[i].set_xticks(plot_iv.columns)
        ax[i].set_xlabel("Voltage (mv)")
        ax[i].set_title(condition)
        ax[i].grid()
        ax[i].set_ylim(ymax=2.5, ymin=-2.5)


        # get the axis
        #ax[i] = plt.gca()
        #ax[i].spines['bottom'].set_position('zero')
        #ax[i].spines['left'].set_position('zero')
    fig.tight_layout()

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
