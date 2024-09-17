import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts import scatter_with_marginal_histograms
from matplotlib.gridspec import GridSpec

sampling_rate = 40000

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)

all_EDTA = pd.DataFrame()
all_zinc = pd.DataFrame()

filepaths = {
    'EDTA1': ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/EDTA1.171025 019.apkl_first_events.csv",
              47,
              fig.add_subplot(gs[0, 0]),
              [0.  , -0.57, -1.18, -1.75, -2.33],
              'EDTA+CTZ'),
    'EDTA2': ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/EDTA2.180125 024._first_events.csv",
              None,
              fig.add_subplot(gs[0, 1]),
              [0.  , -0.5 , -1.04, -1.54, -2.05],
              'EDTA+CTZ'),
    'EDTA3': ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/EDTA3.180518 004.apkl_first_events.csv",
              list(range(344))+list(range(466,486)),
              fig.add_subplot(gs[0, 2]),
              [0.  , -0.53, -1.1 , -1.63, -2.17],
              'EDTA+CTZ'),
    'Zn1': ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/Zn1.180124 024._first_events.csv",
            None,
            fig.add_subplot(gs[1, 0]),
            [0.  , -0.57, -1.18, -1.75, -2.33],
            'Zn+CTZ'),
    "Zn2": ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/Zn2.180426 000._first_events.csv",
            None,
            fig.add_subplot(gs[1, 1 ]),
            [0.  , -0.57, -1.18, -1.75, -2.33],
            'Zn+CTZ'),
    'Zn3': ("/Users/lukasborn/Desktop/analysis/ASCAM/first event analysis/Zn3.180507 014._first_events.csv",
            None,
            fig.add_subplot(gs[1, 2]),
            [0. ,-0.59, -1.22, -1.82, -2.42],
            'Zn+CTZ'),
}


for key in filepaths.keys():
    df = pd.read_csv(filepaths[key][0]).set_index("Episode Number").drop("Unnamed: 0", axis =1)
    try: df.drop(filepaths[key][1])
    except ValueError:
        print('ehhhh')

    for col in df.columns:
            if '[s]' in col:
                # Multiply the column values by 1000
                df[col] = df[col] * 1000
                # Rename the column by replacing '[s]' with '[ms]'
                new_col_name = col.replace('[s]', '[ms]')
                df.rename(columns={col: new_col_name}, inplace=True)

    for col in df.columns:
            if '[A]' in col:
                # Multiply the column values by 1000
                df[col] = df[col] * 1000000000000
                # Rename the column by replacing '[s]' with '[ms]'
                new_col_name = col.replace('[A]', '[pA]')
                df.rename(columns={col: new_col_name}, inplace=True)

    df["Detection error"] = df['First Activation Time [ms]'] - df['First Event Time [ms]']
    substates = df["First Event Amplitude[pA]"].unique()
    substates.sort()
    substates = substates[::-1]
    df["Subconductance state"] = df["First Event Amplitude[pA]"].apply(lambda x: np.where(substates == x)[0][0])

    if "EDTA" in key:
        all_EDTA = pd.concat((all_EDTA, df))
    elif 'Zn' in key:
        all_zinc = pd.concat((all_zinc, df))


    scatter_with_marginal_histograms(ax=filepaths[key][2],
                                     x_values=df['First Event Time [ms]'],
                                     y_values=df['First Event Amplitude[pA]'],
                                     x_bins=np.linspace(start=df['First Activation Time [ms]'].min() - 1 / sampling_rate,
                                                        stop=df['First Activation Time [ms]'].max() + 1 / sampling_rate,
                                                        num=int(df.shape[0] / 5)),
                                     y_bins=np.array(filepaths[key][3]),
                                     x_label='Time [ms]',
                                     y_label='Amplitude [pA]',
                                     title=filepaths[key][4],
                                     log_x=False)


df_list = (all_EDTA, all_zinc)


def log_sqrt_hist(ax, data, bins):
    # Ensure no negative values (subtract min to shift data to positive range)
    data = np.array(data - data.min())
    data.sort()


    # Calculate logarithmic bins based on the data range
    log_min = np.log10(data[data>0][0])  # Add a small constant to avoid log(0)
    log_max = np.log10(data[-1])
    bin_edges = np.logspace(log_min, log_max, bins)

    # Plot the histogram with logarithmic bins
    ax.hist(data, bins=bin_edges, edgecolor='black')

    # Set the x-axis to log scale
    ax.set_xscale('log')
    ax.set_xlabel("log(first latency)")

    # Set the y-axis to square root scale
    ax.set_yscale('function', functions=(np.sqrt, lambda y: y ** 2))  # sqrt scale
    ax.set_ylabel('sqrt(counts)')

names = [('EDTA+CTZ',125), ('Zn+CTZ',305)]
for j, df1 in enumerate(df_list):
    unique_levels = df1["Subconductance state"].unique()
    unique_levels.sort()

    # Create a figure with as many columns as there are unique levels
    fig, ax = plt.subplots(1, ncols=len(unique_levels), figsize=(5 * len(unique_levels), 4))

    for i, level in enumerate(unique_levels):
        subset = df1[df1["Subconductance state"] == level]
        print(subset.shape)
        # Plot the histogram for each sub-level
        log_sqrt_hist(ax=ax[i], data=subset["First Event Time [ms]"], bins=50)
        ax[i].set_title(f"Subconductance state {level+1}")  # Add title for each subplot
    fig.suptitle(names[j][0])
    plt.tight_layout()  # Avoid overlap between subplots
    plt.setp(ax, xlim = (0.01,100), ylim = (0,names[j][1]))
    plt.setp(ax[1], yticklabels=[], ylabel='')
    plt.setp(ax[2], yticklabels=[], ylabel='')
    plt.setp(ax[3], yticklabels=[], ylabel='')

    plt.show()  # Show the figure