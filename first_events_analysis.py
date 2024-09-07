import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts import scatter_with_marginal_histograms
from matplotlib.gridspec import GridSpec

sampling_rate = 40000

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)


filepaths = {
    'EDTA1': ("/Users/lukasborn/Desktop/analysis/ASCAM/EDTA1.171025 019.apkl_first_events.csv",
              47,
              fig.add_subplot(gs[0, 0]),
              [0.  , -0.57, -1.18, -1.75, -2.33],
              'EDTA+CTZ'),
    'EDTA2': ("/Users/lukasborn/Desktop/analysis/ASCAM/EDTA2.180125 024._first_events.csv",
              None,
              fig.add_subplot(gs[0, 1]),
              [0.  , -0.5 , -1.04, -1.54, -2.05],
              'EDTA+CTZ'),
    'EDTA3': ("/Users/lukasborn/Desktop/analysis/ASCAM/EDTA3.180518 004.apkl_first_events.csv",
              list(range(344))+list(range(466,486)),
              fig.add_subplot(gs[0, 2]),
              [0.  , -0.53, -1.1 , -1.63, -2.17],
              'EDTA+CTZ'),
    'Zn1': ("/Users/lukasborn/Desktop/analysis/ASCAM/Zn1.180124 024._first_events.csv",
            None,
            fig.add_subplot(gs[1, 0]),
            [0.  , -0.57, -1.18, -1.75, -2.33],
            'Zn+CTZ'),
    "Zn2": ("/Users/lukasborn/Desktop/analysis/ASCAM/Zn2.180426 000._first_events.csv",
            None,
            fig.add_subplot(gs[1, 1 ]),
            [0.  , -0.57, -1.18, -1.75, -2.33],
            'Zn+CTZ'),
    'Zn3': ("/Users/lukasborn/Desktop/analysis/ASCAM/Zn3.180507 014._first_events.csv",
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


