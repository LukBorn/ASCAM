import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts import bar_scatter_plot_meanbars
from matplotlib.gridspec import GridSpec


filepaths = {'A2': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A2-0908.xlsx',
             'A4': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A4-0910.xlsx',
             'A4G2': '/Users/lukasborn/Desktop/analysis/ASCAM/macro analysis/A4G2-2224.xlsx'}

if True:
    # kinetics
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    # decay and desensitization fits
    fig1 = plt.figure(figsize=(15, 5))
    gs1 = GridSpec(1, 3, figure=fig1)

    subplots500 = [fig.add_subplot(gs[0, 0]),
                   fig.add_subplot(gs[0, 1]),
                   fig.add_subplot(gs[0, 2]),
                   fig1.add_subplot(gs[0, 0]),
                   fig1.add_subplot(gs[0, 1]),
                   ]

    subplots2 = [fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2]),
                fig1.add_subplot(gs[0, 2]),
                ]

    plot500 = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                              names=("condition","holding_potential")),
                           columns= ["Rise time (ms)", "Peak (pA)", "Steady state %","weighted tau (ms)","weighted tau (ms) decay"])


    for key in filepaths:
        ms500 = pd.read_excel(filepaths[key], sheet_name='500ms_ex')

        for variable in plot500.columns:
            plot500[variable][key, 50] = (np.array(ms500[ms500["Holding potential (mV)"] == 50][variable].dropna()))
            plot500[variable][key, -60] = (np.array(ms500[ms500["Holding potential (mV)"] == -60][variable].dropna()))

    for i, variable in enumerate(plot500.columns):
        bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot500[variable].to_list(),
                                                    index = [f"{i[0]}: {i[1]} mV" for i in plot500.index]).T,
                                  ax = subplots500[i],
                                  title = None,
                                  ylabel = variable
        )


    plot2 =  pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                              names=("condition","holding_potential")),
                           columns= ["Rise time (ms)", "Peak (pA)", "Steady state %","weighted tau (ms) decay"])

    for key in filepaths:
        ms2 = pd.read_excel(filepaths[key], sheet_name='2ms_ex')

        for variable in plot500.columns:
            plot2[variable][key, 50] = (np.array(ms2[ms2["Holding potential (mV)"] == 50][variable].dropna()))
            plot2[variable][key, -60] = (np.array(ms2[ms2["Holding potential (mV)"] == -60][variable].dropna()))

    for i, variable in enumerate(plot2.columns):
        bar_scatter_plot_meanbars(df = pd.DataFrame(data = plot2[variable].to_list(),
                                                    index = [f"{i[0]}: {i[1]} mV" for i in plot2.index]).T,
                                  ax = subplots2[i],
                                  title = None,
                                  ylabel = variable
        )

# ms2 = pd.read_excel(filepaths['A4'], sheet_name='2ms_ex')
#
# increment = pd.read_excel(filepaths['A4'], sheet_name='increment_ex')
#
# iv = pd.read_excel(filepaths['A4'], sheet_name='iv_ex')
#
# hz10 = pd.read_excel(filepaths['A4'], sheet_name='10hz_ex')
#
# hz20 = pd.read_excel(filepaths['A4'], sheet_name='20hz_ex')
#
# hz50 = pd.read_excel(filepaths['A4'], sheet_name='50hz_ex')