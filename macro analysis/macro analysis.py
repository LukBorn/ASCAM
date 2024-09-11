import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scripts import bar_scatter_plot_meanbars

filepaths = {'A2': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A2-0908.xlsx',
             'A4': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A4-0910.xlsx',
             'A4G2': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A4G2-2224.xlsx'}

if True:
    plot500 = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                              names=("condition","holding_potential")),
                           columns= ["Rise time (ms)", "Peak (pA)", "Steady state %","weighted tau (ms)","weighted tau (ms) decay"])


    for key in filepaths:
        ms500 = pd.read_excel(filepaths[key], sheet_name='500ms_ex')

        for variable in plot500.columns:
            plot500[variable][key, 50] = (np.array(ms500[ms500["Holding potential (mV)"] == 50][variable].dropna()))
            plot500[variable][key, -60] = (np.array(ms500[ms500["Holding potential (mV)"] == -60][variable].dropna()))

    for variable in plot500.columns:
        x = np.concatenate(plot500[variable].tolist())
        ycounts = [i.shape[0] for i in plot500[variable].tolist()]
        y = []
        for i in range(len(ycounts)):
            y += [np.arange(100)[i]] * ycounts[i]
        y = np.array(y) + np.random.normal(scale=0.1, size=len(y)) # no overlap
        plt.scatter(x,y)
        plt.set_xtick_labels = [f'{i[0]}: {i[1]} mV'for i in plot500.index]

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