import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepaths = {'A2': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A2-0908.xlsx',
             'A4': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A4-0910.xlsx',
             'A4G2': 'C:\\Users\\sepps\\Desktop\\ascam\\macro analysis\\A4G2-2224.xlsx'}

if True:
    plot500 = pd.DataFrame(index = pd.MultiIndex.from_product((list(filepaths.keys()),(50,-60)),
                                                              names=("condition","holding_potential")),
                           columns= ["Rise time (ms)", "Peak (pA)", "Steady_state %","weighted tau (ms)","weighted tau (ms) desensetization"])


    for key in filepaths:
        ms500 = pd.read_excel(filepaths[key], sheet_name='500ms_ex')

        for variable in plot500.columns:
            plot500[variable][key, 50] = (np.array(ms500[ms500["Holding potential (mV)"] == 50][variable].mean()))
            plot500[variable][key, -60] = (np.array(ms500[ms500["Holding potential (mV)"] == -60][variable].mean()))

    for variable in plot500.columns:
        plt.plot

ms2 = pd.read_excel(filepaths['A4'], sheet_name='2ms_ex')

increment = pd.read_excel(filepaths['A4'], sheet_name='increment_ex')

iv = pd.read_excel(filepaths['A4'], sheet_name='iv_ex')

hz10 = pd.read_excel(filepaths['A4'], sheet_name='10hz_ex')

hz20 = pd.read_excel(filepaths['A4'], sheet_name='20hz_ex')

hz50 = pd.read_excel(filepaths['A4'], sheet_name='50hz_ex')