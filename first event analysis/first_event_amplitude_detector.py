import numpy as np

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scripts import scatter_with_marginal_histograms
from sklearn.decomposition import PCA

root = tk.Tk()
root.withdraw()

sampling_rate = 40000

filepaths = pd.DataFrame(index=['EDTA1', 'EDTA2', 'EDTA3', 'Zn1', 'Zn2', 'Zn3'],
                         columns=["t_act", "events", "time_correction_factor"],
                         data=[['~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/171025 019/171025019_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/171025 019/171025019_1-46_2kHz_events_17-61ms.csv',
                                1000],
                               ['~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/180125 024/180125024_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/180125 024/events1_khz.csv',
                                1],
                               ['~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/180518 004/180518004_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 EDTA CTZ/180518 004/180518004_344-465_2kHz_events.csv',
                                0.001],
                               ['~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180124 024/180124024_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180124 024/180124024_62-92_events.csv',
                                1],
                               ['~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180426 000/180426000_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180426 000/180426000_diff_2-101_2kHz.csv',
                                1],
                               ['~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180507 014/180507014_tact.csv',
                                '~/Desktop/analysis/first event amplitude/T1 zinc CTZ/180507 014/180507014_100-494_skHz_events.csv',
                                1]])

experiment = 'Zn1'
verbose = True
drop_nonzero_t0 = False
drop_early_max_t_act = False
dead_time = 0


t_act = pd.read_csv(filepaths.loc[experiment]["t_act"], names=("episode", "t_act"), index_col="episode")
#convert to ms
t_act *= filepaths.loc[experiment]["time_correction_factor"]
#exclude outliers -> mostly the ones that falsely assume tact is right at the begining of the episode
t_act = t_act[t_act > t_act.mean() - 3 * t_act.std()].dropna()
#round
t_act = t_act.round(7)

events = pd.read_csv(filepaths.loc[experiment]["events"]).round(7)

# use onlz episodes in both t_act and events
t_act = t_act[t_act.index.isin(events["episode number"])]
events = events[events["episode number"].isin(t_act.index)]

t0 = t_act.min()[0]
sub_levels = events['amplitude [pA]'].unique()


"""
Results Dataframe
index: episodes
columns: time_e1: time point of first event according to t_act
         amplitude_e1: idealized amplitude at time_e1
         time_max_e1: beginning time point of the first event, after whiich an event with lower subconductance starts
         amplitude_max_e1 : idealized amplitude at time_max_e1
         trace: 1d array starting at t0 and ending at the last t_stop of the truncated episode. 
                with the idealized amplitude at each time step
         dwell_times: the dwell times of the idealized trace during the truncated episode
         
"""
results = pd.DataFrame(index=t_act.index,
                       columns=['amplitude_e1', 'time_e1',
                                'amplitude_max_e1', 'time_max_e1',
                                'trace', 'dwell_times'])

for episode_n in t_act.index:
    # get t_act for this episode
    episode_t_act = t_act.loc[episode_n][0]

    # get all events of episode
    episode = events[events['episode number'] == episode_n]

    full_trace = pd.DataFrame(index=np.arange(start=0,
                                              stop=100,
                                              step=1000 / sampling_rate).round(5),
                              columns=['trace'],
                              data=0)

    for step in episode.index:
        full_trace.loc[episode.loc[step]['t_start']:episode.loc[step]['t_stop']] = episode.loc[step]['amplitude [pA]']

    # time and amplitude of first event
    results.loc[episode_n]['time_e1'] = episode_t_act

    results.loc[episode_n]['amplitude_e1'] = episode.loc[episode[
        (episode_t_act > episode['t_start']) &
        (episode_t_act <= episode['t_stop']) |
        (episode_t_act == episode['t_start'])
        ].index[0]]['amplitude [pA]']


    greater_t0_idx = episode.index[episode['t_start'] >= t0]
    t0_amplitude = episode.loc[greater_t0_idx.union(greater_t0_idx - 1)].iloc[0]["amplitude [pA]"]



    if t0_amplitude == 0.0:
    # delete all events before t_0 but include the last one before
        episode = episode.loc[greater_t0_idx.union(greater_t0_idx - 1)]
    else:
        if not drop_nonzero_t0:
            # delete all events before t_0 but include the last two before
            episode = episode.loc[greater_t0_idx.union(greater_t0_idx - 2)]

            # make sure you didnt get some random noise that happened to be at t0 by checking if the next event after t0 was to 0
            if episode.iloc[2]['amplitude [pA]'] == 0.0:
                episode = episode.iloc[2:]
                if verbose:
                    print(f'Episode {episode_n} idealized amplitude at t0 was {t0_amplitude}, determined to be noise and DROPPED L+ RATIO')
            else:
                if verbose:
                    print(f'Episode {episode_n} idealized amplitude at t0 was {t0_amplitude}, included 2 before.')
        else:
            # exclude episode from analysis if the idealized amplitude at t0 is not 0
            t_act.drop(index=episode_n, inplace=True)
            results.drop(index=episode_n, inplace=True)
            if verbose:
                print(f'Skipped episode {episode_n} because idealized amplitude at t0 was {t0_amplitude}')
            continue


    # include only events up until the first drop in subconductance level
    episode = episode.iloc[:np.where(np.diff(np.abs(episode["amplitude [pA]"])) < 0)[0][0] + 2]

#TODO figure out what to do with these
    if drop_early_max_t_act:
        #exclude episode from analysis if first event is before episode_t_act
        if episode['t_stop'].iloc[-1] < episode_t_act:
            if verbose:
                print(f'Skipped episode {episode_n} the detected first event was before t_act')
            t_act.drop(index=episode_n, inplace=True)
            results.drop(index=episode_n, inplace=True)
            continue


    #time and amplitude of maximal first event
    results.loc[episode_n]['amplitude_max_e1'] = episode.iloc[-2]['amplitude [pA]']
    results.loc[episode_n]['time_max_e1'] = episode.iloc[-2]['t_start']

    # define trace array and set the trace
    trace = pd.DataFrame(index=np.arange(start=t0,
                                         stop=episode.iloc[-2]['t_stop'],
                                         step=1000 / sampling_rate).round(5),
                         columns=['trace'],
                         data=0)


    for step in episode.index[1:-1]:
        trace.loc[episode.loc[step]['t_start']:episode.loc[step]['t_stop']] = episode.loc[step]['amplitude [pA]']

    results.loc[episode_n]['trace'] = np.array(trace['trace'])




    # define dwell times
    dwell_times = np.array(episode['duration [ms]'])[:-1]
    dwell_times[0] = episode.iloc[1]["t_start"] - t0
    if dwell_times.shape[0] != 5:
        dwell_times = np.pad(dwell_times, (0, 5 - dwell_times.shape[0]), mode='constant', constant_values=0.0)
    results.loc[episode_n]['dwell_times'] = dwell_times.round(7)

"""
Plot the results
"""

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig)

# todo possibly - make a new function that actuallz puts a histogram on each line so zou see the distribution and not just dots
# actuallz just make a boxplot
A = fig.add_subplot(gs[0, 0])
scatter_with_marginal_histograms(ax=A,
                                 x_values=results["time_e1"],
                                 y_values=np.abs(results['amplitude_e1']),
                                 x_bins=np.linspace(start=results['time_max_e1'].min() - 1 / sampling_rate,
                                                    stop=results["time_max_e1"].max() + 1 / sampling_rate,
                                                    num=int(results.shape[0] / 5)),
                                 y_bins=np.abs(sub_levels),
                                 x_label='Time [ms]',
                                 y_label='Amplitude [pA]',
                                 title='First event amplitude',
                                 log_x=False)

B = fig.add_subplot(gs[0, 1])
scatter_with_marginal_histograms(ax=B,
                                 x_values=results["time_max_e1"],
                                 y_values=np.abs(results['amplitude_max_e1']),
                                 x_bins=np.linspace(start=results['time_max_e1'].min() - 1 / sampling_rate,
                                                    stop=results["time_max_e1"].max() + 1 / sampling_rate,
                                                    num=int(results.shape[0] / 5)),
                                 y_bins=np.abs(sub_levels),
                                 x_label='Time [ms]',
                                 y_label='Amplitude [pA]',
                                 title='Maximal first event amplitude',
                                 log_x=False)
"""
C = fig.add_subplot(gs[1, 0])
scatter_with_marginal_histograms(ax=C,
                                 x_values=results["time_e1"],
                                 y_values=np.abs(results['amplitude_e1']),
                                 x_bins=np.linspace(start=results['time_max_e1'].min() - 1 / sampling_rate,
                                                    stop=results["time_max_e1"].max() + 1 / sampling_rate,
                                                    num=int(results.shape[0] / 5)),
                                 y_bins=np.abs(sub_levels),
                                 x_label='Time [ms]',
                                 y_label='Amplitude [pA]',
                                 title='First event amplitude log x',
                                 log_x=True)

D = fig.add_subplot(gs[1, 1])
scatter_with_marginal_histograms(ax=D,
                                 x_values=results["time_max_e1"],
                                 y_values=np.abs(results['amplitude_max_e1']),
                                 x_bins=np.linspace(start=results['time_max_e1'].min() - 1 / sampling_rate,
                                                    stop=results["time_max_e1"].max() + 1 / sampling_rate,
                                                    num=int(results.shape[0] / 5)),
                                 y_bins=np.abs(sub_levels),
                                 x_label='Time [ms]',
                                 y_label='Amplitude [pA]',
                                 title='Maximal first event amplitude log x',
                                 log_x=True)
"""


"""
Finding patterns in the dwell times

Attempt 1
PCA of dwell times
yea really this should only be done to a subset of the episodes, for example those where first event amplitude is x
otherwise the maximal number of components extracted is 5, and as there are 4 distinct first events 
(corresponding to each subconductance state) you cant really find patterns that well...

Attempt 2
histogram of the dwell times of the first event
doesnt really work because there is not enough episodes 
"""
dwell_time_analysis = False
if dwell_time_analysis:
    dwell_times = pd.DataFrame(results["dwell_times"].tolist(), columns=sub_levels)
    pca = PCA(whiten = True)
    pca.fit(dwell_times)
    og_components = pd.DataFrame(columns = dwell_times.columns,data =pca.components_.T)
    og_components *= np.sqrt(pca.explained_variance_)
    og_components += np.mean(dwell_times, axis = 0).to_list()

    #reconstruct trace



    # todo plot the principal components and put the explained variance in there somehow
    plt.plot(og_components)






"""
#plot histogram of max first event amplitude
C = fig.add_subplot(gs[1,0])
C.hist(np.abs(results['amplitude_max_e1']), bins = 5)
C.set_title('Maximal first event amplitude')
C.set_xlabel('Amplitude [pA]')

#plot a scatter plot of max first event amplitude and time
D = fig.add_subplot(gs[1,1])
D.scatter(results['time_max_e1'],results['amplitude_max_e1'])
D.set_title("First max events")

#plot a linegraph of all idealized event traces
E = fig.add_subplot(gs[2,:])
for trace in results['trace']:
    E.plot(np.abs(trace))
"""

"""
Save the results
"""
save = False
if save:
    expanded_trace = pd.DataFrame(index=t_act.index,
                                  columns=np.arange(start=t0,
                                                    stop=t0 + max([x.shape[0] + 1 for x in results['trace']]) * 1000 / sampling_rate,
                                                    #sorry, its just the maximal trace length
                                                    step=1000 / sampling_rate).round(5),
                                  data=results['trace'].to_list())

    final_results = pd.concat([results.drop(columns='trace'), expanded_trace])
    final_results.to_csv(filedialog.asksaveasfile())
