import numpy as np

from tools import interval_selection, piezo_selection

def threshold_crossing(signal, amplitudes, thresholds=np.array([]), **kwargs):
    """
    Perform a threshold-crossing idealization on the signal.
    We assume that the current through the channel is always negative
    amplitudes must and thresholds must be given in descending order
    (i.e. increasing in absolute value) and include the baseline amplitude.
    Arguments:
        signal [1D numpy array] - data to be idealized
        amplitudes [1D numpy array] - supposed true amplitudes of signal
        thresholds [1D numpy array] - thresholds between the amplitudes
    If the wrong number of thresholds (or none) are given they will be replaced
    by the midpoint between the pairs of adjacent amplitudes.
    """

    if thresholds.size!=amplitudes.size-1:
        thresholds = (amplitudes[1:]+amplitudes[:-1])/2

    idealization = np.zeros(len(signal))
    #np.where returns a tuple containing array so we have to get the first element to get the indices
    inds = np.where(signal>thresholds[0])[0]
    idealization[inds] = amplitudes[0]

    for t, a in zip(thresholds,amplitudes[1:]):
        inds = np.where(signal<t)[0]
        idealization[inds] = a
    return idealization

# def multilevel_threshold(signal, thresholds, maxAmplitude=False,
#                          relativeThresholds=True):
#     """
#     performs threshold crossing for given thresholds. value should be
#     given as fraction of the maximum amplitude of the signal
#     """
#     if maxAmplitude:
#         signalmax = maxAmplitude
#     else:
#         maxInd = np.argmax(np.abs(signal))
#         signalmax = signal[maxInd]
#         # signalmax = np.max(np.abs(signal))
#
#     thresholds = np.append(thresholds, np.inf)
#
#     if relativeThresholds:
#         thresholds = signalmax*np.sort(thresholds)
#     else:
#         thresholds = np.sort(thresholds)
#     N = len(thresholds)
#     idealization = np.zeros(len(signal))
#
#     for i, point in enumerate(signal):
#         for n, threshold in enumerate(thresholds[:-1]):
#             if np.abs(threshold)<np.abs(point)<=np.abs(thresholds[n+1]):
#                 idealization[i]=(n+1)/N
#                 break
#     idealization *= signalmax
#     return idealization

def baseline_correction(time, signal, fs, intervals=None, time_unit='ms',
                        degree=1, method='poly',select_intvl=False,
                        piezo=None, select_piezo=False, active= False,
                        deviation=0.05):
    """
    Perform baseline correction by offsetting the signal with its mean
    in the selected intervals
    Parameters:
        time - 1D array containing times of the measurements in signal
               units of `time_unit`
        signal - time series of measurements
        intervals - interval or list of intervals from which to
                   estimate the baseline (in ms)
        fs - sampling frequency (in Hz)
        time_unit - units of the time vector, 'ms' or 's'
        method - `baseline` can subtract a fitted polynomial of
                 desired degree OR subtract the mean
        degree - if method is 'poly', the degree of the polynomial
    Returns:
        original signal less the fitted baseline
    """
    if select_intvl:
        t, s = interval_selection(time, signal, intervals, fs, time_unit)
    elif select_piezo:
        t, _, s = piezo_selection(time, piezo, signal, active, deviation)
    else:
        t = time
        s = signal

    if method == 'offset':
        offset = np.mean(s)
        output = signal - offset
    elif method == 'poly':
        coeffs = np.polyfit(t,s,degree)
        baseline = np.zeros_like(time)
        for i in range(degree+1):
            baseline+=coeffs[i]*(time**(degree-i))
        output = signal - baseline
    # print("baseline_correction output:")
    # print("shape :", output.shape)
    # print('\t', output)
    return output
