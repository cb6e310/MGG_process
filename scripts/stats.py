import numpy as np
from scipy import stats
from scipy import signal

def pearsonr(x, y):
    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    len_x = len(x)
    len_y = len(y)
    if len_x < len_y:
        y = y[:len_x]
    elif len_x > len_y:
        x = x[:len_y]
    return stats.pearsonr(x, y)

def xcorr(x, y, maxlags=100):
    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    len_x = len(x)
    len_y = len(y)
    if len_x < len_y:
        y = y[:len_x]
    elif len_x > len_y:
        x = x[:len_y]
    x = (x - np.mean(x))/(np.std(x)*len(x))
    y = (y - np.mean(y))/(np.std(y))
    corr = signal.correlate(x, y, mode="full", method="auto")
    decimal_places = 2  # Set the desired number of decimal places
    corr = np.round(corr, decimal_places)

    return corr.max()