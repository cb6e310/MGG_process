import scipy.signal as signal

from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt

import numpy as np

import matplotlib.pyplot as plt

import os, sys

import emd


def __butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype="low", analog=False, output='sos')


def __butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", analog=False, output='sos')


def butter_filter(data, cutoff, fs, btype, order=5):
    data = np.squeeze(np.array(data))

    if btype == "low":
        sos = __butter_lowpass(cutoff, fs, order=order)
    elif btype == "bandpass":
        assert isinstance(cutoff, list) and len(cutoff) == 2 and cutoff[0] < cutoff[1]
        sos = __butter_bandpass(cutoff[0], cutoff[1], fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def notch_filter(data, fs, cutoff, Q=30):
    if isinstance(cutoff, list):
        for c in cutoff:
            b, a = signal.iirnotch(c, Q, fs)
            data = filtfilt(b, a, data)
    else:
        b, a = signal.iirnotch(cutoff, Q, fs)
        data = filtfilt(b, a, data)
    return data

def emd_filter(data):
    data = np.squeeze(np.array(data))
    emd_data = emd.sift.sift(data)
    return emd_data