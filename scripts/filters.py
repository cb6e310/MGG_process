import scipy.signal as signal

from scipy.signal import butter, lfilter, freqz

import numpy as np

import matplotlib.pyplot as plt

import os, sys


def __butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)

def __butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", analog=False)

def butter_filter(data, cutoff, fs, btype, order=5):
    if btype == "low":
        b, a = __butter_lowpass(cutoff, fs, order=order)
    elif btype == "bandpass":
        assert isinstance(cutoff, list)
        b, a = __butter_bandpass(cutoff[0], cutoff[1], fs, order=order)
    y = lfilter(b, a, data)
    return y

