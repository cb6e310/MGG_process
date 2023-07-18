import scipy.signal as signal

from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt

import numpy as np

import matplotlib.pyplot as plt

import os, sys

import emd
import padasip as pa


def __butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype="low", analog=False, output="sos")


def __butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", analog=False, output="sos")


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


def emd_filter(data, emd_mode, denoise_mode, plot=False, select=None):
    """
    emd_mode: 0 for emd, 1 for ensemble emd
    denoise_mode: 0 for direct denoise, 1 for interval denoise
    """
    def _interval_denoise(imf_noisy):
        H = 0.8
        H_to_params = {
            0.2: {
                "beta": 0.487,
                "a": 0.452,
                "b": -1.951,
            },
            0.5: {
                "beta": 0.719,
                "a": 0.460,
                "b": -1.919,
            },
            0.8: {
                "beta": 1.025,
                "a": 0.495,
                "b": -1.833,
            },
        }

        energies_noisy = (
            np.array([np.power(e1, 2).sum() for e1 in imf_noisy.T]) / imf_noisy.shape[0]
        )

        beta = H_to_params[H]["beta"]
        rho = 2

        model_noisy = np.array(
            [energies_noisy[0]]
            + [
                (energies_noisy[0] / beta) * np.power(rho, -2 * (1 - H) * k)
                for k in range(1, energies_noisy.shape[0])
            ]
        )

        a = H_to_params[H]["a"]
        b = H_to_params[H]["b"]

        model_noisy_interval = (
            np.array(
                [np.exp(np.exp(a * k + b)) for k in range(energies_noisy.shape[0])]
            )
            * model_noisy
        )
        
        denoised = imf_noisy[:, energies_noisy > model_noisy_interval].sum(axis=1)
        return denoised

    def _direct_denoise(imf_noisy, select=[5,6,7]):
        denoised = imf_noisy[:, select].sum(axis=1)
        return denoised

    data = np.squeeze(np.array(data))
    if emd_mode == 0:
        IMFs = emd.sift.sift(data, max_imfs=8)
    elif emd_mode == 1:
        IMFs = emd.sift.ensemble_sift(data, max_imfs=8, nensembles=24, nprocesses=8)
    if denoise_mode == 0:
        denoised = _direct_denoise(IMFs, select=select)
    elif denoise_mode == 1:
        denoised = _interval_denoise(IMFs)

    # threshold = 0.5  # Set the threshold value
    # thresholded_IMFs = [imf * (np.abs(imf) > threshold) for imf in IMFs]

    # # Reconstruct the denoised signal
    # denoised_signal = np.sum(thresholded_IMFs, axis=0)

    return IMFs, denoised

def lms_filter(data):
    data = np.squeeze(np.array(data))
    f = pa.filters.FilterLMS(3, mu=0.1)
    y, e, w = f.run(data, data)