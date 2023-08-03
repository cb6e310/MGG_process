import scipy.signal as signal

from scipy.signal import butter, lfilter, freqz, sosfiltfilt, filtfilt
from sklearn.decomposition import FastICA, PCA
from skimage.restoration import denoise_wavelet

import numpy as np

import matplotlib.pyplot as plt

import os, sys

import emd, pydmd
import padasip as pa
import pywt
from vmdpy import VMD


from sobi import SOBI


def __butter_lowpass(cutoff, fs, order):
    return butter(order, cutoff, fs=fs, btype="low", analog=False, output="sos")


def __butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band", analog=False, output="sos")


def butter_filter(data, cutoff, fs, btype, order=5):
    data = np.squeeze(np.array(data))

    if btype == "lowpass":
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


def pca_filter(data):
    pca = PCA(n_components=1)
    pca.fit(data)
    dominant_component = pca.components_[0]
    return dominant_component


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

    def _direct_denoise(imf_noisy, select=[5, 6, 7]):
        denoised = imf_noisy[:, select]
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


def adaptive_filter(x, desired_output=None):
    x = pa.standardize(x)
    n = 50
    adaptive_filter = pa.filters.FilterRLS(n=n, mu=0.9)
    x_matrix = pa.input_from_history(x, n)[:-1]
    x = x[n:]
    y, e, w = adaptive_filter.run(x, x_matrix)
    # Apply online adaptive filtering
    return y


def sobi_filter(data):
    ica = SOBI(lags=10)
    # data = np.squeeze(np.array(data))
    ica.fit(data)
    return ica.transform(data)


def fastica_filter(data, n_components=8):
    data = np.squeeze(np.array(data))
    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(data.reshape(-1, 1))  # Reconstruct signals
    return S_


def vmd_filter(data, select=[5, 6, 7]):
    data = np.squeeze(np.array(data))

    # . some sample parameters for VMD
    alpha = 20000  # moderate bandwidth constraint
    tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
    K = 8  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    # . Run VMD
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)

    def _direct_denoise(imf_noisy, select=[5, 6, 7]):
        denoised = imf_noisy[:, select].sum(axis=1)
        return denoised

    return u.T


def dmd_filter(data, return_mode="reconstructed"):
    data = np.squeeze(np.array(data))
    mean = np.mean(data)
    scale = np.std(data)
    data = pa.standardize(data, offset=mean, scale=scale)
    hodmd = pydmd.HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(data[None])
    if return_mode == "reconstructed":
        filtered = hodmd.reconstructed_data[0].real
        return pa.preprocess.standardize_back(filtered, offset=mean, scale=scale)
    elif return_mode == "modes":
        imfs = hodmd.modes.T
        dynamic = hodmd.dynamics.T
        return imfs, dynamic
    pass


def calculate_baseline(signal):
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp**2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]


def wavelet_filter(data, wavelet, level=1, threshold=0.5):
    data = np.squeeze(np.array(data))
    denoised = denoise_wavelet(
        data,
        mode="soft",
        wavelet_levels=level,
        wavelet=wavelet,
        rescale_sigma="True",
    )

    return denoised
