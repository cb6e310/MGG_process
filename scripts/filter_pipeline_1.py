import utils

from utils import TDMSData

from filters import butter_filter, notch_filter, emd_filter


import os, sys

import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from matplotlib.colors import LogNorm


from scipy import stats

import emd
import numpy as np


tdms_path_1 = "../datasets/MMG_rat"

tdms_path_2 = "../datasets/MMG_3regions"

"""


tdms structure:


|_data(groups)


    |_Time (Dev2/ai0) datetime64[us]


    |_Dev2/ai0 float64


    |_Time (Dev2/ai1) datetime64[us]


    |_Dev2/ai1 float64


    |_Time (Dev2/ai0 1) datetime64[us]


    |_Dev2/ai0 1 float64


    |_...
"""

"""


make sure the tdms file name has the format of: **hz (hz is case insensitive)
"""


def plot_samples(*data, name=None):
    fig, ax = plt.subplots(len(data), 1, figsize=(18, 18))

    fig.suptitle(name)

    for i, d in enumerate(data):
        ax[i].plot(d)

        ax[i].set_xlabel("time")

        ax[i].set_ylabel("amplitude")

    plt.show()


def plot_psd(*data, fs, name=None, limit=None):
    fig, ax = plt.subplots(len(data), 1, figsize=(18, 18))

    fig.suptitle(name)

    for i, d in enumerate(data):
        ax[i].psd(d, Fs=fs, NFFT=1024, visible=True)

        ax[i].set_xlabel("frequency")

        ax[i].set_ylabel("PSD")

        if limit:
            ax[i].set_xlim(0, limit)

    plt.show()


def plot_psd_with_samples(*data, fs, name=None, limit=None, partial=None):
    """
    limit: the upper limit of frequency to show
    partial: the time interval to show
    """

    fig, ax = plt.subplots(len(data), 2, figsize=(18, 18))
    fig.suptitle(name)
    time = np.arange(0, len(data[0])) / fs
    for i, d in enumerate(data):
        ax[i][0].plot(time, d)
        ax[i][0].set_xlabel("time [s]")
        ax[i][0].set_ylabel("amplitude")
        if i != 0:
            if partial != None:
                ax[i][0].set_ylim(-50, 50)
            else:
                ax[i][0].set_ylim(-200, 200)

        f, t, sxx = TDMSData.calculate_spectrogram(d, fs)
        sxx = np.sqrt(sxx)
        im = ax[i][1].pcolormesh(
            t,
            f,
            sxx,
            shading="auto",
            cmap="viridis",
            norm=LogNorm(vmin=1e-3, vmax=5e1),
        )
        ax[i][1].set_xlabel("time [s]")
        ax[i][1].set_ylabel("frequency [Hz]")
        plt.colorbar(im, ax=ax[i][1])
        if limit:
            ax[i][1].set_ylim(0, limit)

        if partial != None:
            ax[i][0].set_xlim(partial[0], partial[1])
            ax[i][1].set_xlim(partial[0], partial[1])

    plt.show()


def plot_imfs(IMFs, fs, name=None, partial=None):
    colors = cm.rainbow(np.linspace(0, 1, IMFs.shape[1]))
    fig, ax = plt.subplots(IMFs.shape[1], 1, figsize=(18, 18))
    time = np.arange(0, len(IMFs[:, 0])) / fs

    for i in range(IMFs.shape[1]):
        ax[i].plot(time, IMFs[:, i], color=colors[i], linewidth=1)
        # set colors
        ax[i].set_title("IMF {}".format(i + 1))
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("amplitude")
        ax[i].set_ylim(-30, 30)
        if partial != None:
            ax[i].set_xlim(partial[0], partial[1])
    plt.show()
    input("press any key to continue")


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder_path = os.path.join(script_directory, "../datasets/MGG_human")
    folder_path_noise = os.path.join(script_directory, "../datasets/noise")
    MCG_path = os.path.join(script_directory, "../datasets/MCG")
    plank_path = os.path.join(script_directory, "../datasets/MCG/plank")
    # data = TDMSData(folder_path, name_condition='noise' ,resample_mode='30ms')

    # data_noise = TDMSData(folder_path=folder_path_noise, name_condition='noise' ,resample_mode=None)

    # data_2 = TDMSData(
    #     folder_path=folder_path, name_condition="MMG-W", resample_mode=None
    # )

    # data_noise.plot_psd(limit=500)

    # data_noise.plot_samples()

    # data_3 = TDMSData(folder_path=MCG_path, name_condition='MCG' ,resample_mode=None)

    data_4 = TDMSData(
        folder_path=plank_path, name_condition="pingbanzhicheng", resample_mode=None
    )

    partial = [70, 80]
    for name, data_wip in data_4.data_dict.items():
        data_wip_0 = data_wip[2].to_numpy()

        data_wip_1 = butter_filter(
            data_wip_0, [0.5, 45], data_4.desire_freq, btype="bandpass", order=4
        )
        # plot_samples(data_wip_0, data_wip_1, name=name)
        data_wip_2 = notch_filter(data_wip_1, data_4.desire_freq, [50, 40, 20])
        IMFs, data_wip_3 = emd_filter(data_wip_2, emd_mode=1, denoise_mode=0, plot=True, select=[5,6,7])

        plot_imfs(IMFs, data_4.desire_freq, name=name, partial=partial)

        plot_psd_with_samples(
            data_wip_0,
            data_wip_1,
            data_wip_2,
            data_wip_3,
            fs=data_4.desire_freq,
            name=name,
            limit=50,
            partial=partial
        )

        input("press any key to continue")

    # data_4.plot_samples()

    # data_3.plot_psd(limit=500)

    # data_2.plot_butter_lowpass(cutoff=20, order=6, limit=500, plot_psd=True)

    # data_2.plot_butter(

    #     cutoff=[0.5, 20], btype='bandpass', order=6, limit=500, plot_psd=True

    # )
