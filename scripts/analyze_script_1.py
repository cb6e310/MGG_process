import utils


from utils import TDMSData


from filters import (
    butter_filter,
    notch_filter,
    emd_filter,
    pca_filter,
    vmd_filter,
    fastica_filter,
    sobi_filter,
    adaptive_filter,
    calculate_baseline,
    dmd_filter,
    wavelet_filter,
)


import os, sys, pickle
import pandas as pd


import matplotlib.pyplot as plt


from matplotlib.pyplot import cm


from matplotlib.colors import LogNorm


from stats import xcorr

import emd


import pywt


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


def plot_samples(*data, fs=None, name=None, partial=None):
    fig, ax = plt.subplots(len(data), 1, figsize=(18, 18))
    fig.suptitle(name)

    if len(data) == 1:
        if fs:
            time = np.arange(0, len(data[0])) / fs
            ax.plot(time, data[0])
        else:
            ax.plot(data[0])

        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")

        if partial != None:
            ax.set_xlim(partial[0], partial[1])
    else:
        for i, d in enumerate(data):
            if fs:
                time = np.arange(0, len(d)) / fs
                ax[i].plot(time, d)
            else:
                ax[i].plot(d)
            ax[i].set_xlabel("time")
            ax[i].set_ylabel("amplitude")

            if partial != None:
                ax[i].set_xlim(partial[0], partial[1])

    plt.show()


def plot_results(
    data_dict, fs=None, name=None, save_path=None, freq_unit="Hz", limit=None
):
    """
    fs: sampling frequency
    1: 2hz, 2: 20hz
    """
    fig = plt.figure(figsize=(18, 30), constrained_layout=True)
    colors = cm.rainbow(np.linspace(0, 1, len(data_dict)))
    fig.suptitle(name)
    subfigs = fig.subfigures(len(data_dict), 1)
    for i, (curr_name, data) in enumerate(data_dict.items()):
        ax = subfigs[i].subplots(1, 2)
        subfigs[i].suptitle(curr_name)
        if "1" in curr_name or "noise" in curr_name:
            curr_fs = fs[0]
        else:
            curr_fs = fs[1]
        time = np.arange(0, len(data)) / curr_fs
        ax[0].plot(time, data, color=colors[i])
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("amplitude")

        f, pxx = TDMSData.calculate_psd(data, curr_fs)
        if freq_unit == "hz":
            ax[1].plot(f, pxx)
            ax[1].set_xlabel("frequency [Hz]")
        elif freq_unit == "cpm":
            ax[1].plot(f * 60, pxx)
            ax[1].set_xlabel("frequency [cpm]")
        ax[1].set_ylabel("PSD")

        if limit:
            ax[1].set_xlim(0, limit)

    if save_path:
        save_path = os.path.join(save_path, "stats", "results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, name + ".png")
        print("saving to {}".format(save_path))
        plt.savefig(save_path)
    else:
        plt.show()


def plot_psd(*data, fs, name=None, limit=None):
    fig, ax = plt.subplots(len(data), 1, figsize=(18, 18))

    fig.suptitle(name)

    if len(data) == 1:
        ax.psd(data[0], Fs=fs, NFFT=1024, visible=True)

        ax.set_xlabel("frequency")

        ax.set_ylabel("PSD")

    else:
        for i, d in enumerate(data):
            ax[i].psd(d, Fs=fs, NFFT=1024, visible=True)

            ax[i].set_xlabel("frequency")

            ax[i].set_ylabel("PSD")

    if limit:
        ax[i].set_xlim(0, limit)

    plt.show()


def plot_sp_with_samples(*data, fs, name=None, limit=None, partial=None):
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


def plot_psd_with_samples(
    *data, fs, name=None, limit=None, partial=None, freq_unit="Hz", save_path=None
):
    """
    limit: the upper limit of frequency to show
    partial: the time interval to show
    """

    freq_unit = freq_unit.lower()

    fig, ax = plt.subplots(len(data), 2, figsize=(18, 18))

    fig.suptitle(name)

    if len(data) == 1:
        time = np.arange(0, len(data[0])) / fs

        ax[0].plot(time, data[0])

        ax[0].set_xlabel("time [s]")

        ax[0].set_ylabel("amplitude")

        f, pxx = TDMSData.calculate_psd(data[0], fs)

        if freq_unit == "hz":
            ax[1].plot(f, pxx)

            ax[1].set_xlabel("frequency [Hz]")

        elif freq_unit == "cpm":
            ax[1].plot(f * 60, pxx)

            ax[1].set_xlabel("frequency [cpm]")

        ax[1].set_ylabel("PSD")

        if limit:
            ax[1].set_xlim(0, limit)

        if partial != None:
            ax[0].set_xlim(partial[0], partial[1])

            ax[1].set_xlim(partial[0], partial[1])

    else:
        for i, d in enumerate(data):
            time = np.arange(0, len(d)) / fs

            ax[i][0].plot(time, d)

            ax[i][0].set_xlabel("time [s]")

            ax[i][0].set_ylabel("amplitude")

            if i != 0:
                if partial != None:
                    ax[i][0].set_ylim(-50, 50)

                else:
                    pass

                    # ax[i][0].set_ylim(-200, 200)

            f, pxx = TDMSData.calculate_psd(d, fs)

            if freq_unit == "hz":
                ax[i][1].plot(f, pxx)

                ax[i][1].set_xlabel("frequency [Hz]")

            elif freq_unit == "cpm":
                ax[i][1].plot(f * 60, pxx)

                ax[i][1].set_xlabel("frequency [cpm]")

            ax[i][1].set_ylabel("PSD")

            if limit:
                ax[i][1].set_xlim(0, limit)

            if partial != None:
                ax[i][0].set_xlim(partial[0], partial[1])

                ax[i][1].set_xlim(partial[0], partial[1])

    if save_path:
        if freq_unit == "cpm":
            psd_folder = "limit_psd"

        else:
            psd_folder = "psd"

        save_path = os.path.join(save_path, "stats", psd_folder)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, name + ".png")

        print("saving to {}".format(save_path))

        plt.savefig(save_path)

    else:
        plt.show()


def plot_imfs(IMFs, fs, name=None, partial=None, save_path=None):
    colors = cm.rainbow(np.linspace(0, 1, IMFs.shape[1]))

    fig, ax = plt.subplots(IMFs.shape[1], 1, figsize=(18, 18))

    time = np.arange(0, len(IMFs[:, 0])) / fs

    if IMFs.shape[1] == 1:
        ax.plot(time, IMFs, color=colors[0], linewidth=1)

        ax.set_title("IMF {}".format(1))

        ax.set_xlabel("time")

        ax.set_ylabel("amplitude")

        if partial != None:
            ax.set_xlim(partial[0], partial[1])

    else:
        for i in range(IMFs.shape[1]):
            ax[i].plot(time, IMFs[:, i], color=colors[i], linewidth=1)

            # set colors

            ax[i].set_title("IMF {}".format(i + 1))

            ax[i].set_xlabel("time")

            ax[i].set_ylabel("amplitude")

            # ax[i].set_ylim(-30, 30)

            if partial != None:
                ax[i].set_xlim(partial[0], partial[1])

    if save_path:
        save_path = os.path.join(save_path, "stats", "IMFs")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, name + ".png")

        print("saving to {}".format(save_path))

        plt.savefig(save_path)

    else:
        plt.show()

        input("press any key to continue")


if __name__ == "__main__":
    partial = [70, 80]

    clip_second = 5

    noise_selected = ["01_11_55", "01_15_02"]

    MGG_band_1 = [1 / 60, 2]

    MGG_band_2 = [1 / 60, 20]

    wav = "db4"

    emd_selected = [1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3]

    subjects = ["ZJW", "SCX", "ZJL", "FF"]

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder_path_noise = os.path.join(script_directory, "../datasets/noise")
    folder_path_MGG = os.path.join(script_directory, "../datasets/MGG_human")
    folder_path_save = os.path.join(
        script_directory, "../datasets/MGG_human/processed_1"
    )
    processed = True

    if not processed:
        print("processing data")

        # noise
        data_noise = TDMSData(
            folder_path=folder_path_noise, name_condition="noise", resample_mode=None
        )
        noise_length = len(list(data_noise.data_dict.values())[0][2])

        up2down_noise = None
        down2up_noise = None

        for name, data_wip in data_noise.data_dict.items():
            if noise_selected[0] in name or noise_selected[1] in name:
                noise_wip_0 = np.squeeze(data_wip[2].to_numpy())

                filtered_noise_ = butter_filter(
                    noise_wip_0,
                    [1 / 60, 2],
                    data_noise.desire_freq,
                    btype="bandpass",
                    order=4,
                )

                filtered_noise_ = np.resize(filtered_noise_, noise_length)
                filtered_noise = filtered_noise_[clip_second * data_noise.desire_freq :]
                filtered_noise = filtered_noise[
                    :: data_noise.desire_freq // MGG_band_1[1]
                ]
                if noise_selected[1] in name:
                    up2down_noise = filtered_noise
                else:
                    down2up_noise = filtered_noise

        subjects_data = {}
        data_MGG = TDMSData(
            folder_path=folder_path_MGG,
            name_condition="-MMG-W",
            resample_mode=None,
        )

        for i, (name, data_wip) in enumerate(data_MGG.data_dict.items()):
            print("processing {}".format(name))
            MGG_wip_0 = np.squeeze(data_wip[2].to_numpy())
            ECG_wip_0 = np.squeeze(data_wip[3].to_numpy())
            ECG_wip_1 = butter_filter(
                ECG_wip_0, MGG_band_2, data_MGG.desire_freq, btype="bandpass", order=4
            )
            ECG_wip_1 = ECG_wip_1[clip_second * data_MGG.desire_freq :]
            ECG_wip_1 = ECG_wip_1[:: data_MGG.desire_freq // MGG_band_2[1]]

            MGG_wip_1 = butter_filter(
                MGG_wip_0, MGG_band_1, data_MGG.desire_freq, btype="bandpass", order=4
            )
            MGG_wip_1 = MGG_wip_1[clip_second * data_MGG.desire_freq :]
            MGG_wip_1 = MGG_wip_1[:: data_MGG.desire_freq // MGG_band_1[1]]

            if MGG_wip_1[0] * ECG_wip_1[0] < 0:
                ECG_wip_1 = -ECG_wip_1
            if MGG_wip_1[0] > 0:
                current_noise = up2down_noise
            else:
                current_noise = down2up_noise

            IMFs, MGG_wip_emd_1 = emd_filter(
                MGG_wip_1, emd_mode=0, denoise_mode=0, plot=True, select=emd_selected[i]
            )

            MGG_wip_wavelet_1 = wavelet_filter(MGG_wip_1, wavelet=wav)

            MGG_wip_dmd_1 = dmd_filter(MGG_wip_1, return_mode="reconstructed")

            MGG_wip_2 = butter_filter(
                MGG_wip_0, MGG_band_2, data_MGG.desire_freq, btype="bandpass", order=4
            )

            MGG_wip_2 = MGG_wip_2[clip_second * data_MGG.desire_freq :]

            MGG_wip_2 = MGG_wip_2[:: data_MGG.desire_freq // MGG_band_2[1]]

            IMFs, MGG_wip_emd_2 = emd_filter(
                MGG_wip_2, emd_mode=0, denoise_mode=0, plot=True, select=emd_selected[i]
            )

            MGG_wip_wavelet_2 = wavelet_filter(MGG_wip_2, wavelet=wav)

            MGG_wip_dmd_2 = dmd_filter(MGG_wip_2, return_mode="reconstructed")

            subjects_data[name] = {
                "MGG_1": MGG_wip_1,
                "EMD_1": MGG_wip_emd_1,
                "DMD_1": MGG_wip_dmd_1,
                "Wavelet_1": MGG_wip_wavelet_1,
                "MGG_2": MGG_wip_2,
                "EMD_2": MGG_wip_emd_2,
                "DMD_2": MGG_wip_dmd_2,
                "Wavelet_2": MGG_wip_wavelet_2,
                "ECG": ECG_wip_1,
                "noise": current_noise,
            }

        if not os.path.exists(folder_path_save):
            os.makedirs(folder_path_save)

        print("saving data")

        with open(os.path.join(folder_path_save, "subjects_data.pkl"), "wb") as f:
            pickle.dump(subjects_data, f)

    # load data
    else:
        print("loading data")

        subjects_data = {}

        with open(os.path.join(folder_path_save, "subjects_data.pkl"), "rb") as f:
            subjects_data = pickle.load(f)

    results = {}

    for i, (name, data) in enumerate(subjects_data.items()):
        results[name] = {}
        results[name]["origin_MGG/ECG"] = xcorr(data["MGG_2"], data["ECG"])
        results[name]["EMD_MGG/ECG"] = xcorr(data["EMD_2"], data["ECG"])
        results[name]["DMD_MGG/ECG"] = xcorr(data["DMD_2"], data["ECG"])
        results[name]["Wavelet_MGG/ECG"] = xcorr(data["Wavelet_2"], data["ECG"])

        results[name]["origin_MGG/Noise"] = xcorr(data["MGG_1"], data["noise"])
        results[name]["EMD_MGG/Noise"] = xcorr(data["EMD_1"], data["noise"])

        results[name]["DMD_MGG/Noise"] = xcorr(data["DMD_1"], data["noise"])

        results[name]["Wavelet_MGG/Noise"] = xcorr(data["Wavelet_1"], data["noise"])
        plot_results(
            data,
            name=name,
            fs=[MGG_band_1[1], MGG_band_2[1]],
            save_path=folder_path_save,
            freq_unit="cpm",
            limit=25,
        )

    results_df = pd.DataFrame(results).T

    results_df.to_csv(os.path.join(folder_path_save, "results.csv"))
    print("saved to {}".format(os.path.join(folder_path_save, "results.csv")))
