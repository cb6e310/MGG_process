import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

from datetime import datetime, timedelta


from scipy import stats


from scipy import signal


from scipy.fft import fftshift


from sklearn.decomposition import FastICA, PCA


from sklearn.preprocessing import StandardScaler


import sobi
import os
import random
import re


from nptdms import TdmsFile


from filters import butter_filter


class TDMSData:
    def __init__(
        self, folder_path, resample_mode="30ms", name_condition=None, normalize=False
    ):
        """




        make sure alignment in one folder
        """

        self.folder_path = folder_path

        self.data_dict = {}

        self.column_names = []

        self.difference_column_names = []

        self.files = [f for f in os.listdir(self.folder_path) if name_condition in f]

        self.resample_mode = resample_mode

        # calculate resample interval

        freq_pattern = r"(\d+)HZ"

        self.original_freq = int(
            re.findall(freq_pattern, self.files[0], re.IGNORECASE)[0]
        )

        if self.resample_mode is not None:
            self.desire_freq = int(1 / time_string_to_float(self.resample_mode))

            self.interval = int(self.original_freq / self.desire_freq)

        else:
            self.desire_freq = self.original_freq

            self.interval = 1

        self.group_names, self.column_names = read_tdms_properties(
            os.path.join(self.folder_path, self.files[0])
        )

        for file in self.files:
            if not file.endswith(".tdms"):
                continue

            data = read_tdms2df(os.path.join(self.folder_path, file))

            data.columns = self.column_names

            # change file time axis type to pd time

            for i, column in enumerate(self.column_names):
                if "Time" in column:
                    data[column] = pd.to_datetime(data[column])

                if " " in column:
                    self.difference_column_names.append(column)

            # generate channel list

            ai_list = [
                data.iloc[:, i : i + 2] for i in range(0, len(self.column_names), 2)
            ]

            self.scaler = StandardScaler()

            for i, ai in enumerate(ai_list):
                ai = ai.set_index(ai.columns[0])

                if resample_mode != None:
                    count = 0

                    ds_ai = []

                    ai = np.squeeze(ai.to_numpy())[:: self.interval]

                    ai_list[i] = pd.DataFrame(ai, columns=[ai_list[i].columns[0]])

                    ai_list[i] = remove_outliers(ai_list[i], ai_list[i].columns[0])

                else:
                    ai_list[i] = remove_outliers(ai, ai.columns[0])

            # print(ai_list)

            self.data_dict.update({file: ai_list})

    def properties(self):
        print("channels:", self.column_names)

        print("segment count:", len(self.data_dict))

        print(
            "segment length stat (max, min, mean):",
            self.average_mater(self.data_dict.values()),
        )

        for name, ai_list in self.data_dict.items():
            time = ai_list[0].index

            print("duration:", time.max() - time.min())

            break

        return {
            "channels": self.column_names,
            "segment count": len(self.data_dict),
            "segment length stat (max, min, mean)": self.average_mater(
                self.data_dict.values()
            ),
        }

    def plot_samples(self):
        # ai0.plot(x='/'data'/'Time (Dev2/ai0)'')

        for name, ai_list in self.data_dict.items():
            fig, axs = plt.subplots(len(ai_list), 1, figsize=(20, 20))

            fig.suptitle(name)

            for j, ax in enumerate(axs):
                ax.set_xlabel("time")

                ax.set_ylabel("amp")

                if self.resample_mode != None:
                    ax.plot(ai_list[j], linewidth=1)

                else:
                    ax.plot(ai_list[j][:10000], linewidth=0.5)

                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

            plt.show()

            input("press any key to continue")

    def plot_stats(self, idx, save=False):
        [stats_1, stats_2] = self.calculate_stats(idx)

        name_dict_1 = {
            "origin": ["amp", "time"],
            "psd": ["pT^2/Hz", "freq"],
            # "sobi": ["amp", "time"],
        }

        name_dict_2 = {"ica": ["amp", "time"]}

        sup_name = list(self.data_dict.keys())[idx]

        stats_0 = list(self.data_dict.values())[idx]

        self.fig = plt.figure(layout="constrained", figsize=(20, 20))

        subfigs = self.fig.subfigures(1, 3, wspace=0.07)

        axs_0 = subfigs[0].subplots(len(stats_0), 1)

        subfigs[0].suptitle("origin")

        for j, ax in enumerate(axs_0):
            ax.set_xlabel("time")

            ax.set_ylabel("amp")

            ax.plot(stats_0[j], linewidth=1)

            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

        axs_1 = subfigs[1].subplots(len(stats_1), 1)

        subfigs[1].suptitle(list(name_dict_1.keys())[1])

        for j, ax in enumerate(axs_1):
            ax.set_xlabel("freq cpm")

            ax.set_ylabel("pT^2/Hz")

            ax.plot(stats_1[j][0], stats_1[j][1], linewidth=1)

            ax.set_xlim([0, 50])

            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

        axs_2 = subfigs[2].subplots(1, 1)

        subfigs[2].suptitle(list(name_dict_2.keys())[0])

        axs_2.set_xlabel("time")

        axs_2.set_ylabel("amp")

        axs_2.plot(stats_2, linewidth=1)

        plt.setp(axs_2.get_xticklabels(), rotation=30, horizontalalignment="right")

        if save == True:
            save_path = os.path.join(self.folder_path, "stats")

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, sup_name + "_stats.pdf"))

        else:
            plt.show()

    def plot_psd(self, limit=50, freq_unit="hz"):
        for name, ai_list in self.data_dict.items():
            fig, axs = plt.subplots(len(ai_list), 1, figsize=(20, 20))

            for j, ax in enumerate(axs):
                current_psd = self.calculate_psd(
                    ai_list[j], fs=self.desire_freq, freq_unit=freq_unit
                )

                ax.set_xlabel("freq: " + freq_unit)

                ax.set_ylabel("W/Hz")

                ax.plot(current_psd[0], current_psd[1], linewidth=1)

                if limit != None:
                    ax.set_xlim([0, limit])

                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

            plt.show()

            input("press any key to continue")

    def plot_butter(self, cutoff, btype, order=5, limit=50, plot_psd=True, save=False):
        for name, ai_list in self.data_dict.items():
            fig = plt.figure(figsize=(18, 18))

            subfigs = fig.subfigures(1, 2, wspace=0.07)

            subfigs[0].suptitle("origin_psd")

            subfigs[1].suptitle("filtered_psd")

            axs_0 = subfigs[0].subplots(len(ai_list), 1)

            axs_1 = subfigs[1].subplots(len(ai_list), 1)

            for j, ax in enumerate(axs_0):
                if plot_psd == True:
                    current_psd = self.calculate_psd(
                        ai_list[j], fs=self.desire_freq, freq_unit="hz"
                    )

                    ax.set_xlabel("freq: hz")

                    ax.set_ylabel("W/Hz")

                    ax.plot(current_psd[0], current_psd[1], linewidth=1)

                    if limit != None:
                        ax.set_xlim([0, limit])

                else:
                    ax.set_xlabel("time")

                    ax.set_ylabel("amp")

                    ax.plot(ai_list[j], linewidth=1)

                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

            for j, ax in enumerate(axs_1):
                filtered_data = butter_filter(
                    ai_list[j], cutoff, self.desire_freq, btype, order=order
                )

                if plot_psd == True:
                    current_psd = self.calculate_psd(
                        filtered_data, fs=self.desire_freq, freq_unit="hz"
                    )

                    # assert filtered_data != ai_list[j]

                    ax.set_xlabel("freq: hz")

                    ax.set_ylabel("W/Hz")

                    ax.plot(current_psd[0], current_psd[1], linewidth=1)

                    if limit != None:
                        ax.set_xlim([0, limit])

                else:
                    ax.set_xlabel("time")

                    ax.set_ylabel("amp")

                    ax.plot(filtered_data, linewidth=1)

                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

            if save != True:
                plt.show()

                input("press any key to continue")

        if save == True:
            save_path = os.path.join(self.folder_path, "filtered")

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, name + "_filtered.pdf"))

    @staticmethod
    def average_mater(data: list):
        _max = 0

        _min = 1000000

        _sum = 0

        for i, ai_list in enumerate(data):
            if type(ai_list) == list:
                _len = len(ai_list[0])

            else:
                _len = len(ai_list)

            if _max < _len:
                _max = _len

            if _min > _len:
                _min = _len

            _sum += _len

        return _max, _min, _sum / len(data)

    def save_stats(self):
        for i, _ in enumerate(self.files):
            self.plot_stats(i, save=True)

    def calculate_stats(self, idx):
        name = list(self.data_dict.keys())[idx]

        value = list(self.data_dict.values())[idx]

        psd_list = []

        sobi_list = []

        value_np = []

        for i, ai in enumerate(value):
            ai = np.squeeze(ai.to_numpy())

            if i != 2:
                value_np.append(ai)

        value_np = np.array(value_np).T

        value_np = normalize(X=value_np, axis=0)

        ica_res = self.calculate_ica(value_np)

        for ai in value:
            ai = ai.to_numpy()

            ai /= ai.std(axis=0)

            f, pxx = self.calculate_psd(ai, self.desire_freq)

            # sobi_res = self.calculate_sobi(ai)

            psd_list.append([f, pxx])

            # sobi_list.append(sobi_res)

        return [psd_list, ica_res]

    @staticmethod
    def calculate_psd(arr, fs, freq_unit="hz"):
        """


        return f_cpm, pxx
        """

        arr = np.squeeze(arr)

        f, pxx = signal.welch(arr, fs=fs, nperseg=1024)

        if freq_unit == "hz":
            return f, pxx

        elif freq_unit == "cpm":
            return f * 60, pxx

        else:
            raise ("wrong freq unit")

    @staticmethod
    def calculate_ica(arr):
        """


        return [len(arr), n_components] array
        """

        transformer = FastICA(n_components=1)
        return transformer.fit_transform(arr)

    @staticmethod
    def calculate_pca(arr):
        """





        return [len(arr), n_components] array
        """

        transformer = PCA(n_components=1)
        return transformer.fit_transform(arr)

    def calculate_sobi(arr):
        """
        return
        """

        s, _, _ = sobi.sobi(arr)
        return s

    def calculate_total_pca(self):
        """




        return [len(arr), n_components] array
        """

        channel_num = len(list(self.data_dict.values())[0])

        channel_values = []

        length = len(list(self.data_dict.values())[0][0])

        for i in range(channel_num):
            for j in range(len(self.data_dict)):
                current_value = list(self.data_dict.values())[j][i]

                current_value = np.resize(current_value, (length, 1))

                channel_values.append(current_value)

        channel_values = np.squeeze(np.array(channel_values)).T

        res = self.calculate_pca(channel_values)

        return res


def remove_outliers(df, column, alpha=3):
    return df[(np.abs(stats.zscore(df[column])) < alpha)]


def read_tdms_properties(tdms_file_path):
    channel_list = []

    group_list = []

    with TdmsFile.open(tdms_file_path) as tdms_file:
        # Iterate over all items in the file properties and print them

        for group in tdms_file.groups():
            group_name = group.name

            group_list.append(group_name)

            print("group:", group_name)

            for channel in group.channels():
                channel_name = channel.name

                channel_list.append(channel_name)

                print("channel:", channel_name)

                # Access dictionary of properties:

                properties = channel.properties

                # Access numpy array of data for channel:

                data = channel[:]

                # Access a subset of data

                data_subset = channel[100:200]

                # print("data:", data_subset)

    print("channels: ", channel_list)

    print("groups: ", group_list)

    return group_list, channel_list


def read_tdms2df(tdms_file_path):
    with TdmsFile.open(tdms_file_path) as tdms_file:
        return tdms_file.as_dataframe()


def time_string_to_float(time_string):
    # Parse the time string

    # Convert the unit to a timedelta object

    if time_string.endswith("ms"):
        timedelta_unit = timedelta(milliseconds=1)

        value = float(time_string[:-2])

    elif time_string.endswith("us"):
        timedelta_unit = timedelta(microseconds=1)

        value = float(time_string[:-2])

    elif time_string.endswith("s"):
        timedelta_unit = timedelta(seconds=1)

        value = float(time_string[:-1])

    elif time_string.endswith("min"):
        timedelta_unit = timedelta(minutes=1)

        value = float(time_string[:-3])

    elif time_string.endswith("h"):
        timedelta_unit = timedelta(hours=1)

        value = float(time_string[:-1])

    elif time_string.endswith("d"):
        timedelta_unit = timedelta(days=1)

        value = float(time_string[:-1])

    else:
        raise ValueError("Invalid time unit")

    # Perform the conversion

    result = value * timedelta_unit.total_seconds()
    return result
