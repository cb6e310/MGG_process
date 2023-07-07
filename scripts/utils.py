import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime, timedelta


from scipy import stats

from scipy import signal

from scipy.fft import fftshift

from sklearn.decomposition import FastICA 
from sklearn.preprocessing import normalize

import sobi
import os
import random


from nptdms import TdmsFile


class TDMSData:
    def __init__(self, folder_path, resample_mode="30ms", name_condition=None):
        """


        make sure alignment in one folder

        """

        self.folder_path = folder_path

        self.data_dict = {}

        self.column_names = []

        self.difference_column_names = []

        self.files = [f for f in os.listdir(self.folder_path) if name_condition in f]

        self.groups, self.column_names = read_tdms_properties(
            os.path.join(self.folder_path, self.files[0])
        )

        self.resample_mode = resample_mode

        self.freq = int(1/time_string_to_float(self.resample_mode))

        self.interval = 

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

            for i, ai in enumerate(ai_list):
                ai = ai.set_index(ai.columns[0])

                if resample_mode != None:
                    count = 0
                    ds_ai = []
                    ai = np.squeeze(ai.to_numpy())[::4000]
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

    def plot_samples(self, num):
        assert num <= 5

        sub_names = random.sample(list(self.data_dict), num)

        sub_values = [self.data_dict[name] for name in sub_names]

        # ai0.plot(x='/'data'/'Time (Dev2/ai0)'')

        self.fig = plt.figure(layout="constrained", figsize=(18, 18))

        subfigs = self.fig.subfigures(1, num, wspace=0.07)

        for i, name in enumerate(sub_names):
            axs = subfigs[i].subplots(len(sub_values[i]), 1)

            subfigs[i].suptitle(name)

            for j, ax in enumerate(axs):
                ax.set_xlabel("time")

                ax.set_ylabel("amp")

                print(sub_values[i][j])

                ax.plot(sub_values[i][j], linewidth=1)

                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

        plt.show()

    def plot_stats(self, idx, save=False):
        [stats_1, stats_2] = self.calculate_stats(idx)
        name_dict_1 = {
            "origin": ["amp", "time"],
            "psd": ["pT^2/Hz", "freq"],
            # "sobi": ["amp", "time"],
        }
        name_dict_2 = {"ica": ["amp", "time"]}
        sup_name = list(self.data_dict.keys())[idx]

        stats_0= list(self.data_dict.values())[idx]

        self.fig = plt.figure(layout="constrained", figsize=(20, 20))

        subfigs = self.fig.subfigures(
            1, 3, wspace=0.07
        )

        axs_0 = subfigs[0].subplots(len(stats_0), 1)
        subfigs[0].suptitle('origin')
        for j, ax in enumerate(axs_0):
            ax.set_xlabel("time")
            ax.set_ylabel("amp")
            ax.plot(stats_0[j], linewidth=1)
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

        axs_1 = subfigs[1].subplots(len(stats_1), 1)
        subfigs[1].suptitle(list(name_dict_1.keys())[1])
        for j, ax in enumerate(axs_1):
            ax.set_xlabel("freq")
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

        # for i, (name, labels) in enumerate(name_dict_1.items()):
        #     axs = subfigs[i].subplots(len(value), 1)
        #     subfigs[i].suptitle(name)
        #     current_stats = stats_1[i]
        #     for j, ax in enumerate(axs):
        #         ax.set_xlabel(labels[1])
        #         ax.set_ylabel(labels[0])
        #         if i == 1:
        #             ax.plot(current_stats[j][0], current_stats[j][1], linewidth=1)
        #             ax.set_xlim([0, 25])
        #         else:
        #             ax.plot(current_stats[j], linewidth=1)
        #         plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")
        # for i, (name, labels) in enumerate(name_dict_2.items()):
        #     axs = subfigs[i + len(name_dict_1)].subplots(len(name_dict_2), 1)
        #     subfigs[i + len(name_dict_1)].suptitle(name)
        #     for j, ax in enumerate(axs):
        #         ax.set_xlabel(labels[1])
        #         ax.set_ylabel(labels[0])
        #         ax.plot(stats_2, linewidth=1)

        if save == True:
            save_path = os.path.join(self.folder_path, "stats")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, sup_name + "_stats.pdf"))
        else:
            plt.show()

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
            f, pxx = self.calculate_psd(ai)
            # sobi_res = self.calculate_sobi(ai)

            psd_list.append([f, pxx])
            # sobi_list.append(sobi_res)
        return [psd_list, ica_res]

    def calculate_psd(self, arr):
        """

        return f_cpm, pxx
        """
        arr = np.squeeze(arr)
        interval = time_string_to_float(self.resample_mode)
        print(arr.shape)
        f, pxx = signal.welch(arr, fs=2 / interval, nperseg=512)
        return f * 60, pxx

    def calculate_ica(self, arr):
        """

        return [len(arr), n_components] array
        """

        transformer = FastICA(n_components=1)
        return transformer.fit_transform(arr)

    def calculate_sobi(self, arr):
        """
        return
        """

        s, _, _ = sobi.sobi(arr)
        return s


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
