import utils

from utils import TDMSData


import os, sys

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

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
    data_4 = TDMSData(folder_path=plank_path, name_condition='pingbanzhicheng' ,resample_mode=None)
    # data_4.plot_samples()
    data_4.plot_spectrogram(limit=100)
    # data_3.plot_psd(limit=500)

    # data_2.plot_butter_lowpass(cutoff=20, order=6, limit=500, plot_psd=True)
    # data_2.plot_butter(
    #     cutoff=[0.5, 20], btype='bandpass', order=6, limit=500, plot_psd=True
    # )
