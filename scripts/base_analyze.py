import utils
from utils import TDMSData


import os, sys

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

tdms_path_1 = '../datasets/MMG_rat'
tdms_path_2 = '../datasets/MMG_3regions'

'''
tdms structure:
|_data(groups)
    |_Time (Dev2/ai0) datetime64[us]
    |_Dev2/ai0 float64
    |_Time (Dev2/ai1) datetime64[us]
    |_Dev2/ai1 float64
    |_Time (Dev2/ai0 1) datetime64[us]
    |_Dev2/ai0 1 float64
    |_...
'''

'''
make sure the tdms file name has the format of: **hz (hz is case insensitive)
'''


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder_path = os.path.join(script_directory, '../datasets/noise')
    data = TDMSData(folder_path, name_condition='noise' ,resample_mode='30ms')
    # data.plot_samples(num=3)
    data.save_stats()