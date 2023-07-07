import noisereduce as nr

import matplotlib.pyplot as plt
import numpy as np

from utils import TDMSData

import os, sys

save_path = '../datasets/results/noise_reduced'
save = False

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    noise_folder_path = os.path.join(script_directory, '../datasets/noise')
    noise_data = TDMSData(noise_folder_path, name_condition='noise' ,resample_mode='30ms')
    data_folder_path = os.path.join(script_directory, '../datasets/MGG_human')

    total_data = TDMSData(data_folder_path, name_condition='MMG-W' ,resample_mode='30ms')

    principal_noise = noise_data.calculate_total_pca()
    
    result_list = []
    
    for i, name in enumerate(list(total_data.data_dict.keys())):
        ai_list = total_data.data_dict[name]
        current_reduced = []
        for j in range(len(ai_list)):
            ai_list[j] = np.resize(ai_list[j], (len(principal_noise),))
            current_reduced.append(nr.reduce_noise(y=ai_list[j], y_noise=principal_noise, sr=noise_data.desire_freq, stationary=True, time_mask_smooth_ms=None))

        current_fig=plt.figure(layout='constrained', figsize=(20, 20))
        current_fig.suptitle(name)
        current_subfigs = current_fig.subfigures(1, 2)
        axs_0 = current_subfigs[0].subplots(len(ai_list), 1)
        current_subfigs[0].suptitle('original')
        axs_1 = current_subfigs[1].subplots(len(ai_list), 1)
        current_subfigs[1].suptitle('reduced')
        for j in range(len(ai_list)):
            axs_0[j].plot(ai_list[j], linewidth=0.5)
            axs_1[j].plot(current_reduced[j], linewidth=0.5)
            axs_0[j].set_xlabel('time')
            axs_0[j].set_ylabel('amplitude')
            axs_1[j].set_xlabel('time')
            axs_1[j].set_ylabel('amplitude')

        if save:
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            current_fig.savefig(os.path.join(save_path, name + '.png'))
        plt.show()
        input('press any key to continue')
    # plot





