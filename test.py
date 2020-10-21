import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('CE200_sample/1/feature.json') as f:
    data = json.load(f)

target_list = [
    # 'chroma_stft',
    # 'chroma_cqt',
    'chroma_cens',
    # 'rms',
    # 'spectral_centroid',      # 頻譜質心：頻譜質心指示聲音的「質心」位於何處，並按照聲音的頻率的加權平均值來加以計算。
    # 'spectral_bandwidth',
    # 'spectral_contrast',
    # 'spectral_flatness',
    # 'spectral_rolloff',       # 譜滾降：對訊號形狀的測量，表示的是在譜能量的特定百分比 （如 85%）時的頻率。
    # 'poly_features',
    # 'tonnetz',
    # 'zero_crossing_rate'      # 過零率
]
chord_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

fig, axs = plt.subplots(len(target_list))
# fig.suptitle('Vertically stacked subplots')

for index_i, target_key in enumerate(target_list):
    
    if len(target_list) != 1: axs[index_i].set_title(target_key)
    target = data[target_key]

    for index_j, wave in enumerate(target):
        
        ax = axs if len(target_list) == 1 else axs[index_i]
        
        start_frame = 426
        end_frame = 564

        # start_frame = 2403
        # end_frame = 2476

        y = np.array(wave[start_frame:end_frame])
        x = np.arange(start_frame, end_frame)

        if len(target) == len(chord_name):
            ax.plot(x, y, label=chord_name[index_j])
            ax.legend(fontsize='small', ncol=2, loc='center left', bbox_to_anchor=(0.975, 0.5))
        else:
            ax.plot(x, y)

plt.show()