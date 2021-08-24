import random
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes


def gaussian_filter(data, sigma):
    if sigma == 0:
        return data
    else:
        new_data = []
        for i, row in enumerate(data.transpose()):
            if i == 24: new_data.append(row)
            else: new_data.append(gaussian_filter1d(row.astype(np.float64), sigma))
        return np.array(new_data).transpose()

def draw():
    song_no = 1
    interval = (400, 700)
    gaussian_sigmas = [0, 11, 12, 13, 14, 15, 16]
    original_data = pd.read_csv(f'CE200/{song_no}/data.csv', index_col=0).values

    fig, axs = plt.subplots(len(gaussian_sigmas))
    fig.set_size_inches(36, 18) # 2:1
    
    fig.suptitle(f'AI CUP 2020 - Song No.{song_no}: {interval[0]}~{interval[1]}')

    for i, sigma in enumerate(gaussian_sigmas):

        ax = axs if len(gaussian_sigmas) == 1 else axs[i]
        ax.set_title(f'Sigma = {sigma}')

        data = gaussian_filter(original_data, sigma)[interval[0]:interval[1]]
        yLabel = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        xLabel = data[:, 24]
        data = (data[:, :12] * 100).astype(np.int)

        #定義橫縱座標的刻度
        ax.set_yticks(range(len(yLabel)))
        ax.set_yticklabels(yLabel, fontsize=9)
        ax.set_xticks(range(len(xLabel)))
        ax.set_xticklabels(xLabel, fontsize=9, rotation=90)
        #作圖並選擇熱圖的顏色填充風格，這裡選擇hot
        im = ax.imshow(data.transpose(), cmap=plt.cm.hot_r)

    #增加右側的顏色刻度條
    # plt.colorbar(im)

    # save
    plt.subplots_adjust(top=.93, bottom=.07, left=.02, right=.99, hspace = .6)
    plt.savefig(f'test.png', dpi=250)
    # show
    # wm = plt.get_current_fig_manager()
    # wm.window.state('zoomed')
    # plt.show()

d = draw()