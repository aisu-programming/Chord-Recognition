''' Libraries '''
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from process_chord import process_chords


''' Functions '''
def make_dir():
    ckpt_dir = f"ckpt/{time.strftime('%Y.%m.%d_%H.%M', time.localtime())}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        os.makedirs(f"{ckpt_dir}/min_all_loss")
        os.makedirs(f"{ckpt_dir}/min_mid_loss")
        os.makedirs(f"{ckpt_dir}/max_all_acc")
        os.makedirs(f"{ckpt_dir}/max_mid_acc")
    return ckpt_dir


def loss_function(y_real, y_pred, loss_criterion):
    mid_idx  = y_real.shape[1] // 2
    all_loss = loss_criterion(y_real, y_pred)
    mid_loss = all_loss[:, mid_idx]
    return tf.reduce_mean(all_loss), tf.reduce_mean(mid_loss)


def accuracy_function(y_real, y_pred):
    mid_idx    = y_real.shape[1] // 2
    y_real_ids = tf.argmax(y_real, axis=-1)
    y_pred_ids = tf.argmax(y_pred, axis=-1)
    all_acc    = tf.cast(tf.equal(y_real_ids, y_pred_ids), dtype=tf.float32)
    mid_acc    = all_acc[:, mid_idx]
    return tf.reduce_mean(all_acc)*100, tf.reduce_mean(mid_acc)*100


def read_data(path, song_amount, sample_rate, hop_len, model_target, output_mode):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{sample_rate}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_chords(chords, model_target, output_mode))
    return x_dataset, y_dataset


def show_pred_and_truth(y_real, y_pred):

    PRECISION = 10
    np.set_printoptions(precision=PRECISION, edgeitems=8, linewidth=272, suppress=True)

    y_real = y_real[0]
    y_pred = y_pred[0].numpy()
    processed_y_pred = np.eye(y_pred.shape[-1])[np.argmax(y_pred, axis=-1)]
    recognization_idx = ((np.arange(y_pred.shape[-1])+1)/(10**PRECISION))[np.newaxis, :]
    
    print("Sample prediction: Shape =", y_pred.shape)
    print(y_pred)
    print("\nProcessed sample prediction: Shape =", y_pred.shape)
    print(np.concatenate([processed_y_pred, recognization_idx], axis=-0))
    print("\nSample ground truth: Shape =", y_real.shape)
    print(np.concatenate([y_real, recognization_idx], axis=-0))
    
    return