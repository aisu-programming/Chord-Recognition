''' Libraries '''
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from convert import find_quality
from process_chord import process_answers


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


def loss_function(y_real, y_pred, pred_mode, loss_criterion):
    if pred_mode == 'integrate' or pred_mode == 'root' or pred_mode == 'quality':
        mid_idx  = y_real.shape[1] // 2
        all_loss = loss_criterion(y_real, y_pred)
        mid_loss = all_loss[:, mid_idx]
        return tf.reduce_mean(all_loss), tf.reduce_mean(mid_loss)
    elif pred_mode == 'quality_bitmap':
        mid_idx  = y_real.shape[1] // 2
        all_loss = loss_criterion(y_real, y_pred)
        mid_loss = all_loss[:, mid_idx]
        return tf.reduce_mean(all_loss), tf.reduce_mean(mid_loss)
    elif pred_mode == 'separate':
        mid_idx     = y_real.shape[1] // 2
        y_real_root = y_real[:, :, :13]
        y_pred_root = y_pred[:, :, :13]
        y_real_qual = y_real[:, :, 13:]
        y_pred_qual = y_pred[:, :, 13:]
        root_loss   = loss_criterion(y_real_root, y_pred_root)
        qual_loss   = loss_criterion(y_real_qual, y_pred_qual)
        all_loss    = root_loss * qual_loss
        mid_loss    = all_loss[:, mid_idx]
        return tf.reduce_mean(all_loss), tf.reduce_mean(mid_loss)
    else:
        raise Exception


def bitmap_to_number(batchs):
    batchs     = tf.cast(batchs, dtype=tf.int16)
    new_batchs = tf.zeros(batchs.shape[0:2], dtype=tf.int16)
    for i in range(12):
        new_batchs += batchs[:, :, i]
        if i != 11: tf.bitwise.left_shift(new_batchs, 1)
    return new_batchs


def accuracy_function(y_real, y_pred, pred_mode):
    if pred_mode == 'integrate' or pred_mode == 'root' or pred_mode == 'quality':
        mid_idx    = y_real.shape[1] // 2
        y_real_ids = tf.argmax(y_real, axis=-1)
        y_pred_ids = tf.argmax(y_pred, axis=-1)
        all_acc    = tf.cast(tf.equal(y_real_ids, y_pred_ids), dtype=tf.float32)
        mid_acc    = all_acc[:, mid_idx]
        return tf.reduce_mean(all_acc)*100, tf.reduce_mean(mid_acc)*100
    elif pred_mode == 'quality_bitmap':
        mid_idx = y_real.shape[1] // 2
        all_acc = tf.cast(tf.equal(bitmap_to_number(y_real), bitmap_to_number(y_pred)), dtype=tf.float32)
        y_real  = tf.reshape(y_real, shape=(y_real.shape[0] * y_real.shape[1], y_real.shape[2]))
        y_pred  = tf.reshape(y_pred, shape=(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]))
        y_real  = [ find_quality(bitmap, set()) for bitmap in y_real ]
        y_pred  = [ find_quality(bitmap, set()) for bitmap in y_pred ]
        mid_acc = all_acc[:, mid_idx]
        return tf.reduce_mean(all_acc)*100, tf.reduce_mean(mid_acc)*100
    elif pred_mode == 'separate':
        mid_idx         = y_real.shape[1] // 2
        y_real_root_ids = tf.argmax(y_real[:, :, :13], axis=-1)
        y_pred_root_ids = tf.argmax(y_pred[:, :, :13], axis=-1)
        y_real_qual_ids = tf.argmax(y_real[:, :, 13:], axis=-1)
        y_pred_qual_ids = tf.argmax(y_pred[:, :, 13:], axis=-1)
        root_acc        = tf.cast(tf.equal(y_real_root_ids, y_pred_root_ids), dtype=tf.float32)
        qual_acc        = tf.cast(tf.equal(y_real_qual_ids, y_pred_qual_ids), dtype=tf.float32)
        all_acc         = root_acc * qual_acc
        mid_acc         = all_acc[:, mid_idx]
        return tf.reduce_mean(all_acc)*100, tf.reduce_mean(mid_acc)*100
    elif pred_mode == 'root':
        mid_idx         = y_real.shape[1] // 2
        y_real_root_ids = tf.argmax(y_real[:, :, :13], axis=-1)
        y_pred_root_ids = tf.argmax(y_pred[:, :, :13], axis=-1)
        all_acc         = tf.cast(tf.equal(y_real_root_ids, y_pred_root_ids), dtype=tf.float32)
        mid_acc         = all_acc[:, mid_idx]
        return tf.reduce_mean(all_acc)*100, tf.reduce_mean(mid_acc)*100
    else:
        raise Exception


def read_data(path, song_amount, sample_rate, hop_len, model_target, output_mode):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{sample_rate}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_answers(chords, model_target, output_mode))
    return x_dataset, y_dataset


def show_pred_and_truth(y_real, y_pred, pred_mode):

    np.set_printoptions(formatter={'all':lambda x: f"{x:2.0f}"}, linewidth=272)
    # PRECISION = 0
    # np.set_printoptions(precision=PRECISION, linewidth=272)
    # np.set_printoptions(precision=PRECISION, edgeitems=15, linewidth=272, suppress=True)

    mid_idx = len(y_real)//2
    if pred_mode == 'integrate' or pred_mode == 'root' or pred_mode == 'quality':
        argmax_y_pred = np.argmax(y_pred[mid_idx], axis=-1)
        argmax_y_real = np.argmax(y_real[mid_idx], axis=-1)
        print('')
        print("Pred shape:", y_pred.shape)
        print("Real shape:", y_real.shape, '\n')
        print(argmax_y_pred)
        print(argmax_y_real, '\n\n')
    elif pred_mode == 'quality_bitmap':
        print('')
        print("Pred shape:", y_pred.shape)
        print("Real shape:", y_real.shape, '\n')
        mid_idx_2 = len(y_real[mid_idx])//2
        print(y_pred[mid_idx, mid_idx_2-5:mid_idx_2+5])
        print(y_real[mid_idx, mid_idx_2-5:mid_idx_2+5], '\n\n')
    elif pred_mode == 'separate':
        argmax_y_real_root = np.argmax(y_real[mid_idx, :, :13], axis=-1)
        argmax_y_pred_root = np.argmax(y_pred[mid_idx, :, :13], axis=-1)
        argmax_y_real_qual = np.argmax(y_real[mid_idx, :, 13:], axis=-1)
        argmax_y_pred_qual = np.argmax(y_pred[mid_idx, :, 13:], axis=-1)
        print('')
        print("Pred shape:", y_pred.shape)
        print("Real shape:", y_real.shape, '\n')
        print("root   :", argmax_y_real_root)
        print("root   :", argmax_y_pred_root, '\n')
        print("quality:", argmax_y_real_qual)
        print("quality:", argmax_y_pred_qual, '\n\n')
    
    return