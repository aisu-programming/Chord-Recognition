''' Libraries '''
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from process_chord import process_chords_13_6


''' Functions '''
def loss_function(y_real, y_pred, loss_objects, mid_idx):
    y_real_roots, y_real_qualities = y_real[:, mid_idx, :13], y_real[:, mid_idx, 13:]
    y_pred_roots, y_pred_qualities = y_pred[:, mid_idx, :13], y_pred[:, mid_idx, 13:]
    root_losses = loss_objects[0](y_real_roots, y_pred_roots)
    quality_losses = loss_objects[1](y_real_qualities, y_pred_qualities)
    # return tf.reduce_sum(root_losses*quality_losses)
    return tf.reduce_sum(root_losses*quality_losses)/len(root_losses)


def accuracy_function(y_real, y_pred, mid_idx):
    y_real_root_ids, y_real_quality_ids = tf.argmax(y_real[:, mid_idx, :13], axis=-1), tf.argmax(y_real[:, mid_idx, 13:], axis=-1)
    y_pred_root_ids, y_pred_quality_ids = tf.argmax(y_pred[:, mid_idx, :13], axis=-1), tf.argmax(y_pred[:, mid_idx, 13:], axis=-1)
    root_accs = tf.equal(y_real_root_ids, y_pred_root_ids)
    quality_accs = tf.equal(y_real_quality_ids, y_pred_quality_ids)
    accs = tf.cast(tf.logical_and(root_accs, quality_accs), dtype=tf.float32)
    # tf.print("")
    # tf.print("y_real_root_ids[0]   : ", y_real_root_ids[0], summarize=-1)
    # tf.print("y_pred_root_ids[0]   : ", y_pred_root_ids[0], summarize=-1)
    # tf.print("root_accs[0]         : ", root_accs[0], summarize=-1)
    # tf.print("")
    # tf.print("y_real_quality_ids[0]: ", y_real_quality_ids[0], summarize=-1)
    # tf.print("y_pred_quality_ids[0]: ", y_pred_quality_ids[0], summarize=-1)
    # tf.print("quality_accs[0]      : ", quality_accs[0], summarize=-1)
    # tf.print("")
    # tf.print("accs[0]              : ", accs[0], summarize=-1)
    # tf.print("")
    return tf.reduce_sum(accs)/len(accs)


def read_data(path, song_amount, sample_rate, hop_len):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{sample_rate}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_chords_13_6(chords))
    return x_dataset, y_dataset


def show_pred_and_truth(y_real, y_pred, mid_idx):

    PRECISION = 6
    np.set_printoptions(precision=PRECISION, linewidth=272, suppress=True)

    y_real = (y_real[0, mid_idx])[np.newaxis, :]
    y_pred = (y_pred[0, mid_idx].numpy())[np.newaxis, :]
    # Not sure
    processed_y_pred = np.concatenate([
        np.eye(13)[np.argmax(y_pred[:, :, :13], axis=-1)],
        np.eye(6)[np.argmax(y_pred[:, :, 13:], axis=-1)]],
        axis=-1
    )
    recognization_idx = np.concatenate([
        ((np.arange(13)+1)/(10**PRECISION))[np.newaxis, :],
        ((np.arange(6)+1)/(10**PRECISION))[np.newaxis, :],
        axis=-1
    )
    
    print("\nSample prediction: Shape =", y_pred.shape)
    print(y_pred)
    print("\nProcessed sample prediction: Shape =", y_pred.shape)
    print(np.concatenate([processed_y_pred, recognization_idx], axis=-0))
    print("\nSample ground truth: Shape =", y_real.shape)
    print(np.concatenate([y_real, recognization_idx], axis=-0))
    print('\n')
    
    return