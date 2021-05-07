''' Libraries '''
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from process_chord import process_chords_63


''' Functions '''
def loss_function(y_real, y_pred, loss_objects, mid_idx):
    y_real = y_real[:, mid_idx]
    y_pred = y_pred[:, mid_idx]
    losses = loss_objects[0](y_real, y_pred)
    # return tf.reduce_sum(losses)
    return tf.reduce_sum(losses)/len(losses)


def accuracy_function(y_real, y_pred, mid_idx):
    y_real_ids = tf.argmax(y_real[:, mid_idx], axis=-1)
    y_pred_ids = tf.argmax(y_pred[:, mid_idx], axis=-1)
    accs = tf.cast(tf.equal(y_real_ids, y_pred_ids), dtype=tf.float32)
    return tf.reduce_sum(accs)/len(accs)


def read_data(path, song_amount, sample_rate, hop_len):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{sample_rate}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_chords_63(chords))
    return x_dataset, y_dataset


def show_pred_and_truth(y_real, y_pred, mid_idx):

    PRECISION = 3
    np.set_printoptions(precision=PRECISION, edgeitems=15, linewidth=272, suppress=True)

    y_real = (y_real[0, mid_idx])[np.newaxis, :]
    y_pred = (y_pred[0, mid_idx].numpy())[np.newaxis, :]
    processed_y_pred = np.eye(63)[np.argmax(y_pred, axis=-1)]
    recognization_idx = ((np.arange(63)+1)/(10**PRECISION))[np.newaxis, :]
    
    print("\nSample prediction: Shape =", y_pred.shape)
    print(y_pred)
    print("\nProcessed sample prediction: Shape =", y_pred.shape)
    print(np.concatenate([processed_y_pred, recognization_idx], axis=-0))
    print("\nSample ground truth: Shape =", y_real.shape)
    print(np.concatenate([y_real, recognization_idx], axis=-0))
    print('\n')
    
    return