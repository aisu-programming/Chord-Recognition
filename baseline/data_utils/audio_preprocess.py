# usage: python audio_preprocess.py <dir_path> <target_path>

import librosa
import numpy as np
import os, sys
import threading

num_thread = 20

sr = 16000
hop_length = 2048
n_bins = 192
bins_per_octave = 24

def cqt_preprocess(file_path):
    y, _ = librosa.load(file_path, sr=sr)
    S_db = librosa.amplitude_to_db(np.abs(librosa.cqt(
        y, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)), ref=np.max)

    # Pad to x64
    frame_num = S_db.shape[1]
    if frame_num % 64 != 0:
        padding = np.ones((S_db.shape[0], 64 - (frame_num % 64))) * -80
        S_db = np.concatenate((S_db, padding), axis=1)
    return S_db

def preprocess(dir_path, target_path, file_name, semaphore):
    file_path = os.path.join(dir_path, file_name)

    semaphore.acquire()
    S_db = cqt_preprocess(file_path)
    semaphore.release()

    # Save to .npy file
    npy_path = os.path.join(target_path, os.path.splitext(file_name)[0] + ".npy")
    np.save(npy_path, S_db)
    print(file_name + " done.")