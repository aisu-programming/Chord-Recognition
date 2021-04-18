# usage: python predict <audio_path> <target_path> <model_path>

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import ast
import json
import sys
import os

# from CNN_2D_sp_att import Net
from net import Net
from trainer import *

from data_utils.audio_preprocess import cqt_preprocess
from data_utils.post_process import PostProcessor, BPM_selector
from data_utils.chord_utils import get_chord_str
from chord_mapping import *

from librosa import get_duration
from datetime import datetime
from collections import OrderedDict

from madmom.features.beats import RNNBeatProcessor
from madmom.features.beats import BeatTrackingProcessor
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.onsets import RNNOnsetProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

def _predict(estimation, quality_list, sec_per_frame, min_duration, quality_mapping=mapping_default, verbose=False):
    mapping = {}
    for quality, new_quality in quality_mapping.items():
        mapping[quality_list.index(quality)] = quality_list.index(new_quality)

    predict_list = []
    pre_chord = "N"
    pre_root = 12
    pre_quality = 0
    pre_time = 0
    for idx, colx in enumerate(estimation.T):
        root = np.argmax(colx[:13])
        quality = np.argmax(colx[13:])
        quality = mapping[quality]

        chord_str = get_chord_str(root, quality, quality_list)
        if chord_str == pre_chord:
            continue
        if idx * sec_per_frame - pre_time < min_duration:
            pre_root, pre_quality, pre_chord = root, quality, chord_str
            continue
        if verbose:
            print("%.3f %.3f %s" % (pre_time, idx * sec_per_frame, pre_chord))
        predict_list.append((pre_time, idx * sec_per_frame, pre_chord))
        pre_root, pre_quality, pre_chord = root, quality, chord_str
        pre_time = idx * sec_per_frame
    if verbose:
        print("%.3f %.3f %s" % (pre_time, len(estimation.T) * sec_per_frame, pre_chord))
    predict_list.append((pre_time, len(estimation.T) * sec_per_frame, pre_chord))
    return predict_list 


def predict(flac_path, title="", model_path="./model", diff_root_only=True, max_num_chord=4):
    label_path = "chord_labels.txt"

    # Estimate the bpm of the audio
    beat_proc = RNNBeatProcessor()
    tempo_proc = TempoEstimationProcessor(min_bpm=50, max_bpm=180, fps=100)

    beat_processed = beat_proc(flac_path)
    tempo_estimation = tempo_proc(beat_processed)
    
    BPM = BPM_selector(tempo_estimation)
    sec_per_beat = 60 / BPM
    
    sec_per_frame = 2048 / 16000
    # set eighth note as the minimum duration of the chord
    min_duration = sec_per_beat / 2

    # Read chord labels file
    with open(label_path) as f:
        with torch.no_grad():
            chord_labels = ast.literal_eval(f.read())

            # Process raw audio
            X = cqt_preprocess(flac_path)
            X = Variable(torch.from_numpy(
                np.expand_dims(X, axis=0)).float().cpu())

            # Load model
            model = Net(1).cpu()
            state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.eval()

            # Estimate
            estimation = np.zeros((22, X.shape[2]))
            estimation = model(X).data.cpu()[0][0]
            estimation = to_probability(estimation)

            # Post-processing
            estimation = dp_post_processing(estimation)

            # predict_list_majmin = _predict(estimation, chord_labels[13:], sec_per_frame, min_duration, mapping_majmin)
            predict_list_seventh = _predict(estimation, chord_labels[13:], sec_per_frame, min_duration, mapping_seventh)
        
        text = ''
        for chord in predict_list_seventh:
            text += f'{chord[0]}\t{chord[1]}\t{chord[2]}\n'

        return text

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python predict <audio_path> <target_path> <model_path>")

    audio_path = sys.argv[1]
    target_path = sys.argv[2]
    model_path = sys.argv[3]

    result = predict(audio_path, model_path=model_path)
    with open(os.path.join(target_path, os.path.splitext(os.path.basename(audio_path))[0] + ".txt"), "w") as f:
        f.write(result)
