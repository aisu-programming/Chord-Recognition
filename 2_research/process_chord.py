''' Libraries '''
import numpy as np
import mir_eval

from my_mapping import (
    mapping_root2id,
    mapping_majmin_quality2id,  mapping_majmin_chord2id,
    mapping_seventh_quality2id, mapping_seventh_chord2id,
)


''' Functions '''
def process_chords(chords, model_target, pred_mode):

    root_qualities = [
        (mir_eval.chord.split(chord)[0], mir_eval.chord.split(chord)[1])
        for chord in chords
    ]

    if model_target == 'majmin':
        mapping_chord2id   = mapping_majmin_chord2id
        mapping_quality2id = mapping_majmin_quality2id
    elif model_target == 'seventh':
        mapping_chord2id   = mapping_seventh_chord2id
        mapping_quality2id = mapping_seventh_quality2id

    if pred_mode == 'integrate':
        chord_ids = []
        for (root, quality) in root_qualities:
            if quality == 'dim': quality = 'min'
            chord = 'N' if root == 'N' else f"{root}:{quality}"
            if chord in mapping_chord2id.keys():
                chord_ids.append(mapping_chord2id[chord])
            else:
                chord_ids.append(mapping_chord2id['X'])
        onehots = np.eye(len(mapping_chord2id))[chord_ids].tolist()
    elif pred_mode == 'separate':
        root_ids    = []
        quality_ids = []
        for (root, quality) in root_qualities:
            if quality == 'dim': quality = 'min'
            root_ids.append(mapping_root2id[root])
            if root == 'N':
                quality_ids.append(mapping_quality2id['N'])
            elif quality in mapping_quality2id.keys():
                quality_ids.append(mapping_quality2id[quality])
            else:
                quality_ids.append(mapping_quality2id['X'])
        root_onehots = np.eye(len(mapping_root2id))[root_ids].tolist()
        quality_onehots = np.eye(len(mapping_quality2id))[quality_ids].tolist()
        onehots = np.concatenate([root_onehots, quality_onehots], axis=-1).tolist()

    return onehots