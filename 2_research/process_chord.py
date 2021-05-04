''' Libraries '''
import numpy as np
import mir_eval

from my_mapping import mapping_quality2id, mapping_chord2id


''' Functions '''
def process_chords_13_6(chords):
    qualities = [ mir_eval.chord.split(chord)[1] for chord in chords ]
    for i, quality in enumerate(qualities):
        if quality in mapping_quality2id.keys(): qualities[i] = mapping_quality2id[quality]
        else: qualities[i] = mapping_quality2id['']
    quality_ids = np.eye(len(mapping_quality2id))[qualities].tolist()
    root_ids, quality_bitmaps, _ = mir_eval.chord.encode_many(chords)
    root_ids = [ id+1 for id in root_ids ]
    root_ids = np.eye(13)[root_ids].tolist()
    return np.concatenate([root_ids, quality_ids], axis=-1).tolist()


def process_chords_63(chords):
    root_qualities = [
        (mir_eval.chord.split(chord)[0], mir_eval.chord.split(chord)[1])
        for chord in chords
    ]
    chord_ids = []
    for (root, quality) in root_qualities:
        chord = 'N' if root == 'N' else f"{root}:{quality}"
        if chord in mapping_chord2id.keys():
            chord_ids.append(mapping_chord2id[chord])
        else:
            chord_ids.append(mapping_chord2id['X'])
    return np.eye(63)[chord_ids].tolist()