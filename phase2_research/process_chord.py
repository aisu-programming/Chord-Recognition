''' Libraries '''
import numpy as np
import mir_eval

from my_mapping import (
    mapping_root2id,
    mapping_majmin_quality2id,  mapping_majmin_chord2id,
    mapping_seventh_quality2id, mapping_seventh_chord2id,
)


''' Functions '''
def process_answers(chords, model_target, pred_mode):

    if pred_mode == 'root':
        roots    = [ mir_eval.chord.split(chord)[0] for chord in chords ]
        root_ids = []
        for root in roots: root_ids.append(mapping_root2id[root])
        answers = np.eye(13)[root_ids].tolist()

    elif pred_mode == 'quality_bitmap':
        # root_ids, quality_bitmaps, bass_ids = mir_eval.chord.encode_many(targets)
        root_ids, quality_bitmaps, _ = mir_eval.chord.encode_many(chords)
        # quality_bitmaps = mir_eval.chord.rotate_bitmap_to_root(quality_bitmaps, root_ids)
        answers = quality_bitmaps

    else:

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
                chord = 'N' if root == 'N' else f"{root}:{quality}"
                if chord in mapping_chord2id.keys():
                    chord_ids.append(mapping_chord2id[chord])
                else:
                    chord_ids.append(mapping_chord2id['X'])
            answers = np.eye(len(mapping_chord2id))[chord_ids].tolist()

        elif pred_mode == 'quality':
            qual_ids = []
            for (_, quality) in root_qualities:
                if root == 'N':
                    qual_ids.append(mapping_quality2id['N'])
                elif quality in mapping_quality2id.keys():
                    qual_ids.append(mapping_quality2id[quality])
                else:
                    qual_ids.append(mapping_quality2id['X'])
            answers = np.eye(len(mapping_quality2id))[qual_ids].tolist()

        elif pred_mode == 'separate':
            root_ids = []
            qual_ids = []
            for (root, quality) in root_qualities:
                root_ids.append(mapping_root2id[root])
                if root == 'N':
                    qual_ids.append(mapping_quality2id['N'])
                elif quality in mapping_quality2id.keys():
                    qual_ids.append(mapping_quality2id[quality])
                else:
                    qual_ids.append(mapping_quality2id['X'])
            root_onehots = np.eye(13)[root_ids].tolist()
            # root_onehots = np.eye(len(mapping_root2id.values()))[root_ids].tolist()
            quality_onehots = np.eye(len(mapping_quality2id))[qual_ids].tolist()
            answers = np.concatenate([root_onehots, quality_onehots], axis=-1).tolist()

    return answers