# usage: python convert.py <source_path> <target_path> <components/shorthand>

import sys, os
import re
import numpy as np
import mir_eval
from mir_eval.chord import QUALITIES
from mir_eval.chord import EXTENDED_QUALITY_REDUX as EXTENDED

SCALE_DEGREES_INV = dict([(s, d) for d, s in mir_eval.chord.SCALE_DEGREES.items()])

def find_components(chords):
    output_chords = []
    for chord in chords:
        root, quality, extra, bass = mir_eval.chord.split(chord)
        quality, extend = mir_eval.chord.reduce_extended_quality(quality)
        quality_bmap = mir_eval.chord.quality_to_bitmap(quality)
 
        tmp = [i for i, x in enumerate(quality_bmap) if x == 1]
        components = set()
        for x in tmp:
            if x in SCALE_DEGREES_INV.keys():
                components.add(SCALE_DEGREES_INV[x])
            else:
                components.add('b' + SCALE_DEGREES_INV[x + 1])
        if quality == "dim7":
            components.remove("6")
            components.add("bb7")
        
        components = components | extend
        for component in extra - components:
            if component[0] in "*b#" and component[1:] in components:
                components.remove(component[1:])
                if component[0] in "b#":
                    components.add(component)
            else:
                components.add(component)

        components = sorted(list(components), key=lambda x: int(re.search(r"\d+", x).group()))
        output_chords.append(mir_eval.chord.join(root, "", components, bass))
    return(np.array(output_chords))

def find_quality(bitmap, extra):
    min_mismatch = 24
    best_quality = None
    isExtend = False
    for quality in QUALITIES.keys():
        num_mismatch = np.count_nonzero(bitmap != QUALITIES[quality]) + len(extra)
        if num_mismatch < min_mismatch:
            min_mismatch = num_mismatch
            best_quality = quality

    for extended_quality in EXTENDED.keys():
        quality, extension = EXTENDED[extended_quality]
        num_mismatch = np.count_nonzero(bitmap != QUALITIES[quality]) + len(extension ^ extra)
        if num_mismatch < min_mismatch and extended_quality[0] != 'b' and extended_quality[0] != '#':
            min_mismatch = num_mismatch
            best_quality = extended_quality
            isExtend = True

    if not isExtend:
        bitmap -= np.array(QUALITIES[best_quality])
        new_extra = extra
    else:
        bitmap -= np.array(QUALITIES[EXTENDED[best_quality][0]])
        new_extra = extra - EXTENDED[best_quality][1]
        for degree in EXTENDED[best_quality][1] - extra:
            new_extra.add('*' + str(degree))

    for idx in range(bitmap.shape[0]):
        if bitmap[idx] == 0:
            continue

        removed = '*' if bitmap[idx] == -1 else ''
        if idx in SCALE_DEGREES_INV.keys():
            new_extra.add(removed + SCALE_DEGREES_INV[idx])
        elif idx + 1 in SCALE_DEGREES_INV.keys():
            new_extra.add(removed + 'b' + SCALE_DEGREES_INV[idx + 1])

    # clear dummpy degrees
    for degree in range(1, 14): # two octaves
        degree = str(degree)
        if degree in new_extra or '*' + degree in new_extra or 'b' + degree in new_extra or '#' + degree in new_extra:
            if degree in new_extra and '*' + degree in new_extra:
                new_extra.remove(degree)
                new_extra.remove('*' + degree)
            if degree in new_extra and ('b' + degree in new_extra or '#' + degree in new_extra):
                new_extra.remove(degree)
            if '*' + degree in new_extra and ('b' + degree in new_extra or '#' + degree in new_extra):
                new_extra.remove('*' + degree)

    return best_quality, new_extra

def find_shorthand(chords):
    output_chords = []
    for chord in chords:
        root, quality, extra, bass = mir_eval.chord.split(chord)
        if quality or chord == mir_eval.chord.NO_CHORD:
            output_chords.append(chord)
            continue
            
        bitmap = np.zeros(12)
        extra_set = set()
        for note in extra:
            if int(re.search(r"\d+", note).group()) > 7:
                extra_set.add(note)
            else:
                bitmap += mir_eval.chord.scale_degree_to_bitmap(note)
        quality, extra = find_quality(bitmap, extra_set)
        output_chord = mir_eval.chord.join(root, quality, extra, bass)
        if '(' in output_chord:
            tmp = sorted(output_chord.split('(')[1].split(')')[0].split(','), key=lambda x: int(re.search(r"\d+", x).group()))
            output_chord = output_chord.split('(')[0]
            output_chord += '('
            for degree in tmp:
                output_chord += degree + ','
            output_chord = output_chord[:-1] + ')' if bass == "" or bass == '1' else output_chord[:-1] + ')' + '/' + bass
        output_chords.append(output_chord)

    return np.array(output_chords)

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ("components", "shorthand"):
        sys.exit("usage: python convert.py <source_path> <target_path> <components/shorthand>")

    src_pth = sys.argv[1]
    tar_pth = sys.argv[2]
    mode = sys.argv[3]

    with open(src_pth) as src_f:
        labels = np.array([line.split() for line in src_f], dtype="<U64")
    
    chords = labels[:, 2]

    if mode == "components":
        labels[:, 2] = find_components(chords)
    else:
        labels[:, 2] = find_shorthand(chords)

    with open(tar_pth, "w") as tar_f:
        for label in labels:
            tar_f.write(f"{label[0]}\t{label[1]}\t{label[2]}\n")
