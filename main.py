import json
import numpy as np
from pychord import Chord, note_to_chord

from visualization import display_figure
from score import get_sevenths_scores_array

# Parameters


def get_significant_chord(frame_array):

    chord_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord = { chord_name[i]: frame_array[i] for i in range(12) }
    sorted_chord = [ k for k, v in sorted(chord.items(), key=lambda item: item[1], reverse=True) ]

    answer = note_to_chord(sorted_chord[:3])
    if answer == []: answer = note_to_chord(sorted_chord[:4])

    return answer


if __name__ == "__main__":
    
    with open('CE200_sample/1/feature.json') as f:
        data = json.load(f)
    
    chroma_cens = np.array(data['chroma_cens'])
    chroma_cens = chroma_cens.transpose()

    answer_tmp = Chord("")
    for index, frame_array in enumerate(chroma_cens):
        answer = get_significant_chord(frame_array)
        if answer != [] and answer[0] != answer_tmp:
            print('{:4}: {}'.format(index, answer[0]))
            answer_tmp = answer[0]