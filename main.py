'''
Libraries
'''
import sys
import json
import numpy as np
# from pychord import Chord, note_to_chord

# from visualization import display_figure
from score import get_sevenths_scores_array


'''
Parameters
'''
SR = 22050
N_FFT = 512
HOP_LENGTH = 512
FRAME_LENGTH = 512


'''
Global variables
'''
file_amount = 200


'''
Codes
'''
# def find_chord_by_pychord(frame_array):

#     chord_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     chord = { chord_name[i]: frame_array[i] for i in range(12) }
#     sorted_chord = [ k for k, v in sorted(chord.items(), key=lambda item: item[1], reverse=True) ]
#     answer = note_to_chord(sorted_chord[:3])
#     if answer == []: answer = note_to_chord(sorted_chord[:4])

#     # return answer
#     return None


def read_input_data():

    all_input_data = []

    toolbar_width = 100
    sys.stdout.write("Reading feature.json of each songs in 'CE200'.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):
        
        file_name = f'CE200/{song_index+1}/feature.json'
        with open(file_name) as f:
            data = json.load(f)

        chroma_cqt = np.array(data['chroma_cqt'])
        chroma_cens = np.array(data['chroma_cens'])
        input_data = np.vstack((chroma_cqt, chroma_cens)).transpose()
        all_input_data.append(input_data)

        if (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")
    return all_input_data


def produce_answer_data():

    all_answer_data = []

    toolbar_width = 100
    sys.stdout.write("Reading ground_truth.txt of each songs in 'CE200'.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):

        answer_data = []

        file_name = f'CE200/{song_index+1}/ground_truth.txt'
        with open(file_name) as f:
            while True:
                row = f.readline()
                if row == '': break
                row_data = row[:-1].split('\t')
                answer_data.append([row_data[0], row_data[1], row_data[2]])

        all_answer_data.append(answer_data)
        if (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")
    return all_answer_data


if __name__ == "__main__":

    # all_input_data = read_input_data()
    all_answer_data = produce_answer_data()
    