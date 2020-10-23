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
DEBUG_MODE = True


'''
Global variables
'''
if DEBUG_MODE:
    file_directory = 'CE200_sample'
    file_amount = 20
else:
    file_directory = 'CE200'
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
    sys.stdout.write("Reading feature.json of each songs in '%s'.\n[%s]" % (file_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):
        
        file_name = file_directory + f'/{song_index+1}/feature.json'
        with open(file_name) as f:
            data = json.load(f)

        chroma_cqt = np.array(data['chroma_cqt'])
        chroma_cens = np.array(data['chroma_cens'])
        input_data = np.vstack((chroma_cqt, chroma_cens)).transpose()
        all_input_data.append(input_data)

        if DEBUG_MODE:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")
    return all_input_data


def process_answer_data():

    all_answer_data = []

    toolbar_width = 100
    sys.stdout.write("Processing ground_truth.txt of each songs in '%s'.\n[%s]" % (file_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):

        answer_data = []

        file_name = file_directory + f'/{song_index+1}/ground_truth.txt'
        with open(file_name) as f:
            while True:
                row = f.readline()
                if row == '': break
                row_data = row[:-1].split('\t')
                second_per_frame = float(HOP_LENGTH) / float(SR)
                start_time = float(row_data[0])
                end_time = float(row_data[1])
                label = row_data[2]
                for _ in range(round((end_time - start_time) / second_per_frame)):
                    answer_data.append(label)

        all_answer_data.append(answer_data)
        if DEBUG_MODE:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")
    return all_answer_data

if __name__ == "__main__":
    all_input_data = read_input_data()
    all_answer_data = process_answer_data()
    
    print('index:  input <---> answer')
    print('-------------------------')
    for index in range(file_amount):
        print(f'{index+1:5}: {len(all_input_data[index]):6} <---> {len(all_answer_data[index]):6}')

    # all_input_data[10]
    # tmp = ''
    # for chord in all_answer_data[10]:
    #     if tmp != chord:
    #         print(chord)
    #         tmp = chord