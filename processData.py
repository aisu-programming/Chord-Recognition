''' Libraries '''
import sys
import json
import numpy as np
import pandas as pd


''' Parameters '''
SR = 22050
N_FFT = 512
HOP_LENGTH = 512
FRAME_LENGTH = 512


''' Global variables '''
debug_mode = True
if debug_mode:
    output_file_name = 'CE200_sample.csv'
    file_directory = 'CE200_sample'
    file_amount = 20
else:
    output_file_name = 'CE200.csv'
    file_directory = 'CE200'
    file_amount = 200


''' Codes '''
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

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
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
        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
    return all_answer_data


if __name__ == "__main__":

    if debug_mode: print('\nDEBUG MODE\n')
    else: print('\nNORMAL MODE\n')
    
    all_input_data = read_input_data()
    all_answer_data = process_answer_data()
    
    # print('index:  input <---> answer')
    # print('--------------------------')
    for index in range(file_amount):
        # 刪除多餘的辨識答案
        while len(all_answer_data[index]) > len(all_input_data[index]):
            all_answer_data[index].pop()
        # 填補欠缺的辨識答案
        while len(all_answer_data[index]) < len(all_input_data[index]):
            all_answer_data[index].append('N')

    toolbar_width = 100
    sys.stdout.write("Combining inputs and answers.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    all_data = np.empty((0, 27))
    for index in range(file_amount):

        frame_length = len(all_input_data[index])
        song_index_array = np.array([[index+1] * frame_length]).transpose()
        frame_index_array = np.array([range(1, frame_length+1)]).transpose()
        input_data = np.array(all_input_data[index])
        answer_data = np.array([all_answer_data[index]]).transpose()

        combined_data = np.hstack((song_index_array, frame_index_array, input_data, answer_data))
        all_data = np.vstack((all_data, combined_data))

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    sys.stdout.write("]\n\n")
    print('Convert into DataFrame... ', end='')
    data = pd.DataFrame(
        all_data,
        columns=[
            'Song No.',
            'Frame No.',
            'chroma_cqt C',
            'chroma_cqt C#',
            'chroma_cqt D',
            'chroma_cqt D#',
            'chroma_cqt E',
            'chroma_cqt F',
            'chroma_cqt F#',
            'chroma_cqt G',
            'chroma_cqt G#',
            'chroma_cqt A',
            'chroma_cqt A#',
            'chroma_cqt B',
            'chroma_cens C',
            'chroma_cens C#',
            'chroma_cens D',
            'chroma_cens D#',
            'chroma_cens E',
            'chroma_cens F',
            'chroma_cens F#',
            'chroma_cens G',
            'chroma_cens G#',
            'chroma_cens A',
            'chroma_cens A#',
            'chroma_cens B',
            'label'
        ]
    )
    print(f"Done.\n\nWriting data into '{output_file_name}'... ", end='')
    data.to_csv(output_file_name)
    print('Done.')