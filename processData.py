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
debug_mode = False
if debug_mode:
    output_file_name = 'CE200_sample.csv'
    file_directory = 'CE200_sample'
    file_amount = 20
else:
    output_file_name = 'CE200.csv'
    file_directory = 'CE200'
    file_amount = 200

data_divide_amount = 15 # For original data, set as 1

''' Codes '''
def read_input_data():

    all_input_data = []

    toolbar_width = 100
    sys.stdout.write("Reading feature.json of each songs in '%s'.\n[%s]" % (file_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):
        
        file_name = f'{file_directory}/{song_index+1}/feature.json'
        with open(file_name) as f:
            data = json.load(f)

        chroma_cqt = np.array(data['chroma_cqt'])
        chroma_cens = np.array(data['chroma_cens'])
        input_data = np.vstack((chroma_cqt, chroma_cens)).transpose()

        divided_input_data = []
        if data_divide_amount > 1:
            for data_index in range(len(input_data)):
                divided_input_data.append(input_data[data_index])
                if data_index == len(input_data) - 1: break
                for divide_index in range(1, data_divide_amount):
                    divided_input_data.append(
                        input_data[data_index] * ((data_divide_amount - divide_index) / data_divide_amount) +
                        input_data[data_index + 1] * (divide_index / data_divide_amount)
                    )

        all_input_data.append(divided_input_data)

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
    return all_input_data


def process_answer_data(all_input_data):

    all_answer_data = []

    toolbar_width = 100
    sys.stdout.write("Processing ground_truth.txt of each songs in '%s'.\n[%s]" % (file_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index, input_data in enumerate(all_input_data):

        answer_reference = []
        file_name = f'{file_directory}/{song_index+1}/ground_truth.txt'
        with open(file_name) as f:
            while True:
                row = f.readline()
                if row == '': break
                row_data = row[:-1].split('\t')
                # start_time = float(row_data[0])
                end_time = float(row_data[1])
                label = row_data[2]
                answer_reference.append({
                    'end_time': end_time,
                    'label': label
                })
        
        second_per_frame = (float(HOP_LENGTH) / float(SR)) / float(data_divide_amount)
        answer_data = []
        answer_reference_index = 0
        for frame_index in range(len(input_data)):
            if answer_reference_index == len(answer_reference):
                answer_data.append('N')
                continue
            end_time = second_per_frame * (frame_index + 1)
            if end_time >= answer_reference[answer_reference_index]['end_time']:
                answer_reference_index += 1
                if answer_reference_index == len(answer_reference):
                    answer_data.append('N')
                    continue
            answer_data.append(answer_reference[answer_reference_index]['label'])
        all_answer_data.append(answer_data)

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
    return all_answer_data


def match_data_and_write_into_csv(all_in_one=False):

    if all_in_one: print('All in one: ON\n')
    else: print('All in one: OFF\n')

    all_input_data = read_input_data()
    all_answer_data = process_answer_data(all_input_data)

    # for index in range(file_amount):
    #     # 刪除多餘的辨識答案
    #     while len(all_answer_data[index]) > len(all_input_data[index]):
    #         all_answer_data[index].pop()
    #     # 填補欠缺的辨識答案
    #     while len(all_answer_data[index]) < len(all_input_data[index]):
    #         all_answer_data[index].append('N')

    cqt = 'chroma_cqt '
    cen = 'chroma_cens '
    columns = [
        'Song No.', 'Frame No.',
        cqt+'C', cqt+'C#', cqt+'D', cqt+'D#', cqt+'E', cqt+'F', cqt+'F#', cqt+'G', cqt+'G#', cqt+'A', cqt+'A#', cqt+'B',
        cen+'C', cen+'C#', cen+'D', cen+'D#', cen+'E', cen+'F', cen+'F#', cen+'G', cen+'G#', cen+'A', cen+'A#', cen+'B',
        'label'
    ]

    toolbar_width = 100
    if all_in_one: sys.stdout.write("Combining all inputs and answers together.\n[%s]" % (" " * toolbar_width))
    else: sys.stdout.write("Combining inputs and answers in each files.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    all_data = np.empty((0, 27))
    for song_index in range(file_amount):

        frame_length = len(all_input_data[song_index])
        song_index_array = np.array([[song_index+1] * frame_length]).transpose()
        frame_index_array = np.array([range(1, frame_length+1)]).transpose()
        input_data = np.array(all_input_data[song_index])
        answer_data = np.array([all_answer_data[song_index]]).transpose()
        combined_data = np.hstack((song_index_array, frame_index_array, input_data, answer_data))

        if all_in_one: all_data = np.vstack((all_data, combined_data))
        else:
            data = pd.DataFrame(combined_data, columns=columns)
            if data_divide_amount == 1:
                csv_file_path = f'{file_directory}/{song_index+1}/data.csv'
            else:
                csv_file_path = f'{file_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
            data.to_csv(csv_file_path)

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    sys.stdout.write("]\n\n")

    if all_in_one:
        print('Convert into DataFrame... ', end='')
        data = pd.DataFrame(all_data, columns=columns)
        print(f"Done.\n\nWriting data into '{output_file_name}'... ", end='')
        data.to_csv(output_file_name)
    print('Done.')


def generate_mapping_dictionary():
    all_input_data = read_input_data()
    all_answer_data = process_answer_data(all_input_data)
    all_label = []
    for answer_data in all_answer_data:
        for label in answer_data:
            if label.strip() not in all_label:
                all_label.append(label.strip())
    all_label.sort()
    all_label.remove('N')
    all_label.insert(0, 'N')
    mapping_dict = { label: index for index, label in enumerate(all_label) }
    return mapping_dict


if __name__ == "__main__":

    if debug_mode: print('\nDEBUG MODE\n')
    else: print('\nNORMAL MODE\n')

    match_data_and_write_into_csv(all_in_one=False)

    # print(generate_mapping_dictionary())