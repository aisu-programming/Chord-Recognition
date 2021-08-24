''' Libraries '''
import os
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
data_divide_amount = 1 # For original data, set as 1

formal_mode = True
if formal_mode:
    file_directory = 'CE500_test'
    file_amount = 300
else:
    file_directory = 'CE200'
    file_amount = 200


''' Codes '''
def read_input_data(song_index, data_divide_amount):
        
    file_name = f'{file_directory}/{song_index+1}/feature.json'
    with open(file_name) as f:
        data = json.load(f)

    input_data = np.array(data['chroma_cqt'] + data['chroma_cens']).transpose()

    if data_divide_amount > 1:
        divided_input_data = []
        for data_index in range(len(input_data)):
            divided_input_data.append(input_data[data_index])
            if data_index == len(input_data) - 1: break
            for divide_index in range(1, data_divide_amount):
                divided_input_data.append(
                    input_data[data_index] * ((data_divide_amount - divide_index) / data_divide_amount) +
                    input_data[data_index + 1] * (divide_index / data_divide_amount)
                )
        return divided_input_data

    else:
        return input_data


def process_answer_data(song_index, input_data, data_divide_amount):

    answer_reference = []
    file_name = f'{file_directory}/{song_index+1}/ground_truth.txt'
    with open(file_name) as f:
        while True:
            row = f.readline()
            if row == '': break
            row_data = row[:-1].split('\t')
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

    return answer_data


def match_data_and_write_into_csv():

    cqt = 'chroma_cqt '
    cen = 'chroma_cens '
    columns = [
        cqt+'C', cqt+'C#', cqt+'D', cqt+'D#', cqt+'E', cqt+'F', cqt+'F#', cqt+'G', cqt+'G#', cqt+'A', cqt+'A#', cqt+'B',
        cen+'C', cen+'C#', cen+'D', cen+'D#', cen+'E', cen+'F', cen+'F#', cen+'G', cen+'G#', cen+'A', cen+'A#', cen+'B',
    ]
    if not formal_mode: columns.append('label')

    toolbar_width = 100
    print(f"Combining inputs and answers in '{file_directory}'.")
    if data_divide_amount > 1: print(f"Each frame will be divide into {data_divide_amount} parts.")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):

        input_data = read_input_data(song_index, data_divide_amount)
        if not formal_mode: answer_data = process_answer_data(song_index, input_data, data_divide_amount)

        combined_data = [ list(data) for data in input_data ]
        if not formal_mode:
            for i in range(len(combined_data)):
                combined_data[i].append(answer_data[i])

        combined_data = np.array(combined_data)
        combined_data = pd.DataFrame(combined_data, columns=columns)
        if data_divide_amount == 1:
            csv_file_path = f'{file_directory}/{song_index+1}/data.csv'
        else:
            csv_file_path = f'{file_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
        combined_data.to_csv(csv_file_path)

        if (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
        
    sys.stdout.write("]\n\n")
    print('Done.')


def generate_mapping_dictionary():

    file_directory = 'CE200'
    file_amount = 200

    all_label = []

    toolbar_width = 100
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    for song_index in range(file_amount):
        file_name = f'{file_directory}/{song_index+1}/ground_truth.txt'
        with open(file_name) as f:
            while True:
                row = f.readline()
                if row == '': break
                label = row[:-1].split('\t')[2].strip()
                if label not in all_label:
                    all_label.append(label)

        if (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    sys.stdout.write("]\n\n")

    all_label.sort()
    all_label.remove('N')
    all_label.insert(0, 'N')
    mapping_dict = { label: index for index, label in enumerate(all_label) }
    print(mapping_dict)
    return


if __name__ == "__main__":

    os.system('cls')

    if formal_mode: print('\nFORMAL DATA MODE\n')
    else: print('\nTRAIN DATA MODE\n')

    match_data_and_write_into_csv()

    # generate_mapping_dictionary()