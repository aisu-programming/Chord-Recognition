''' Libraries '''
import pandas as pd

from mapping import my_mapping_dict


''' Global variables '''
debug_mode = False
if debug_mode:
    data_directory = 'CE200_sample'
    file_amount = 20
else:
    data_directory = 'CE200'
    file_amount = 200

mapping_dictionary = my_mapping_dict


''' Codes '''
def answer_label_static(file_index):

    data = pd.read_csv(f'{data_directory}/{file_index+1}/data.csv', index_col=0)
    data = data.values

    label_amount = { k: 0 for k in mapping_dictionary.keys() }
    for row in data:
        label = row[24]
        changed = False
        for chord in ['maj6', 'maj9', 'maj11', 'maj13', 'min6', 'min9', 'min11', 'min13']:
            if chord in label:
                label = 'Other'
                break
        for chord in mapping_dictionary.keys():
            if chord in label:
                label = chord
                changed = True
                break
        if not changed: label = 'Other'
        label_amount[label] += 1

    return label_amount


def chord_analyze():

    label_amount = { k: {} for k in mapping_dictionary.keys() }
    
    for file_index in range(200):

        data = pd.read_csv(f'{data_directory}/{file_index+1}/data.csv', index_col=0)
        data = data.values

        for row in data:
            label = row[24]
            changed = False
            for chord in ['maj6', 'maj9', 'maj11', 'maj13', 'min6', 'min9', 'min11', 'min13']:
                if chord in label:
                    label = 'Other'
                    break
            for chord in mapping_dictionary.keys():
                if chord in label:
                    label = chord
                    changed = True
                    break
            if not changed: label = 'Other'
            if label != 'N' and label != 'Other':
                # chroma_cqt
                chord = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                chroma_cqt = { chord[i]: c for i, c in enumerate(row[:12]) }
                chroma_cqt = [ k for k, v in sorted(chroma_cqt.items(), key=lambda item: item[1]) ]
                chroma_cqt = sorted(chroma_cqt[:4])
                compose = f'{chroma_cqt[0]}{chroma_cqt[1]}{chroma_cqt[2]}{chroma_cqt[3]}'
                # print(f'{label}: {compose}')
                if compose not in label_amount[label]: label_amount[label][compose] = 0
                label_amount[label][compose] += 1

    for label, label_dict in label_amount.items():
        label_amount[label] = { k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1], reverse=True) }

    print(label_amount)
    return label_amount


def view_ground_truth(file_index):

    with open(f'{data_directory}/{file_index+1}/ground_truth.txt') as f:

        index = 0
        while True:

            index += 1
            row = f.readline()
            if row == '': break
            data_array = row[:-1].split('\t')

            start_time = float(data_array[0])
            end_time = float(data_array[1])
            during_time = end_time - start_time
            chord = str(data_array[2])

            print('{:3} : {:10.6f} ~ {:10.6f} ({:9.6f}) : {}'.format(index, start_time, end_time, during_time, chord))


def min_interval_analyze():

    min_interval = 10.0

    for file_index in range(200):

        with open(f'{data_directory}/{file_index+1}/ground_truth.txt') as f:

            index = 0
            while True:

                index += 1
                row = f.readline()
                if row == '': break
                data_array = row[:-1].split('\t')

                start_time = float(data_array[0])
                end_time = float(data_array[1])
                interval = end_time - start_time
                chord = str(data_array[2])

                if interval > 0.02322 and interval < min_interval:
                    min_interval = interval
                    file_no = file_index+1
                    row_no = index
                    label = chord

    print(min_interval, file_no, row_no, label)

''' Testing '''
if __name__ == "__main__":
    # view_ground_truth(file_index=1)
    # for i in range(5):
    #     total = answer_label_static(file_index=i)
    #     total = dict(filter(lambda x: x[1] > 0, total.items()))
    #     print(f'{i:3d}: {total}')
    
    # chord_analyze()

    min_interval_analyze()