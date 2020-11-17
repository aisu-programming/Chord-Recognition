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
        for chord in mapping_dictionary.keys():
            if chord in label:
                label = chord
                changed = True
                break
        if not changed: label = 'Other'
        label_amount[label] += 1

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


''' Testing '''
if __name__ == "__main__":
    # view_ground_truth(file_index=1)
    for i in range(200):
        total = answer_label_static(file_index=i)
        total = list(filter(lambda x: x[1] > 0, total.items()))
        print(total)