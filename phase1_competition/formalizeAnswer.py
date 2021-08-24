import json

formal = False

directory = 'CE500_test'
amount = 300

data = {}

with open('answer.json', mode='w') as output_file:
    for i in range(amount):
        with open(f'{directory}/{i+1}/est_file_last.txt', mode='r') as est_file:
            data[i+1] = []
            file_content = est_file.read()
            file_content = file_content.split('\n')
            for row in file_content:
                if row == '': break
                row_items = row.split('\t')
                start_time = row_items[0]
                end_time = row_items[1]
                chord = row_items[2]
                data[i+1].append([start_time, end_time, chord])
    json.dump(data, output_file)