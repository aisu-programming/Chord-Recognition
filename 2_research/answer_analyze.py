''' Libraries '''
# pass


''' Codes '''
def view_ground_truth(song_i):

    with open(f"../customized_data/CE10/{song_i+1}/ground_truth.txt") as f:

        index = 0
        min_during_time = 1000.
        while True:

            index += 1
            row = f.readline()
            if row == '': break
            data_array = row[:-1].split('\t')

            start_time = float(data_array[0])
            end_time = float(data_array[1])
            during_time = end_time - start_time
            chord = str(data_array[2])

            # print('{:3} : {:10.6f} ~ {:10.6f} ({:9.6f}) : {}'.format(index, start_time, end_time, during_time, chord))

            if during_time < min_during_time: min_during_time = during_time
        
        print(min_during_time)


''' Testing '''
if __name__ == "__main__":
    for i in range(10):
        view_ground_truth(song_i=i)