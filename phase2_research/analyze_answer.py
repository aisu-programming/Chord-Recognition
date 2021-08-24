''' Libraries '''
import mir_eval


''' Codes '''
def view_ground_truth(song_i):

    qualities = {}
    for i in range(song_i):
        chords = {}
        try:
            with open(f"../customized_data/CE200/{i+1}/ground_truth.txt") as f:
                index = 0
                while True:
                    index += 1
                    row = f.readline()
                    if row == '': break
                    data_array = row[:-1].split('\t')

                    start_time = float(data_array[0])
                    end_time = float(data_array[1])
                    during_time = end_time - start_time
                    chord = str(data_array[2]).strip()

                    if chord not in chords.keys(): chords[chord] = 1
                    else: chords[chord] += 1

                    # quality = mir_eval.chord.split(chord)[1]
                    # if quality not in qualities.keys(): qualities[quality] = 1
                    # else: qualities[quality] += 1

            chords = dict(sorted(chords.items()))
            print(i, chords)

        except:
            print(f"Read file no.{i+1} error.")

    # qualities = dict(sorted(qualities.items()))
    # print(qualities)


''' Testing '''
if __name__ == "__main__":
    view_ground_truth(200)