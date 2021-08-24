''' Libraries '''
import pandas as pd
import numpy as np
import mir_eval
from tqdm import tqdm


''' Parameters '''
SR = 44800


''' Functions '''
def find_seventh_chord(label):
    root = mir_eval.chord.split(label)[0]
    criteion_func = mir_eval.chord.sevenths
    # criteion_func = mir_eval.chord.sevenths_inv
    try:
        if label == 'N': return 'N'
        seventh_target = [ ":min", ":maj", ":7", ":maj7", ":min7" ]
        for target in seventh_target:
            if criteion_func([f"{root}{target}"], [label]): return f"{root}{target}"
            else: return f"{label}"
    except mir_eval.chord.InvalidChordException:
        print(f"{label} <---> {root}{target}")
        raise mir_eval.chord.InvalidChordException
    except:
        print(f"{label} <---> {root}{target}")
        raise Exception



def main(path, song_amount, hop_len):

    for i in tqdm(range(song_amount), total=song_amount, ascii=True):

        if i+1 in [168, 187, 190]: continue

        output = pd.read_csv(f"{path}/{i+1}/feature_{SR}_{hop_len}.csv", index_col=0).values.tolist()

        y_filename = f"{path}/{i+1}/ground_truth.txt"
        y_reference = []

        with open(y_filename) as f:
            while True:
                row = f.readline()
                if row == '': break
                row_data   = row[:-1].split('\t')
                start_time = float(row_data[1])
                end_time   = float(row_data[1])
                label      = row_data[2].strip()
                # seventh    = find_seventh_chord(label)
                y_reference.append({
                    'start_time': start_time,
                    'end_time'  : end_time,
                    'label'     : label,
                    # 'seventh'   : seventh,
                })

        # length of feature & last end_time might be different
        x_time = len(output) * hop_len / SR
        y_time = y_reference[-1]['end_time']

        y_j = 0
        for k, _ in enumerate(output):
            # end_time = (k+1) * hop_len / SR + ((y_time-x_time) / 2)
            end_time = (k+1) * hop_len / SR
            while True:
                if y_j == len(y_reference): break
                elif end_time > y_reference[y_j]['end_time']: y_j += 1
                else: break
                
            if end_time < 0:
                output[k].append('N')
                # output[k].append('N')
            elif y_j < len(y_reference):
                output[k].append(y_reference[y_j]['label'])
                # output[k].append(y_reference[y_j]['seventh'])
            else:
                output[k].append('N')
                # output[k].append('N')

        pd.DataFrame(np.array(output)).to_csv(f"{path}/{i+1}/data_{SR}_{hop_len}.csv")

    return


''' Exection '''
if __name__ == '__main__':
    main(r"../customized_data/CE200", 200, 4480)
    # main(r"../customized_data/CE200", 200, 1024)
    # main(r"../customized_data/CE200", 200, 512)