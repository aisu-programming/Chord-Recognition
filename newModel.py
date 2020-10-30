''' Libraries '''
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from score import get_sevenths_score
from mapping import my_mapping_dict


''' Parameters '''
RAMDON_SEED = 1

EPOCHS = 1000
BATCH_SIZE = 50000
PATIENCE = 50

''' Global variables'''
debug_mode = False
if debug_mode:
    data_directory = 'CE200_sample'
    file_amount = 20
else:
    data_directory = 'CE200'
    file_amount = 200

data_divide_amount = 1
sec_per_frame = 512.0 / 22050.0 / data_divide_amount

frames_per_data = data_divide_amount * 25
one_side_frames = int((frames_per_data - 1) / 2)

mapping_dictionary = my_mapping_dict

load_exist_model = False
load_model_path = 'model/2020-10-28/20.57.14'

display_figure = False

auto_save_path = True
if auto_save_path:
    now = time.localtime()
    now_date = time.strftime('%Y-%m-%d', now)
    now_time = time.strftime('%H.%M.%S', now)
    save_directory = f'model/{now_date}/{now_time}'
else:
    save_directory = ''

output_answer = True


''' Codes '''
def adjust_model(model):
    model.add(Dense(480, input_shape=(600, ), activation='sigmoid'))
    model.add(Dense(480, input_shape=(480, ), activation='relu'))
    model.add(Dense(len(mapping_dictionary), input_shape=(480, ), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def readAndSplitData(train_file_index):
    
    X = {'train': [], 'validation': []}
    Y = {'train': [], 'validation': []}

    toolbar_width = 100
    sys.stdout.write("\nReading train & validation datas from each songs in '%s'.\n[%s]" % (data_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    filling_row = [0.0] * 24
    for song_index in range(file_amount):

        for _ in range(one_side_frames):
            if song_index in train_file_index:
                X['train'].append(filling_row)
                Y['train'].append(0)
            else:
                X['validation'].append(filling_row)
                Y['validation'].append(0)

        if data_divide_amount == 1: read_csv_file_path = f'{data_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{data_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
        
        data = pd.read_csv(read_csv_file_path, index_col=0)
        data = data.values

        for row in data:
            label = row[24]
            for chord in mapping_dictionary.keys():
                if chord in label:
                    label = int(mapping_dictionary[chord])
                    break
            if label == row[24]: label = 0

            if song_index in train_file_index:
                X['train'].append(row[:24])
                Y['train'].append(label)
            else:
                X['validation'].append(row[:24])
                Y['validation'].append(label)

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    for _ in range(one_side_frames):
        X['train'].append(filling_row)
        Y['train'].append(0)
        X['validation'].append(filling_row)
        Y['validation'].append(0)

    sys.stdout.write("]\n\n")

    X['train'] = np.asarray(X['train']).astype(np.float32)
    Y['train'] = to_categorical(Y['train'], num_classes=len(mapping_dictionary))
    X['validation'] = np.asarray(X['validation']).astype(np.float32)
    Y['validation'] = to_categorical(Y['validation'], num_classes=len(mapping_dictionary))
    
    print(f"Train data: {len(X['train']):7d} / All data: {len(X['validation']) + len(X['train']):7d}\n")
    time.sleep(3)

    return X, Y


def compare_figure(history):
    
    import matplotlib.pyplot as plt
    
    loss = history['loss']
    accuracy = history['accuracy']
    validation_loss = history['val_loss']
    validation_accuracy = history['val_accuracy']
    epochs_length = range(1, len(loss)+1)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(12, 16) # 3:4
    fig.suptitle('Training & Validation Comparition')
    # plt.title('Training & Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    axs[0].plot(epochs_length, loss, "b-", label='Training Loss')
    axs[0].plot(epochs_length, validation_loss, "r-", label='Validation Loss')
    axs[1].plot(epochs_length, accuracy, "b-", label='Training Accuracy')
    axs[1].plot(epochs_length, validation_accuracy, "r-", label='Validation Accuracy')
    axs[0].legend()
    axs[1].legend()

    plt.savefig(f'{save_directory}/figure.png', dpi=200)
    if display_figure: plt.show()

    return


def estimate_and_write_to_file(model):

    toolbar_width = 100
    sys.stdout.write(f"\nEstimating and write to '{data_directory}/xxx/est_file.txt'.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):

        if data_divide_amount == 1: read_csv_file_path = f'{data_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{data_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'

        data = pd.read_csv(read_csv_file_path, index_col=0)
        # 這邊用不到 label，就不處理了
        # data['label'] = data['label'].map(mapping_dict)
        data = data.values

        label_index = data.shape[1]
        original_data_length = len(data)
        data = np.vstack((np.zeros((one_side_frames, label_index)), data, np.zeros((one_side_frames, label_index))))

        X = []
        for i in range(original_data_length):
            X.append(data[i:i+frames_per_data, 0:label_index-1].reshape((label_index-1)*frames_per_data))
            
        X = np.array(X).astype(np.float32)
        Y_pred = model.predict_classes(X)
        with open(f'{data_directory}/{song_index+1}/est_file.txt', mode='w') as f:
            index_now = 0
            index_last = 0
            while index_now < len(Y_pred):
                if (index_now == len(Y_pred) - 1) or (Y_pred[index_now] != Y_pred[index_now+1]):
                    for k, v in mapping_dictionary.items():
                        if v == Y_pred[index_now]:
                            if k == 'Other': f.write(f'{sec_per_frame*index_last:.06f}\t{sec_per_frame*(index_now+1):.06f}\tN\n')
                            else: f.write(f'{sec_per_frame*index_last:.06f}\t{sec_per_frame*(index_now+1):.06f}\t{k}\n')
                            index_last = index_now + 1
                            break
                index_now += 1

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
    return


def record_details(cost_time, model_scores):
    print(f"\nRecording details... ", end='')
    with open(f'{save_directory}/details.txt', mode='w') as f:
        f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
        f.write(f'\n')
        f.write(f'EPOCHS = {EPOCHS}\n')
        f.write(f'BATCH_SIZE = {BATCH_SIZE}\n')
        f.write(f'PATIENCE = {PATIENCE}\n')
        f.write(f'\n')
        f.write(f'data_divide_amount = {data_divide_amount}\n')
        f.write(f'frames_per_data = {frames_per_data}\n')
        f.write(f'\n')
        f.write(f'cost_time: {cost_time}\n')
        f.write(f'\n')
        for model_name, model_score in model_scores.items():
            f.write(f"Score by '{model_name}' model: {model_score}\n")
    print('Done.')
    return


def main():

    np.random.seed(RAMDON_SEED)
    file_index = np.arange(file_amount)
    np.random.shuffle(file_index)
    train_file_index = file_index[:int(file_amount * 0.4)]
    validation_file_index = file_index[int(file_amount * 0.4):]

    ''' Model '''
    model = Sequential()
    if load_exist_model:

        model_dict = {
            'last': 'model.h5',
            'min val_loss': 'best_model_min_val_loss.h5',
            'max val_accuracy': 'best_model_max_val_accuracy.h5',
        }

        for model_name, file_name in model_dict.items():

            model = load_model(f'{load_model_path}/{file_name}')
            # (X, Y), (X_train, Y_train), (X_validation, Y_validation) =reaAndSplitdData(train_file_index)
            # loss, accuracy = model.evaluate(X, Y)
            # print(f'\nEvaluate with original data (all) - Loss: {loss}, Accuracy: {accuracy * 100:.3f}%')
            estimate_and_write_to_file(model)
        
            total_score = 0
            train_score = 0
            validation_score = 0

            for song_index in file_index:
                ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
                est_file = f'{data_directory}/{song_index+1}/est_file.txt'
                score = get_sevenths_score(ref_file_path, est_file)
                total_score += score
                if song_index in train_file_index: train_score += score
                else: validation_score += score

            total_score /= file_amount
            train_score /= len(train_file_index)
            validation_score /= len(validation_file_index)

            print(f"\nTotal average score by '{model_name}': {total_score}")
            print(f"Train average score by '{model_name}': {train_score}")
            print(f"Validation average score by '{model_name}': {validation_score}")

    else:

        X, Y = readAndSplitData(train_file_index)

        model = adjust_model(model)
        os.system('cls')
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        last_model_path = f'{save_directory}/model.h5'
        best_model_min_val_loss_path = f'{save_directory}/best_model_min_val_loss.h5'
        best_model_max_val_accuracy_path = f'{save_directory}/best_model_max_val_accuracy.h5'

        rename_save_directory = save_directory

        start_time = datetime.datetime.now() # Set timer

        train_index_array = np.arange(one_side_frames, len(X['train'])-one_side_frames)
        validation_index_array = np.arange(one_side_frames, len(X['validation'])-one_side_frames)
        my_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        min_val_loss = 1000.0
        max_val_acc  = 0.0

        ''' Train '''
        for epoch in range(EPOCHS):
            np.random.shuffle(train_index_array)
            np.random.shuffle(validation_index_array)
            X_train_tmp = []
            Y_train_tmp = []
            X_validation_tmp = []
            Y_validation_tmp = []
            for i in range(BATCH_SIZE):
                ti = train_index_array[i]
                vi = validation_index_array[i]
                X_train_tmp.append(X['train'][ti-one_side_frames:ti+one_side_frames+1])
                Y_train_tmp.append(Y['train'][ti])
                X_validation_tmp.append(X['validation'][vi-one_side_frames:vi+one_side_frames+1])
                Y_validation_tmp.append(Y['validation'][vi])
            print(f'epoch: {epoch+1}/{EPOCHS} | batch_size: {BATCH_SIZE}')
            history = model.fit(
                np.array(X_train_tmp).reshape(BATCH_SIZE, frames_per_data*24),
                np.array(Y_train_tmp).reshape(BATCH_SIZE, len(mapping_dictionary)),
                validation_data=(
                    np.array(X_validation_tmp).reshape(BATCH_SIZE, frames_per_data*24),
                    np.array(Y_validation_tmp).reshape(BATCH_SIZE, len(mapping_dictionary))
                ),
                # epochs=EPOCHS,
                batch_size=int(BATCH_SIZE/10),
                # callbacks=[MCP_min_val_loss, MCP_max_val_acc, ES_max_val_accuracy]
            )
            print('')

            # ES_max_val_acc
            if (epoch+1) > PATIENCE:
                ES = True
                for i in range(PATIENCE):
                    if history.history['val_accuracy'][0] > my_history['val_accuracy'][-1*(i+1)]:
                        ES = False
                        break
                if ES:
                    print(f"EarlyStopping at epoch {epoch+1}.")
                    break

            for key in my_history.keys(): my_history[key].append(history.history[key][0])

            # MCP_min_val_loss
            if history.history['val_loss'][0] < min_val_loss:
                print(f"Saving new val_loss model. (Improved from {min_val_loss} to {history.history['val_loss'][0]})")
                model.save(best_model_min_val_loss_path)
                min_val_loss = history.history['val_loss'][0]
                min_val_loss_epoch = epoch+1
            print(f"Best min_val_loss model now --> val_loss: {min_val_loss}. (epoch: {min_val_loss_epoch})\n")

            # MCP_max_val_acc
            if history.history['val_accuracy'][0] > max_val_acc:
                print(f"Saving new val_acc  model. (Improved from {max_val_acc} to {history.history['val_accuracy'][0]})")
                model.save(best_model_max_val_accuracy_path)
                max_val_acc = history.history['val_accuracy'][0]
                max_val_acc_epoch = epoch+1
            print(f"Best max_val_acc  model now --> val_acc : {max_val_acc}. (epoch: {max_val_acc_epoch})\n")

        end_time = datetime.datetime.now()
        cost_time = str(end_time-start_time)
        print(f'\nLearning cost time: {cost_time}')
        
        ''' Save figure & model '''
        if data_divide_amount == 1: rename_save_directory = f'{rename_save_directory}-MF'
        else: rename_save_directory = f'{rename_save_directory}-DMF'
        # rename_save_directory = f'{rename_save_directory}-{validation_accuracy * 100:.2f}%'
        
        compare_figure(my_history)
        model.save(last_model_path)

        if output_answer:
            model_paths = {
                'last': last_model_path,
                'min val_loss': best_model_min_val_loss_path,
                'max val_accuracy': best_model_max_val_accuracy_path,
            }
            model_scores = {}
            best_score_not_repaired = 0
            score_not_repaired = 0
            score_repaired_0 = 0
            score_repaired_1 = 0
            for model_name, model_path in model_paths.items():
                model = Sequential()
                model = load_model(model_path)
                estimate_and_write_to_file(model)
                for song_index in validation_file_index:
                    ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
                    est_file = f'{data_directory}/{song_index+1}/est_file.txt'
                    score_not_repaired += get_sevenths_score(ref_file_path, est_file)
                    score_repaired_0 += get_sevenths_score(ref_file_path, est_file, True)
                    score_repaired_1 += get_sevenths_score(ref_file_path, est_file, True, 1)
                score_not_repaired /= len(validation_file_index)
                score_repaired_0 /= len(validation_file_index)
                score_repaired_1 /= len(validation_file_index)
                model_scores[model_name] = [score_not_repaired, score_repaired_0, score_repaired_1]
                if score_not_repaired > best_score_not_repaired: best_score_not_repaired = score_not_repaired
                print(f"\nAverage validation score by '{model_name}' model: {score_not_repaired} (not repaired)")
                print(f"Average validation score by '{model_name}' model: {score_repaired_0} (repaired 0)")
                print(f"Average validation score by '{model_name}' model: {score_repaired_1} (repaired 1)")
            rename_save_directory = f'{rename_save_directory}-{best_score_not_repaired * 100:.5f}'

        record_details(cost_time, model_scores)

        os.rename(save_directory, rename_save_directory)


if __name__ == "__main__":
    os.system('cls')
    main()
