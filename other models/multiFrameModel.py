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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from score import get_sevenths_score
from mapping import mapping_dict


''' Parameters '''
RAMDON_SEED = 1


''' Global variables'''
debug_mode = True
if debug_mode:
    data_directory = 'CE200_sample'
    file_amount = 20
else:
    data_directory = 'CE200'
    file_amount = 200

data_divide_amount = 5
sec_per_frame = 512.0 / 22050.0 / data_divide_amount
each_song_data_proportion = 5.0 / 5.0

frames_per_data = 25
frames_on_one_side = int((frames_per_data - 1) / 2)

test_split = 0.15
validation_split = 0.2

epochs = 500
batch_size = 28418

load_exist_model = True
load_model_path = 'model/2020-10-28/16.19.39-DMF-79.12%-83.03837/model.h5'

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
    model.add(Dense(900, input_shape=(600, ), activation='sigmoid'))
    model.add(Dense(900, input_shape=(900, ), activation='relu'))
    model.add(Dense(544, input_shape=(900, ), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def processData():
    ''' Dataset '''
    np.random.seed(RAMDON_SEED)

    X = []
    X_shuffle = []
    Y = []
    Y_shuffle = []

    error_list = []

    toolbar_width = 100
    sys.stdout.write("Reading data from each songs in '%s'.\n[%s]" % (data_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    for song_index in range(file_amount):
    
        if data_divide_amount == 1: read_csv_file_path = f'{data_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{data_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
        
        data = pd.read_csv(read_csv_file_path, index_col=0)
        data['label'] = data['label'].map(mapping_dict)
        data = data.values

        label_index = data.shape[1]-1
        data_index = np.arange(len(data))
        np.random.shuffle(data_index)
        # print(f'{int(len(data_index) * each_song_data_proportion)}:6d / {len(X):8d}')
        data_index = data_index[:int(len(data_index) * each_song_data_proportion)]
        data = np.vstack((np.zeros((frames_on_one_side, 25)), data, np.zeros((frames_on_one_side, 25))))

        for index, shuffle_index in enumerate(data_index):
            try:
                label = int(data[index+frames_on_one_side, label_index])
                shuffle_label = int(data[shuffle_index+frames_on_one_side, label_index])
                Y.append(label)
                Y_shuffle.append(shuffle_label)
                X.append(data[index:index+frames_per_data, 0:label_index].reshape(label_index*frames_per_data))
                X_shuffle.append(data[shuffle_index:shuffle_index+frames_per_data, 0:label_index].reshape(label_index*frames_per_data))
            except:
                error_list.append({
                    'index': index+frames_on_one_side,
                    'label string': data[index+frames_on_one_side, label_index],
                    'shuffle_index': shuffle_index+frames_on_one_side,
                    'shuffle_label string': data[shuffle_index+frames_on_one_side, label_index]
                })
                continue

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")

    X = np.array(X)
    X_shuffle = np.array(X_shuffle)
    Y = to_categorical(Y, num_classes=len(mapping_dict))
    Y_shuffle = to_categorical(Y_shuffle, num_classes=len(mapping_dict))

    X_train, X_test = np.split(X_shuffle, [int(len(X_shuffle)*(1.0-test_split))])
    Y_train, Y_test = np.split(Y_shuffle, [int(len(Y_shuffle)*(1.0-test_split))])
    
    print('Error amount: ', len(error_list))
    print(f'Train data: {int(len(X_train)):7d} / All data: {len(X):7d}\n')
    time.sleep(3)
    # print(f'Length of X: {len(X)}, X_shuffle: {len(X_shuffle)}, Y: {len(Y)}, Y_shuffle: {len(Y_shuffle)}')
    # os.system('pause')

    # return (X, Y), (X_shuffle, Y_shuffle), (X_train, Y_train), (X_test, Y_test)
    return (X, Y), (X_train, Y_train), (X_test, Y_test)


def compare_figure(history):
    
    import matplotlib.pyplot as plt
    
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    validation_loss = history.history['val_loss']
    validation_accuracy = history.history['val_accuracy']
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

    print('\n')

    for song_index in range(file_amount):

        print(f"Estimating and write to '{data_directory}/{song_index+1}/est_file.txt'... ", end='')

        if data_divide_amount == 1: read_csv_file_path = f'{data_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{data_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'

        data = pd.read_csv(read_csv_file_path, index_col=0)
        data['label'] = data['label'].map(mapping_dict)
        data = data.values

        label_index = data.shape[1]
        original_data_length = len(data)
        data = np.vstack((np.zeros((frames_on_one_side, label_index)), data, np.zeros((frames_on_one_side, label_index))))

        X = []
        for i in range(original_data_length):
            X.append(data[i:i+frames_per_data, 0:label_index-1].reshape((label_index-1)*frames_per_data))
            
        X = np.array(X)
        Y_pred = model.predict_classes(X)
        with open(f'{data_directory}/{song_index+1}/est_file.txt', mode='w') as f:
            index_now = 0
            index_last = 0
            while index_now < len(Y_pred):
                if (index_now == len(Y_pred) - 1) or (Y_pred[index_now] != Y_pred[index_now+1]):
                    for k, v in mapping_dict.items():
                        if v == Y_pred[index_now]:
                            f.write(f'{sec_per_frame*index_last:.06f}\t{sec_per_frame*(index_now+1):.06f}\t{k}\n')
                            index_last = index_now + 1
                            break
                index_now += 1

        print('Done.')

    return


def record_details(cost_time, test_loss, test_accuracy, original_loss, original_accuracy, model_scores):
    print(f"\nRecording details... ", end='')
    with open(f'{save_directory}/details.txt', mode='w') as f:
        f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
        f.write(f'\n')
        f.write(f'data_divide_amount = {data_divide_amount}\n')
        f.write(f'frames_per_data = {frames_per_data}\n')
        f.write(f'\n')
        f.write(f'test_split = {test_split}\n')
        f.write(f'validation_split = {validation_split}\n')
        # f.write(f'train data proportion: {test_split * (1 - validation_split):2.0f}%\n')
        # f.write(f'test data proportion: {(1 - test_split) * 100:2.0f}%\n')
        # f.write(f'validation data proportion: {test_split * validation_split:2.0f}%\n')
        f.write(f'\n')
        f.write(f'epochs = {epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'\n')
        f.write(f'cost_time: {cost_time}\n')
        f.write(f'test_loss: {test_loss}\n')
        f.write(f'test_accuracy: {test_accuracy}\n')
        f.write(f'original_loss (all data): {original_loss}\n')
        f.write(f'original_accuracy (all data): {original_accuracy}\n')
        f.write(f'\n')
        for model_name, model_score in model_scores.items():
            f.write(f"Score by '{model_name}' model: {model_score}\n")
    print('Done.')
    return


def main():

    ''' Data '''
    # (X, Y), (X_shuffle, Y_shuffle), (X_train, Y_train), (X_test, Y_test) = processData()
    (X, Y), (X_train, Y_train), (X_test, Y_test) = processData()

    ''' Model '''
    model = Sequential()
    if load_exist_model:

        model = load_model(load_model_path)
        # loss, accuracy = model.evaluate(X, Y)
        # print(f'\nEvaluate with original data (all) - Loss: {loss}, Accuracy: {accuracy * 100:.3f}%')
        
        estimate_and_write_to_file(model)
        score = 0
        for song_index in range(file_amount):
            ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
            est_file = f'{data_directory}/{song_index+1}/est_file.txt'
            score += get_sevenths_score(ref_file=ref_file_path, est_file=est_file)
        score /= file_amount
        print(f"\nAverage score by model: {score}")

    else:
        model = adjust_model(model)
        os.system('cls')
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        last_model_path = f'{save_directory}/model.h5'
        best_model_min_val_loss_path = f'{save_directory}/best_model_min_val_loss.h5'
        best_model_max_val_accuracy_path = f'{save_directory}/best_model_max_val_accuracy.h5'

        rename_save_directory = save_directory
        start_time = datetime.datetime.now() # Set timer

        MCP_min_val_loss = ModelCheckpoint(
            best_model_min_val_loss_path,
            monitor='val_loss', mode='min', verbose=1, save_best_only=True
        )
        MCP_max_val_acc = ModelCheckpoint(
            best_model_max_val_accuracy_path,
            monitor='val_accuracy', mode='max', verbose=1, save_best_only=True
        )
        ES = EarlyStopping(
            monitor='val_accuracy', mode='max',
            verbose=1, patience=20
        )
        history = model.fit(
            X_train, Y_train, 
            validation_split=validation_split,
            # validation_data=(X_test, Y_test),
            epochs=epochs, batch_size=batch_size,
            callbacks=[MCP_min_val_loss, MCP_max_val_acc, ES]
        )

        end_time = datetime.datetime.now()
        cost_time = str(end_time-start_time)

        print(f'\nLearning cost time: {cost_time}\n')
        test_loss, test_accuracy = model.evaluate(X_test, Y_test)
        print(f'Evaluate with test data - Loss: {test_loss}, Accuracy: {test_accuracy * 100:.2f}%\n')
        original_loss, original_accuracy = model.evaluate(X, Y)
        print(f'Evaluate with original data (all) - Loss: {original_loss}, Accuracy: {original_accuracy * 100:.2f}%')
        
        ''' Save figure & model '''
        if data_divide_amount == 1: rename_save_directory = f'{rename_save_directory}-MF-{original_accuracy * 100:.2f}%'
        else: rename_save_directory = f'{rename_save_directory}-DMF-{original_accuracy * 100:.2f}%'
        
        compare_figure(history)
        model.save(last_model_path)

        if output_answer:
            model_paths = {
                'last': last_model_path,
                'min val_loss': best_model_min_val_loss_path,
                'max val_accuracy': best_model_max_val_accuracy_path,
            }
            model_scores = {}
            best_score = 0
            score = 0
            for model_name, model_path in model_paths.items():
                model = Sequential()
                model = load_model(model_path)
                estimate_and_write_to_file(model)
                for song_index in range(file_amount):
                    ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
                    est_file = f'{data_directory}/{song_index+1}/est_file.txt'
                    score += get_sevenths_score(ref_file=ref_file_path, est_file=est_file)
                score /= file_amount
                model_scores[model_name] = score
                if score > best_score: best_score = score
                print(f"\nAverage score by '{model_name}' model: {score}")
            rename_save_directory = f'{rename_save_directory}-{best_score * 100:.5f}'

        record_details(cost_time, test_loss, test_accuracy, original_loss, original_accuracy, model_scores)

        os.rename(save_directory, rename_save_directory)


if __name__ == "__main__":
    os.system('cls')
    main()