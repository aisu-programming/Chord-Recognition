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

from attentionModel import AttentionModel, BiDirectionalSelfAttention, SelfAttentionBlock, PositionwiseConvolution, customized_loss
from score import get_sevenths_score
from mapping import my_mapping_dict

from tensorflow import keras


''' Parameters '''
RAMDON_SEED = 1

EPOCHS = 14
# 872950
BATCH_PER_TRAIN = 10000
BATCH_SIZE = BATCH_PER_TRAIN * 10


''' Global variables'''
est_file_directory = 'CE500_test'
est_file_amount = 300
train_file_directory = 'CE200'
train_file_amount = 200

data_divide_amount = 1
sec_per_frame = 512.0 / 22050.0 / data_divide_amount

# frames_per_data = data_divide_amount * 401
frames_per_data = data_divide_amount * 501
one_side_frames = int((frames_per_data - 1) / 2)

only_cqt = True
if only_cqt: input_amount = 12
else: input_amount = 24

mapping_dictionary = my_mapping_dict

display_figure = False

now = time.localtime()
now_date = time.strftime('%Y-%m-%d', now)
now_time = time.strftime('%H.%M.%S', now)
save_directory = f'model/{now_date}/{now_time}'


''' Codes '''
def build_model():
    model = Sequential()
    """
    model.add(Dense(4812, input_shape=(4812, ), activation='relu'))
    model.add(Dense(4812, activation='relu'))
    model.add(Dense(4812, activation='relu'))
    model.add(Dense(4812, activation='relu'))
    """
    model.add(Dense(6012, input_shape=(6012, ), activation='relu'))
    model.add(Dense(6012, activation='relu'))
    model.add(Dense(6012, activation='relu'))
    model.add(Dense(6012, activation='relu'))

    model.add(Dense(len(mapping_dictionary), activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss=customized_loss, optimizer=opt, metrics=["accuracy"])
    print('')
    model.summary()
    print('')
    return model


def readAndSplitData(train_file_index):
    
    X = {'train': [], 'validation': []}
    Y = {'train': [], 'validation': []}

    toolbar_width = 100
    sys.stdout.write("\nReading train & validation datas from each songs in '%s'.\n[%s]" % (train_file_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    filling_row = [0.0] * input_amount
    for song_index in range(train_file_amount):

        for _ in range(one_side_frames):
            if song_index in train_file_index:
                X['train'].append(filling_row)
                Y['train'].append(0)
            else:
                X['validation'].append(filling_row)
                Y['validation'].append(0)

        if data_divide_amount == 1: read_csv_file_path = f'{train_file_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{train_file_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
        
        data = pd.read_csv(read_csv_file_path, index_col=0)
        data = data.values

        for row in data:
            label = row[24]
            for chord in ['maj6', 'maj9', 'maj11', 'maj13', 'min6', 'min9', 'min11', 'min13']:
                if chord in label:
                    label = 1
                    break
            if label == row[24]:
                for chord in mapping_dictionary.keys():
                    if chord in label:
                        label = int(mapping_dictionary[chord])
                        break
            if label == row[24]: label = 1

            if song_index in train_file_index:
                X['train'].append(row[:input_amount])
                Y['train'].append(label)
            else:
                X['validation'].append(row[:input_amount])
                Y['validation'].append(label)

        if (song_index + 1) % (train_file_amount / toolbar_width) == 0:
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


def estimate_and_write_to_file(model_dict):

    toolbar_width = 100
    sys.stdout.write(f"\nEstimating and write to est_files.\n[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(est_file_amount):

        if data_divide_amount == 1: read_csv_file_path = f'{est_file_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{est_file_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'

        original_data = pd.read_csv(read_csv_file_path, index_col=0)
        original_data = original_data.values
        original_data_length = len(original_data)
        
        filling_row = [0.0] * input_amount
        data = []
        for _ in range(one_side_frames): data.append(filling_row)
        for row in original_data: data.append(row[:input_amount])
        for _ in range(one_side_frames): data.append(filling_row)

        X = []
        for i in range(original_data_length):
            X_tmp = []
            for j in range(frames_per_data):
                X_tmp.append(data[i+j][:input_amount])
            X.append(X_tmp)
            
        X = np.array(X).astype(np.float32).reshape(original_data_length, input_amount*frames_per_data)
        for model_name, model in model_dict.items():
            Y_pred = model.predict_classes(X)
            # Y_pred = Y_pred[:, -1].reshape(len(Y_pred), )
            with open(f'{est_file_directory}/{song_index+1}/est_file_{model_name}.txt', mode='w') as f:
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

        if (song_index + 1) % (est_file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")
    return


def record_details(cost_time):
    print(f"\nRecording details... ", end='')
    with open(f'{save_directory}/details.txt', mode='w') as f:
        f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
        f.write(f'\n')
        f.write(f'EPOCHS = {EPOCHS}\n')
        f.write(f'BATCH_SIZE = {BATCH_SIZE}\n')
        f.write(f'\n')
        f.write(f'data_divide_amount = {data_divide_amount}\n')
        f.write(f'frames_per_data = {frames_per_data}\n')
        f.write(f'\n')
        f.write(f'cost_time: {cost_time}\n')
    print('Done.')
    return


def main():

    np.random.seed(RAMDON_SEED)
    train_file_index = np.arange(train_file_amount)
    np.random.shuffle(train_file_index)

    ''' Model '''
    X, Y = readAndSplitData(train_file_index)

    model = build_model()
    # model = AttentionModel(frames_per_data=frames_per_data)
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    last_model_path = f'{save_directory}/model.h5'

    rename_save_directory = save_directory

    start_time = datetime.datetime.now() # Set timer

    train_index_array = np.arange(one_side_frames, len(X['train'])-one_side_frames)
    my_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    ''' Train '''
    for epoch in range(EPOCHS):
        np.random.shuffle(train_index_array)
        X_train_batch = []
        Y_train_batch = []
        for i in range(BATCH_SIZE):
            ti = train_index_array[i]
            X_train_batch.append(X['train'][ti-one_side_frames:ti+one_side_frames+1])
            Y_train_batch.append(Y['train'][ti])
            # Y_train_batch.append(Y['train'][ti-one_side_frames:ti+one_side_frames+1])
        print(f'epoch: {epoch+1}/{EPOCHS} | BATCH_SIZE: {BATCH_SIZE} | BATCH_PER_TRAIN: {BATCH_PER_TRAIN}')
        model.fit(
            np.array(X_train_batch).reshape(BATCH_SIZE, frames_per_data*input_amount),
            np.array(Y_train_batch), # .reshape(BATCH_SIZE, len(mapping_dictionary)),
            batch_size=BATCH_PER_TRAIN,
        )
        print('')

    end_time = datetime.datetime.now()
    cost_time = str(end_time-start_time)
    print(f'\nLearning cost time: {cost_time}')
    
    ''' Save figure & model '''
    if data_divide_amount == 1: rename_save_directory = f'{rename_save_directory}-MF'
    else: rename_save_directory = f'{rename_save_directory}-DMF'
    # rename_save_directory = f'{rename_save_directory}-{validation_accuracy * 100:.2f}%'
    
    compare_figure(my_history)
    model.save(last_model_path)
    time.sleep(1)

    custom_objects = {
        'BiDirectionalSelfAttention': BiDirectionalSelfAttention,
        'SelfAttentionBlock': SelfAttentionBlock,
        'PositionwiseConvolution': PositionwiseConvolution,
        'customized_loss':customized_loss,
    }
    estimate_and_write_to_file({
        'last': load_model(last_model_path, custom_objects=custom_objects),
    })

    record_details(cost_time)

    os.rename(save_directory, rename_save_directory)


if __name__ == "__main__":
    os.system('cls')
    main()
