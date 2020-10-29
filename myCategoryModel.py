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
from mapping import my_mapping_dict


''' Parameters '''
RAMDON_SEED = 1

EPOCHS = 3000
BATCH_SIZE = 305600
PATIENCE = 100

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

frames_per_data = data_divide_amount * 21
frames_on_one_side = int((frames_per_data - 1) / 2)

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
    model.add(Dense(504, input_shape=(504, ), activation='sigmoid'))
    model.add(Dense(504, input_shape=(504, ), activation='relu'))
    # model.add(Dense(504, input_shape=(504, ), activation='sigmoid'))
    # model.add(Dense(504, input_shape=(504, ), activation='relu'))
    model.add(Dense(len(mapping_dictionary), input_shape=(504, ), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def processData(train_file_index):
    ''' Dataset '''
    X = []
    X_train = []
    X_validation = []
    Y = []
    Y_train = []
    Y_validation = []

    error_list = []

    toolbar_width = 100
    sys.stdout.write("\nReading train & validation datas from each songs in '%s'.\n[%s]" % (data_directory, " " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))

    for song_index in range(file_amount):

        if data_divide_amount == 1: read_csv_file_path = f'{data_directory}/{song_index+1}/data.csv'
        else: read_csv_file_path = f'{data_directory}/{song_index+1}/data_divide_{data_divide_amount}.csv'
        
        data = pd.read_csv(read_csv_file_path, index_col=0)
        data = data.values

        label_index = data.shape[1]-1
        data_index = np.arange(len(data))
        np.random.shuffle(data_index)
        data_index = data_index[:int(len(data_index) * 0.75)]
        data = np.vstack((np.zeros((frames_on_one_side, 25)), data, np.zeros((frames_on_one_side, 25))))

        for index, shuffle_index in enumerate(data_index):
            try:
                label = int(mapping_dictionary['Other'])
                for chord in mapping_dictionary.keys():
                    if chord in str(data[index+frames_on_one_side, label_index]).strip():
                        label = int(mapping_dictionary[chord])
                        break

                shuffle_label = int(mapping_dictionary['Other'])
                for chord in mapping_dictionary.keys():
                    if chord in str(data[shuffle_index+frames_on_one_side, label_index]).strip():
                        shuffle_label = int(mapping_dictionary[chord])
                        break

                X.append(data[index:index+frames_per_data, 0:label_index].reshape(label_index*frames_per_data))
                Y.append(label)
                if song_index in train_file_index:
                    X_train.append(data[shuffle_index:shuffle_index+frames_per_data, 0:label_index].reshape(label_index*frames_per_data))
                    Y_train.append(shuffle_label)
                else:
                    X_validation.append(data[shuffle_index:shuffle_index+frames_per_data, 0:label_index].reshape(label_index*frames_per_data))
                    Y_validation.append(shuffle_label)
            except:
                error_list.append({
                    'index': index+frames_on_one_side,
                    'label string': data[index+frames_on_one_side, label_index],
                    'shuffle_index': shuffle_index+frames_on_one_side,
                    'shuffle_label string': data[shuffle_index+frames_on_one_side, label_index]
                })

        if debug_mode:
            sys.stdout.write("=" * 5)
            sys.stdout.flush()
        elif (song_index + 1) % (file_amount / toolbar_width) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n\n")

    X = np.asarray(X).astype(np.float32)
    X_train = np.asarray(X_train).astype(np.float32)
    X_validation = np.asarray(X_validation).astype(np.float32)
    Y = to_categorical(Y, num_classes=len(mapping_dictionary))
    Y_train = to_categorical(Y_train, num_classes=len(mapping_dictionary))
    Y_validation = to_categorical(Y_validation, num_classes=len(mapping_dictionary))
    
    print('Error amount: ', len(error_list))
    print(f'Train data: {int(len(X_train)):7d} / All data: {len(X):7d}\n')
    time.sleep(3)

    return (X, Y), (X_train, Y_train), (X_validation, Y_validation)


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
        data = np.vstack((np.zeros((frames_on_one_side, label_index)), data, np.zeros((frames_on_one_side, label_index))))

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
        f.write(f'data_divide_amount = {data_divide_amount}\n')
        f.write(f'frames_per_data = {frames_per_data}\n')
        f.write(f'\n')
        f.write(f'epochs = {EPOCHS}\n')
        f.write(f'batch_size = {BATCH_SIZE}\n')
        f.write(f'\n')
        f.write(f'cost_time: {cost_time}\n')
        # f.write(f'validation_loss: {validation_loss}\n')
        # f.write(f'validation_accuracy: {validation_accuracy}\n')
        # f.write(f'original_loss (all data): {original_loss}\n')
        # f.write(f'original_accuracy (all data): {original_accuracy}\n')
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
            # (X, Y), (X_train, Y_train), (X_validation, Y_validation) = processData(train_file_index)
            # loss, accuracy = model.evaluate(X, Y)
            # print(f'\nEvaluate with original data (all) - Loss: {loss}, Accuracy: {accuracy * 100:.3f}%')
            estimate_and_write_to_file(model)
        
            total_score = 0
            train_score = 0
            validation_score = 0

            for song_index in file_index:
                ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
                est_file = f'{data_directory}/{song_index+1}/est_file.txt'
                score = get_sevenths_score(ref_file=ref_file_path, est_file=est_file)
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

        (X, Y), (X_train, Y_train), (X_validation, Y_validation) = processData(train_file_index)

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
        # ES_min_loss = EarlyStopping(
        #     monitor='loss', mode='min',
        #     verbose=1, patience=20
        # )
        # ES_min_val_loss = EarlyStopping(
        #     monitor='val_loss', mode='min',
        #     verbose=1, patience=50
        # )
        ES_max_val_accuracy = EarlyStopping(
            monitor='val_accuracy', mode='max',
            verbose=1, patience=PATIENCE
        )
        history = model.fit(
            X_train, Y_train, 
            # validation_split=validation_split,
            validation_data=(X_validation, Y_validation),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[MCP_min_val_loss, MCP_max_val_acc, ES_max_val_accuracy]
        )

        end_time = datetime.datetime.now()
        cost_time = str(end_time-start_time)

        print(f'\nLearning cost time: {cost_time}\n')
        # validation_loss, validation_accuracy = model.evaluate(X_validation, Y_validation)
        # print(f'Evaluate with validation data - Loss: {validation_loss}, Accuracy: {validation_accuracy * 100:.2f}%\n')
        # original_loss, original_accuracy = model.evaluate(X, Y)
        # print(f'Evaluate with original data (all) - Loss: {original_loss}, Accuracy: {original_accuracy * 100:.2f}%')
        
        ''' Save figure & model '''
        if data_divide_amount == 1: rename_save_directory = f'{rename_save_directory}-MF'
        else: rename_save_directory = f'{rename_save_directory}-DMF'
        # rename_save_directory = f'{rename_save_directory}-{validation_accuracy * 100:.2f}%'
        
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
                for song_index in validation_file_index:
                    ref_file_path = f'{data_directory}/{song_index+1}/ground_truth.txt'
                    est_file = f'{data_directory}/{song_index+1}/est_file.txt'
                    score += get_sevenths_score(ref_file=ref_file_path, est_file=est_file)
                score /= len(validation_file_index)
                model_scores[model_name] = score
                if score > best_score: best_score = score
                print(f"\nAverage validation score by '{model_name}' model: {score}")
            rename_save_directory = f'{rename_save_directory}-{best_score * 100:.5f}'

        record_details(cost_time, model_scores)

        os.rename(save_directory, rename_save_directory)


if __name__ == "__main__":
    os.system('cls')
    main()
