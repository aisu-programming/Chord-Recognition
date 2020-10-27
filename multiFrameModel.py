''' Libraries '''
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from score import get_sevenths_score
from mapping import mapping_dict


''' Parameters '''
RAMDON_SEED = 1


''' Global variables '''
data_divide_amount = 5
sec_per_frame = 512.0 / 22050.0 / data_divide_amount

frames_per_data = 7 # please set an odd number
frames_on_one_side = int((frames_per_data - 1) / 2)

data_split_percentage = 0.6

epochs = 50
batch_size = 1

load_exist_model = False
load_model_path = 'model/2020-10-27/14.49.42-MF-96.71%-97.312/model.h5'

auto_save_path = True
if auto_save_path:
    now = time.localtime()
    now_date = time.strftime('%Y-%m-%d', now)
    now_time = time.strftime('%H.%M.%S', now)
    save_directory = f'model/{now_date}/{now_time}'
else:
    save_directory = ''

output_answer = True
output_file_path = 'test.txt'


''' Codes '''
def adjust_model(model):
    model.add(Dense(252, input_shape=(168, ), activation='sigmoid'))
    model.add(Dense(378, input_shape=(252, ), activation='relu'))
    model.add(Dense(544, input_shape=(378, ), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def processData():
    ''' Dataset '''
    np.random.seed(RAMDON_SEED)
    
    if data_divide_amount == 1: read_csv_file_path = f'CE200_sample/1/data.csv'
    else: read_csv_file_path = f'CE200_sample/1/data_divide_{data_divide_amount}.csv'
    
    data = pd.read_csv(read_csv_file_path, index_col=0)
    data = data.drop(['Song No.', 'Frame No.'], axis=1)
    data['label'] = data['label'].map(mapping_dict)

    dataset = data.values

    label_index = dataset.shape[1]-1
    data_index = np.arange(len(dataset))
    np.random.shuffle(data_index)

    dataset = np.vstack((np.zeros((frames_on_one_side, 25)), dataset))
    dataset = np.vstack((dataset, np.zeros((frames_on_one_side, 25))))

    X = []
    X_shuffle = []
    Y_shuffle = []
    for i, shuffle_i in enumerate(data_index):

        X_tmp = dataset[i:i+frames_per_data, 0:label_index]
        X_shuffle_tmp = dataset[shuffle_i:shuffle_i+frames_per_data, 0:label_index]

        X_tmp = X_tmp.reshape(label_index*frames_per_data)
        X_shuffle_tmp = X_shuffle_tmp.reshape(label_index*frames_per_data)

        X.append(X_tmp)
        X_shuffle.append(X_shuffle_tmp)
        Y_shuffle.append(dataset[:, label_index][shuffle_i+frames_on_one_side])

    X = np.array(X)
    X_shuffle = np.array(X_shuffle)
    Y_shuffle = to_categorical(Y_shuffle, num_classes=len(mapping_dict))

    X_train, X_validate = np.split(X_shuffle, [int(len(X_shuffle)*data_split_percentage)])
    Y_train, Y_validate = np.split(Y_shuffle, [int(len(Y_shuffle)*data_split_percentage)])
    
    return (X), (X_train, Y_train), (X_validate, Y_validate)


def compare_figure(history, display=True):
    
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
    if display: plt.show()

    return


def estimate_and_write_to_file(model, X):
    print(f"\nEstimating and write to '{output_file_path}'...\n")
    Y_pred = model.predict_classes(X)
    with open(output_file_path, mode='w') as f:
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
    print('\nDone.')
    return


def record_details(save_directory, accuracy, score):
    print(f"\nRecording details... ", end='')
    with open(f'{save_directory}/details.txt', mode='w') as f:
        f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
        f.write(f'\n')
        f.write(f'frames_per_data = {frames_per_data}\n')
        f.write(f'data_split_percentage = {data_split_percentage}\n')
        f.write(f'epochs = {epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'\n')
        f.write(f'accuracy = {accuracy}\n')
        f.write(f'score = {score}\n')
    print('Done.')
    return


def main():

    ''' Data '''
    (X), (X_train, Y_train), (X_validate, Y_validate) = processData()

    ''' Model '''
    model = Sequential()
    if load_exist_model:
        model = load_model(load_model_path)
        loss, accuracy = model.evaluate(X_validate, Y_validate)
        print(f'\nEvaluate with test data - Loss: {loss}, Accuracy: {accuracy * 100:.3f}%')
        # loss, accuracy = model.evaluate(X, Y)
        # print(f'\nEvaluate with original data. Accuracy: {accuracy * 100:.2f}%')
        if output_answer:
            estimate_and_write_to_file(model, X)
            score = get_sevenths_score(ref_file='CE200_sample/1/ground_truth.txt', est_file='test.txt')
            print(f'\nScore: {score}')
    else:
        model = adjust_model(model)
        os.system('cls')
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_validate, Y_validate),
            epochs=epochs, batch_size=batch_size
        )
        loss, accuracy = model.evaluate(X_validate, Y_validate)
        print(f'\nEvaluate with validation data. Accuracy: {accuracy * 100:.2f}%')
        # loss, accuracy = model.evaluate(X, Y)
        # print(f'\nEvaluate with original data. Accuracy: {accuracy * 100:.2f}%')
        
        ''' Save figure & model '''
        global save_directory
        save_directory = f'{save_directory}-MF-{accuracy * 100:.2f}%'
        if output_answer:
            estimate_and_write_to_file(model, X)
            score = get_sevenths_score(ref_file='CE200_sample/1/ground_truth.txt', est_file='test.txt')
            print(f'\nScore: {score}')
            save_directory = f'{save_directory}-{score * 100:.3f}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        compare_figure(history, display=False)
        model.save(f'{save_directory}/model.h5')
        record_details(save_directory, accuracy, score)


if __name__ == "__main__":
    os.system('cls')
    main()