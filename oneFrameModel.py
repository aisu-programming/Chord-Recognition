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
epochs = 150
batch_size = 500

load_exist_model = False
load_model_path = 'model/2020-10-24/third_try.h5'

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
    model.add(Dense(48, input_shape=(24, ), activation='sigmoid'))
    model.add(Dense(96, input_shape=(48, ), activation='relu'))
    model.add(Dense(544, input_shape=(96, ), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def estimate_and_write_to_file(model, X):
    print(f"\nEstimating and write to '{output_file_path}'...\n")
    Y_pred = model.predict_classes(X)
    with open(output_file_path, mode='w') as f:
        sec_per_frame = 512.0 / 22050.0
        for index, label in enumerate(Y_pred):
            for k, v in mapping_dict.items():
                if v == label:
                    f.write(f'{sec_per_frame * index:.06f}\t{sec_per_frame * (index + 1):.06f}\t{k}\n')
                    break
    print('\nDone.')
    return


def record_details(save_directory, accuracy, score):
    print(f"\nRecording details... ", end='')
    with open(f'{save_directory}/details.txt', mode='w') as f:
        f.write(f'RAMDON_SEED = {RAMDON_SEED}\n')
        f.write(f'\n')
        f.write(f'epochs = {epochs}\n')
        f.write(f'batch_size = {batch_size}\n')
        f.write(f'\n')
        f.write(f'accuracy = {accuracy}\n')
        f.write(f'score = {score}\n')
    print('Done.')
    return


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


def main():

    ''' Dataset '''
    np.random.seed(RAMDON_SEED)
    
    data = pd.read_csv(f'CE200_sample/1/data.csv', index_col=0)
    data = data.drop(['Song No.', 'Frame No.'], axis=1)
    data['label'] = data['label'].map(mapping_dict)

    dataset = data.values

    X = dataset[:, 0:24]
    Y = to_categorical(dataset[:, 24])

    # np.random.shuffle(dataset)
    # X = dataset[:, 0:24]
    # Y = to_categorical(dataset[:, 24])
    train_amount = 5500
    X_train, Y_train = X[:train_amount], Y[:train_amount]
    X_validate, Y_validate = X[train_amount:], Y[train_amount:]

    ''' Model '''
    model = Sequential()
    if load_exist_model:
        model = load_model(load_model_path)
        loss, accuracy = model.evaluate(X_validate, Y_validate)
        print(f'Evaluate with validation data. Accuracy: {accuracy * 100:.3f}%')
        # loss, accuracy = model.evaluate(X, Y)
        # print(f'\nEvaluate with original data. Accuracy: {accuracy * 100:.2f}%')
    else:
        model = adjust_model(model)
        os.system('cls')
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_validate, Y_validate),
            epochs=epochs, batch_size=batch_size
        )
        loss, accuracy = model.evaluate(X_validate, Y_validate)
        print(f'\nEvaluate with validation data. Accuracy: {accuracy * 100:.2f}%\n')
        # loss, accuracy = model.evaluate(X, Y)
        # print(f'\nEvaluate with original data. Accuracy: {accuracy * 100:.2f}%')

        ''' Save figure & model '''
        global save_directory
        save_directory = f'{save_directory}-of-{accuracy * 100:.2f}%'
        if output_answer:
            estimate_and_write_to_file(model, X)
            score = get_sevenths_score(ref_file='CE200_sample/1/ground_truth.txt', est_file='test.txt')
            print(f'\nScore: {score}')
            save_directory = f'{save_directory}-{score * 100:.3f}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        compare_figure(history, display=True)
        model.save(f'{save_directory}/model.h5')
        record_details(save_directory, accuracy, score)


if __name__ == "__main__":
    os.system('cls')
    main()