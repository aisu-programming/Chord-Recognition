''' Libraries '''
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

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
    X_test, Y_test = X[train_amount:], Y[train_amount:]

    ''' Model '''
    model = Sequential()
    if load_exist_model:
        model = load_model(load_model_path)
        loss, accuracy = model.evaluate(X_test, Y_test)
        print(f'Evaluate with test data. Accuracy: {accuracy * 100:.3f}%')
    else:
        model = adjust_model(model)
        os.system('cls')
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs, batch_size=batch_size
        )
        loss, accuracy = model.evaluate(X_test, Y_test)
        print(f'\nEvaluate with test data. Accuracy: {accuracy * 100:.2f}%\n')

        ''' Save figure & model '''
        global save_directory
        save_directory = f'{save_directory}-oneFrame-{accuracy * 100:.2f}%'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        compare_figure(history, display=True)
        model.save(f'{save_directory}/model.h5')

    ''' Estimate then Output '''
    if output_answer:
        Y_pred = model.predict_classes(X)
        with open(output_file_path, mode='w') as f:
            sec_per_frame = 512.0 / 22050.0
            for index, label in enumerate(Y_pred):
                for k, v in mapping_dict.items():
                    if v == label:
                        f.write(f'{sec_per_frame * index:.06f}\t{sec_per_frame * (index + 1):.06f}\t{k}\n')
                        break


if __name__ == "__main__":
    os.system('cls')
    main()