import os

import tensorflow as tf
from tensorflow.keras.models import Sequential

from mapping import my_mapping_dict
mapping_dictionary = my_mapping_dict


from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense
def ConvolutionModel():
    model = Sequential()
    model.add(ZeroPadding2D(padding=1, input_shape=(120, 12, 1)))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(16, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D(padding=1))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Reshape((-1, 64)))
    model.add(Bidirectional(LSTM(50, input_length=10, input_dim=64)))
    model.add(Dense(len(mapping_dictionary), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print('')
    model.summary()
    print('')
    return model

if __name__ == "__main__":
    ConvolutionModel()