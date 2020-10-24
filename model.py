import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from mapping import mapping_dict

np.random.seed(1)

data = pd.read_csv(f'CE200_sample/{1}/data.csv', index_col=0)
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

model = Sequential()
# model = load_model('model/third_try.h5')

model.add(Dense(36, input_shape=(24, ), activation='relu'))
model.add(Dense(48, input_shape=(36, ), activation='relu'))
model.add(Dense(544, input_shape=(48, ), activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=20, batch_size=1)

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)

model.save('model/third_try.h5')

Y_pred = model.predict_classes(X)
# for index in range(len(Y_pred)):
#     if Y_pred[index] != np.where(Y[index]==1)[0]:
#         print(f'{index:6} | My: {Y_pred[index]:3} <---> Right: {int(np.where(Y[index]==1)[0]):3}')

with open('test.txt', mode='w') as f:
    sec_per_frame = 512.0 / 22050.0
    for index, label in enumerate(Y_pred):
        for k, v in mapping_dict.items():
            if v == label:
                f.write(f'{sec_per_frame * index:.06f}\t{sec_per_frame * (index + 1):.06f}\t{k}\n')
                break