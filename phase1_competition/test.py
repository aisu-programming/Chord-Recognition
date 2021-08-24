# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dropout, Dense

# from mapping import my_mapping_dict


''' Codes '''
# import sys
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Input
# from tensorflow.keras.layers import Layer
# class TestLayer(Layer):
#     def __init__(self, units, direction, **kwargs):
#         super(TestLayer, self).__init__(**kwargs)
#         self.units = units
#         self.direction = direction

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'units': self.units,
#             'direction': self.direction,
#         })
#         return config

#     def call(self, inputs):
#         x = inputs
#         tf.print("X: ", x, output_stream=sys.stdout)
#         if self.direction == 'backward': x = tf.reverse(x, axis=[1])
#         tf.print("X: ", x, output_stream=sys.stdout)
#         return x

# def test_model():
    
#     x = np.arange(20).reshape(2, 5, 2)
#     print(x)

#     model = Sequential()
#     model.add(Input(shape=(5, 2)))
#     model.add(Dense(2, activation='linear', use_bias=False))
#     model.add(TestLayer(1, direction='backward'))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
#     print('')
#     model.summary()
#     print('')
    
#     model.predict(x)
    
#     return model


# def use_previous_model():
#     model = load_model(f'model/2020-10-29/19.57.31-MF-51.00880/best_model_max_val_accuracy.h5')
#     model.summary()


def test(min_timescale=1.0, max_timescale=1.0e4):
    import math
    import numpy as np

    length = 10
    channels = 4

    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    print(signal)

    bias_mask = np.tril(np.full([10, 10], True), 0)
    print(bias_mask)


if __name__ == "__main__":
    test()