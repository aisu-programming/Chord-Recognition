import os
import sys

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Reshape, Conv1D, ZeroPadding1D, Dropout, LayerNormalization, MultiHeadAttention, Concatenate, Dense

from mapping import my_mapping_dict
mapping_dictionary = my_mapping_dict

only_cqt = True
if only_cqt: input_dim = 12
else: input_dim = 24

class PositionwiseConvolution(Layer):
    def __init__(self, units, **kwargs):
        super(PositionwiseConvolution, self).__init__(**kwargs)
        self.units = units

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
        })
        return config

    def build(self, input_shape):
        self.layers = [
            Conv1D(filters=self.units, kernel_size=3, activation='relu'), 
            ZeroPadding1D(padding=(1, 1)), 
            Dropout(0.2), 
            Conv1D(filters=self.units, kernel_size=3), 
            ZeroPadding1D(padding=(1, 1)), 
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class SelfAttentionBlock(Layer):
    def __init__(self, units, frames_per_data, direction, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.frames_per_data = frames_per_data
        self.direction = direction

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'frames_per_data': self.frames_per_data,
            'direction': self.direction,
        })
        return config

    def build(self, input_shape):
        self.layer_normalization_1 = LayerNormalization() # layer_norm_mha
        self.multi_head_attention = MultiHeadAttention(num_heads=4, key_dim=12, dropout=0.2, attention_axes=None)
        self.dropout = Dropout(0.2)
        self.layer_normalization_2 = LayerNormalization() # layer_norm_ffn
        self.positionwise_convolution = PositionwiseConvolution(self.units)
    
    def call(self, inputs):
        x = inputs
        x_norm = self.layer_normalization_1(x)
        if self.direction == 'backward': x = tf.reverse(x, axis=[1])
        bias_mask = np.tril(np.full([self.frames_per_data, self.frames_per_data], True), 0)
        x += self.multi_head_attention(query=x_norm, value=x_norm, key=x_norm, attention_mask=bias_mask)
        x = self.dropout(x)
        x_norm = self.layer_normalization_2(x)
        x += self.positionwise_convolution(x_norm)
        x = self.dropout(x)
        return x

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = tf.convert_to_tensor(signal.reshape([1, length, channels]), dtype=tf.float32)
    return signal

class BiDirectionalSelfAttention(Layer):
    def __init__(self, units, frames_per_data, **kwargs):
        super(BiDirectionalSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.frames_per_data = frames_per_data
        # self.timing_signal = _gen_timing_signal(frames_per_data, units)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'frames_per_data': self.frames_per_data,
            # 'timing_signal': self.timing_signal,
        })
        return config

    def build(self, input_shape):
        self.linear_1 = Dense(self.units, activation='linear', use_bias=False) # embedding_proj
        self.forward_self_attention_block = SelfAttentionBlock(self.units, self.frames_per_data, 'forward')
        self.backward_self_attention_block = SelfAttentionBlock(self.units, self.frames_per_data, 'backward')
        self.linear_2 = Dense(self.units, activation='linear') # 256 -> 128
    
    def call(self, inputs):
        x = inputs
        x = self.linear_1(x)
        # x += tf.cast(self.timing_signal[:, :inputs.shape[1], :], inputs.dtype)
        x += _gen_timing_signal(self.frames_per_data, self.units)[:, :inputs.shape[1], :]
        for _ in range(8):
            x_forward = self.forward_self_attention_block(x)
            x_backward = self.backward_self_attention_block(x)
            x = Concatenate()([x_forward, x_backward])
            x = self.linear_2(x)
        return x

def customized_loss(y_true, y_pred):
    return tf.where(tf.equal(y_true, 1), tf.math.multiply(tf.math.log(y_pred), -100), 0) 

def AttentionModel(frames_per_data):
    model = Sequential()
    model.add(Input(shape=(frames_per_data, input_dim)))
    model.add(Dropout(0.2))
    model.add(BiDirectionalSelfAttention(128, frames_per_data))
    model.add(LayerNormalization())
    model.add(Dense(len(mapping_dictionary), activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.compile(loss=customized_loss, optimizer=opt, metrics=["accuracy"])
    print('')
    model.summary()
    print('')
    return model

if __name__ == "__main__":
    AttentionModel(frames_per_data=11)