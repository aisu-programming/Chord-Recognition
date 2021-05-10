''' Libraries '''
import math
from typing import Dict

import numpy as np
import tensorflow as tf


''' Function '''
def PositionalEncoding(len, dim):
    posit = np.arange(len)[:, np.newaxis]                            # (len, dim)
    depth = np.arange(dim)[np.newaxis, :]                            # (  1, dim)
    angle = posit / np.power(10000, (2*(depth//2)/np.float32(dim)))  # (len, dim)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.convert_to_tensor(angle[np.newaxis, :, :], dtype=tf.float32)


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, units, frames_per_data, direction, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.frames_per_data = frames_per_data
        self.direction = direction


class ReLUs(tf.keras.Model):
    def __init__(self):
        super(BTC, self).__init__()
        # TODO: model architecture

        self.encoder = Encoder(
            embeddings=embeddings,
            dropout=dropout,
            num_heads=8,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.decoder = Decoder(
            dropout=dropout,
            num_heads=8,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.final_layers = [
            # tf.keras.layers.Dense(300),
            # tf.keras.layers.Dense(200),
            # tf.keras.layers.Dense(100),
            # tf.keras.layers.GRU(100, return_sequences=True),
            # tf.keras.layers.Dense(50),
            # tf.keras.layers.Dense(30), 
            tf.keras.layers.Dense(12),
            # tf.keras.layers.Dense(num_class, activation=tf.nn.softmax),
        ]

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def call(self, x, y, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(x, training, enc_padding_mask)
        x = self.decoder(y, enc_output, training, look_ahead_mask, dec_padding_mask)
        for final_layer in self.final_layers: x = final_layer(x)
        return x