''' Libraries '''
import math
from typing import Dict

import numpy as np
import tensorflow as tf


''' Function '''
def PositionalEncoding(dim):
    text_len = 40  # 37
    posit = np.arange(text_len)[:, np.newaxis]                       # (28, 1)
    depth = np.arange(dim)[np.newaxis, :]                            # (1, 300)
    angle = posit / np.power(10000, (2*(depth//2)/np.float32(dim)))  # (28, 300)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.convert_to_tensor(angle[np.newaxis, :, :], dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads  = num_heads
        self.dim = dim

        assert dim % self.num_heads == 0
        self.depth = dim // self.num_heads

        self.W_q = tf.keras.layers.Dense(dim)
        self.W_k = tf.keras.layers.Dense(dim)
        self.W_v = tf.keras.layers.Dense(dim)

        self.dense = tf.keras.layers.Dense(dim)

    def split_heads(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], x.shape[1], self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def scaled_dot_product_attention(self, Q, K, V, mask):
        QK = tf.matmul(Q, K, transpose_b=True)
        dim_k = tf.cast(tf.shape(K)[-1], tf.float32)

        scaled_attention_logits = QK / tf.math.sqrt(dim_k)
        if mask is not None: scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)

        return output

    def call(self, Q, K, V, mask):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scaled_attention = self.scaled_dot_product_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
            shape=(tf.shape(scaled_attention)[0], scaled_attention.shape[1], self.dim)
        )
        output = self.dense(concat_attention)

        return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, dropout, hidden_size):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dim, num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(dropout)
        self.mha_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Pointwise Feed Forward network
        self.pff = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(dim),
        ])
        self.pff_dropout = tf.keras.layers.Dropout(dropout)
        self.pff_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training, mask):
        x_tmp = self.mha(x, x, x, mask)
        x += self.mha_dropout(x_tmp, training=training)
        x = self.mha_layer_norm(x)

        x_tmp = self.pff(x)
        x += self.pff_dropout(x_tmp, training=training)
        x = self.pff_layer_norm(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, embeddings, dropout, num_heads, hidden_size, num_layers):
        super(Encoder, self).__init__()
        # self.embeddings = tf.keras.layers.Embedding(
        #     input_dim=4119,
        #     output_dim=300,
        #     embeddings_initializer=tf.constant_initializer(embeddings.numpy())
        # )
        self.embeddings = tf.keras.layers.Embedding(
            input_dim=4119,
            output_dim=24,
        )
        self.pos_enc = PositionalEncoding(dim=24)
        self.encoder_layers = [ EncoderLayer(
            dim=24,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout
        ) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, training, mask):

        x = self.embeddings(x)
        x *= tf.math.sqrt(tf.cast(24, tf.float32))

        x += self.pos_enc[:tf.shape(x)[0], :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training, mask)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, dropout, hidden_size):
        super(DecoderLayer, self).__init__()

        self.mha_1 = MultiHeadAttention(dim, num_heads)
        self.mha_1_dropout = tf.keras.layers.Dropout(dropout)
        self.mha_1_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha_2 = MultiHeadAttention(dim, num_heads)
        self.mha_2_dropout = tf.keras.layers.Dropout(dropout)
        self.mha_2_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.pff = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(dim),
        ])
        self.pff_dropout = tf.keras.layers.Dropout(dropout)
        self.pff_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, y, enc_output, training, look_ahead_mask, padding_mask):
        y_tmp = self.mha_1(y, y, y, look_ahead_mask)
        y += self.mha_1_dropout(y_tmp, training=training)
        y = self.mha_1_layer_norm(y)

        y_tmp = self.mha_2(y, enc_output, enc_output, padding_mask)
        y += self.mha_2_dropout(y_tmp, training=training)
        y = self.mha_2_layer_norm(y)

        y_tmp = self.pff(y)
        y += self.pff_dropout(y_tmp, training=training)
        y = self.pff_layer_norm(y)

        return y


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dropout, num_heads, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embeddings = tf.keras.layers.Embedding(
            input_dim=12,
            output_dim=24,
        )
        self.pos_enc = PositionalEncoding(dim=24)
        self.decoder_layers = [ DecoderLayer(
            dim=24,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout
        ) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, y, enc_output, training, look_ahead_mask, padding_mask):

        y = self.embeddings(y)
        y *= tf.math.sqrt(tf.cast(24, tf.float32))
        
        # y = tf.one_hot(tf.cast(y, dtype=tf.uint8), depth=12)
        # y *= tf.math.sqrt(tf.cast(300, tf.float32))

        y += self.pos_enc[:tf.shape(y)[0], :tf.shape(y)[1], :]
        y = self.dropout(y, training=training)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, enc_output, training, look_ahead_mask, padding_mask)
        return y


class SeqClassifier(tf.keras.Model):
    def __init__(
        self,
        embeddings: tf.Tensor,
        dropout: float,
        hidden_size: int,
        num_layers: int,
        num_class: int,
    ):
        super(SeqClassifier, self).__init__()
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