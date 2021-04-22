''' Libraries '''
import math
from typing import Dict

import numpy as np
import tensorflow as tf


''' Function '''
class MyMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim):
        super(MyMultiHeadAttention, self).__init__()
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

    def scaled_dot_product_attention(self, Q, K, V, attention_mask):
        QK = tf.matmul(Q, K, transpose_b=True)
        dim_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        scaled_attention_logits = QK / tf.math.sqrt(dim_k)
        if attention_mask is not None: scaled_attention_logits += (attention_mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output

    def call(self, Q, K, V, attention_mask):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        scaled_attention = self.scaled_dot_product_attention(Q, K, V, attention_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
            shape=(tf.shape(scaled_attention)[0], scaled_attention.shape[1], self.dim)
        )
        output = self.dense(concat_attention)
        return output


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, conv_num, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.Convs = [ 
            tf.keras.layers.Conv1D(dim, 3, padding='same')
            for _ in range(conv_num)
        ]
        self.ReLU = tf.keras.layers.ReLU()
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        for conv in self.Convs:
            x = conv(x)
            x = self.ReLU(x)
            x = self.Dropout(x)
        return x


class MaskedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, batch_len, num_heads, dim, dropout, conv_num):
        super(MaskedSelfAttention, self).__init__()
        self.batch_len = batch_len
        self.LayerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)
        # self.MHA = tf.keras.layers.MultiHeadAttention(num_heads, dim)
        self.MHA = MyMultiHeadAttention(num_heads, dim)
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.LayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)
        self.PFF = PositionwiseFeedForward(dim, conv_num, dropout)

    @property
    def mask(self):
        return tf.convert_to_tensor(np.triu(np.ones(self.batch_len), k=1), dtype=tf.float32)

    def call(self, x, training):
        x_tmp = self.LayerNorm1(x)
        x += self.MHA(x_tmp, x_tmp, x_tmp, attention_mask=self.mask)
        x = self.Dropout(x, training=training)
        x_tmp = self.LayerNorm2(x)
        x += self.PFF(x_tmp)
        return x


class BidirectionalMaskedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, batch_len, num_heads, dim, dropout, conv_num):
        super(BidirectionalMaskedSelfAttention, self).__init__()
        self.forwardMaskedSelfAttention  = MaskedSelfAttention(batch_len, num_heads, dim, dropout, conv_num)
        self.backwardMaskedSelfAttention = MaskedSelfAttention(batch_len, num_heads, dim, dropout, conv_num)
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.Linear = tf.keras.layers.Dense(dim)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)

    def call(self, x, training):
        x_forward  = x
        x_backward = tf.reverse(x, axis=[-2])
        x_forward  = self.forwardMaskedSelfAttention(x_forward, training=training)
        x_backward = self.backwardMaskedSelfAttention(x_backward, training=training)
        x_backward = tf.reverse(x_backward, axis=[-2])
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = self.Dropout(x, training=training)
        x = self.Linear(x)
        x = self.LayerNorm(x)
        return x


class MyModel(tf.keras.Model):
    def __init__(self, batch_len, dim, N, num_heads, dropout, conv_num):
        super(MyModel, self).__init__()
        self.batch_len = batch_len
        self.dim = dim
        self.BidirectionalMaskedSelfAttentions = [
            BidirectionalMaskedSelfAttention(batch_len, num_heads, dim, dropout, conv_num)
            for _ in range(N)
        ]
        self.ChordSoftmax = tf.keras.layers.Dense(63, activation=tf.keras.activations.softmax)

    @property
    def PositionalEncoding(self):
        posit = np.arange(self.batch_len)[:, np.newaxis]
        depth = np.arange(self.dim)[np.newaxis, :]
        angle = posit / np.power(10000, (2*(depth//2)/np.float32(self.dim)))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.convert_to_tensor(angle[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x, training):
        x += self.PositionalEncoding
        for BidirectionalMaskedSelfAttention in self.BidirectionalMaskedSelfAttentions:
            x = BidirectionalMaskedSelfAttention(x, training=training)
        x = self.ChordSoftmax(x)
        return x