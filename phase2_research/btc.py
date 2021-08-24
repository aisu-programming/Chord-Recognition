''' Libraries '''
import math
from typing import Dict

import numpy as np
import tensorflow as tf


''' Function '''
class MyMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, qkv_dim, dim):
        super(MyMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv_dim   = qkv_dim
        assert qkv_dim % self.num_heads == 0
        self.depth = qkv_dim // self.num_heads
        self.W_q = tf.keras.layers.Dense(qkv_dim)
        self.W_k = tf.keras.layers.Dense(qkv_dim)
        self.W_v = tf.keras.layers.Dense(qkv_dim)
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
        return output, attention_weights

    def call(self, Q, K, V, attention_mask):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, attention_mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
            shape=(tf.shape(scaled_attention)[0], scaled_attention.shape[1], self.qkv_dim)
        )
        output = self.dense(concat_attention)
        return output, attention_weights


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, conv_num, conv_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.Convs = [ 
            tf.keras.layers.Conv1D(conv_dim, 3, padding='same')
            for _ in range(conv_num-1)
        ]
        self.Convs.append(tf.keras.layers.Conv1D(dim, 3, padding='same'))
        self.ReLU = tf.keras.layers.ReLU()
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        for conv in self.Convs:
            x = conv(x)
            x = self.ReLU(x)
            x = self.Dropout(x)
        return x


class MaskedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, batch_len, num_heads, qkv_dim, dim, dropout, conv_num, conv_dim):
        super(MaskedSelfAttention, self).__init__()
        self.batch_len = batch_len
        self.LayerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)
        # self.MHA = tf.keras.layers.MultiHeadAttention(num_heads, dim)
        self.MHA = MyMultiHeadAttention(num_heads, qkv_dim, dim)
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.LayerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)
        self.PFF = PositionwiseFeedForward(dim, conv_num, conv_dim, dropout)

    @property
    def mask(self):
        return tf.convert_to_tensor(np.triu(np.ones(self.batch_len), k=1), dtype=tf.float32)

    def call(self, x, training):
        x_tmp = self.LayerNorm1(x)
        x_tmp, attn = self.MHA(x_tmp, x_tmp, x_tmp, attention_mask=self.mask)
        x = self.Dropout(x_tmp, training=training)
        x_tmp = self.LayerNorm2(x)
        x += self.PFF(x_tmp)
        return x, attn


class BidirectionalMaskedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, batch_len, num_heads, qkv_dim, dim, dropout, conv_num, conv_dim):
        super(BidirectionalMaskedSelfAttention, self).__init__()
        self.forwardMaskedSelfAttention  = MaskedSelfAttention(batch_len, num_heads, qkv_dim, dim, dropout, conv_num, conv_dim)
        self.backwardMaskedSelfAttention = MaskedSelfAttention(batch_len, num_heads, qkv_dim, dim, dropout, conv_num, conv_dim)
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.Linear = tf.keras.layers.Dense(dim)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 0.001)

    def call(self, x, training):
        x_forward  = x
        x_backward = tf.reverse(x, axis=[-2])
        x_forward,  attn_forward  = self.forwardMaskedSelfAttention(x_forward, training=training)
        x_backward, attn_backward = self.backwardMaskedSelfAttention(x_backward, training=training)
        x_backward = tf.reverse(x_backward, axis=[-2])
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = self.Dropout(x, training=training)
        x = self.Linear(x)
        x = self.LayerNorm(x)
        return x, attn_forward, attn_backward


class MyModel(tf.keras.Model):
    def __init__(self, model_target, pred_mode, batch_len, dim, dropout, qkv_dim, N, num_heads, conv_num, conv_dim):
        super(MyModel, self).__init__()
        self.pred_mode = pred_mode
        self.batch_len = batch_len
        self.dim = dim
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.BidirectionalMaskedSelfAttentions = [
            BidirectionalMaskedSelfAttention(batch_len, num_heads, qkv_dim, dim, dropout, conv_num, conv_dim)
            for _ in range(N)
        ]

        if pred_mode == 'root':
            self.RootSoftmax = tf.keras.layers.Dense(13, activation=tf.keras.activations.softmax)
        elif pred_mode == 'quality_bitmap':
            raise NotImplementedError
            self.QualitySigmoid = tf.keras.layers.Dense(12, activation=tf.keras.activations.sigmoid)
        else:
            if model_target == 'majmin':
                if pred_mode == 'integrate':
                    self.ChordSoftmax = tf.keras.layers.Dense(26, activation=tf.keras.activations.softmax)
                elif pred_mode == 'quality':
                    self.QualitySoftmax = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)
                elif pred_mode == 'separate':
                    self.RootSoftmax = tf.keras.layers.Dense(13, activation=tf.keras.activations.softmax)
                    self.QualitySoftmax = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)
                else:
                    raise Exception
            elif model_target == 'seventh':
                if pred_mode == 'integrate':
                    self.ChordSoftmax = tf.keras.layers.Dense(62, activation=tf.keras.activations.softmax)
                elif pred_mode == 'quality':
                    self.QualitySoftmax = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)
                elif pred_mode == 'separate':
                    self.RootSoftmax = tf.keras.layers.Dense(13, activation=tf.keras.activations.softmax)
                    self.QualitySoftmax = tf.keras.layers.Dense(7, activation=tf.keras.activations.softmax)
                else:
                    raise Exception
            else:
                raise Exception

    @property
    def PositionalEncoding(self):
        posit = np.arange(self.batch_len)[:, np.newaxis]
        depth = np.arange(self.dim)[np.newaxis, :]
        angle = posit / np.power(10000, (2*(depth//2)/np.float32(self.dim)))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.convert_to_tensor(angle[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x, training):
        x = self.Dropout(x)
        x += self.PositionalEncoding
        attns_forward  = []
        attns_backward = []
        for BidirectionalMaskedSelfAttention in self.BidirectionalMaskedSelfAttentions:
            x, attn_forward, attn_backward = BidirectionalMaskedSelfAttention(x, training=training)
            attns_forward.append(attn_forward)
            attns_backward.append(attn_backward)

        if self.pred_mode == 'integrate':
            x = self.ChordSoftmax(x)
        elif self.pred_mode == 'root':
            x = self.RootSoftmax(x)
        elif self.pred_mode == 'quality':
            x = self.QualitySoftmax(x)
        elif self.pred_mode == 'quality_bitmap':
            raise NotImplementedError
            x = self.QualitySigmoid(x)
        elif self.pred_mode == 'separate':
            root = self.RootSoftmax(x)
            quality = self.QualitySoftmax(x)
            x = tf.concat([root, quality], axis=-1)
        else:
            raise Exception

        return x, attns_forward, attns_backward