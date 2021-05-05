import mir_eval

# root_ids, quality_bitmaps, bass_ids = mir_eval.chord.encode_many([
#     'C:maj',
#     'C#:maj7',
#     'Db:7',
#     'E#:min',
#     'F:min7',
#     'N',
# ])
# print(root_ids)
# print(quality_bitmaps)
# print(bass_ids)

# print(mir_eval.chord.sevenths_inv(['N'], ['D:(1,3,5)']))


import tensorflow as tf

# x = tf.constant([[0.8, 0.4, 0.9], [1., 1., 1.]])
# y = tf.constant([0.5])
# print(tf.cast(tf.math.greater(x, y), dtype=tf.float32))

# y_pred = tf.constant([[1., 0., 0., 0.], [0., 1., 0., 0.], [0.1, 0.4, 0.5, 0.]])
# y_true = tf.constant([[1., 0., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]])

# print(tf.keras.losses.CategoricalCrossentropy(reduction='none')(y_true, y_pred))


from utils import plot_attns
import numpy as np

a = np.random.random((4, 64, 16, 100, 100))
plot_attns(r"test", a, a)