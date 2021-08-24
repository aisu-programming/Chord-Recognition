targets = [
    'A:(1,5)',
    'A:(1,5)/5',
    'A:7',
    'A:7(#11)',
    'A:7(*3)',
    'A:7(*3,4)',
    'A:7(*3,4,13)',
    'A:7(*3,4,13)/b2',
    'A:7(11,13)',
    'A:7(13)',
    'A:7(2,*3,13)',
    'A:7(b13)',
    'A:7(b13)/3',
    'A:7(b9)',
    'A:7/3',
    'A:9(*3,4)',
    'A:aug/b7',
    'A:dim',
    'A:hdim7',
    'A:maj',
    'A:maj(11)',
    'A:maj(2)',
    'A:maj(6)',
    'A:maj(9)',
    'A:maj(9)/3',
    'A:maj/2',
    'A:maj/3',
    'A:maj/5',
    'A:maj/7',
    'A:maj/b7',
    'A:maj6',
    'A:maj7',
    'A:maj7/3',
    'A:maj7/5',
    'A:maj9',
    'A:maj9(13)',
    'A:min',
    'A:min(9)',
    'A:min/3',
    'A:min/5',
    'A:min/b7',
    'A:min6',
    'A:min6/3',
    'A:min6/5',
    'A:min7',
    'A:min7/5',
    'A:min7/b7',
    'A:min9',
    'A:min9(11)',
    'A:sus2',
    'B:sus4',
]


# import mir_eval
# score_A         = mir_eval.chord.sevenths(targets, ['A'] * len(targets))
# score_A_inv     = mir_eval.chord.sevenths_inv(targets, ['A'] * len(targets))
# score_Amaj      = mir_eval.chord.sevenths(targets, ['A:maj'] * len(targets))
# score_Amaj_inv  = mir_eval.chord.sevenths_inv(targets, ['A:maj'] * len(targets))
# score_Amin      = mir_eval.chord.sevenths(targets, ['A:min'] * len(targets))
# score_Amin_inv  = mir_eval.chord.sevenths_inv(targets, ['A:min'] * len(targets))
# score_A7        = mir_eval.chord.sevenths(targets, ['A:7'] * len(targets))
# score_A7_inv    = mir_eval.chord.sevenths_inv(targets, ['A:7'] * len(targets))
# score_Amaj7     = mir_eval.chord.sevenths(targets, ['A:maj7'] * len(targets))
# score_Amaj7_inv = mir_eval.chord.sevenths_inv(targets, ['A:maj7'] * len(targets))
# score_Amin7     = mir_eval.chord.sevenths(targets, ['A:min7'] * len(targets))
# score_Amin7_inv = mir_eval.chord.sevenths_inv(targets, ['A:min7'] * len(targets))
# print(f"{'Label':15} |   A   | A maj | A min |  A 7  | Amaj7 | Amin7 |")
# for i in range(len(targets)):
#     print(f"{targets[i]:15} | " +
#           f"{int(score_A[i]):2d} {int(score_A_inv[i]):2d} | " +
#           f"{int(score_Amaj[i]):2d} {int(score_Amaj_inv[i]):2d} | " +
#           f"{int(score_Amin[i]):2d} {int(score_Amin_inv[i]):2d} | " +
#           f"{int(score_A7[i]):2d} {int(score_A7_inv[i]):2d} | " +
#           f"{int(score_Amaj7[i]):2d} {int(score_Amaj7_inv[i]):2d} | " +
#           f"{int(score_Amin7[i]):2d} {int(score_Amin7_inv[i]):2d} |")


import mir_eval
# root_ids, quality_bitmaps, bass_ids = mir_eval.chord.encode_many(targets)
# outputs = zip(targets, root_ids, quality_bitmaps, bass_ids)
# print(f"{'Label':15} | root | {'qualtiy bitmap':25} | {'absolute qualtiy bitmap':25} | bass |")
# for output in outputs: print(f"{output[0]:15} | {output[1]:4} | {output[2]} | {mir_eval.chord.rotate_bitmap_to_root(output[2], output[1])} | {output[3]:4} |")


print(mir_eval.chord.encode('X'))


# import mir_eval
# print(mir_eval.chord.sevenths(['A:7(*3)'], ['A:7']))


# import tensorflow as tf
# x = tf.constant([[0.8, 0.4, 0.9], [1., 1., 1.]])
# y = tf.constant([0.5])
# print(tf.cast(tf.math.greater(x, y), dtype=tf.float32))
# y_pred = tf.constant([[1., 0., 0., 0.], [0., 1., 0., 0.], [0.1, 0.4, 0.5, 0.]])
# y_true = tf.constant([[1., 0., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]])
# print(tf.keras.losses.CategoricalCrossentropy(reduction='none')(y_true, y_pred))


# from utils import plot_attns
# import numpy as np
# a = np.random.random((4, 64, 16, 100, 100))
# plot_attns(r"test", a, a)


# from seq2seq_63 import all_accuracy_function, mid_accuracy_function
# import numpy as np
# real = np.random.random((4, 5, 2))
# pred = np.random.random((4, 5, 2))
# mid_accuracy_function(real, pred, 5//2)