''' Libraries '''
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import mir_eval

from chord_mapping import mapping_seventh
from convert import find_components, find_quality, find_shorthand
from my_mapping import mapping_quality2id
from btc import MyModel
from make_dir_and_plot import make_dir, plot_history


''' Parameters '''
# Data
SAMPLE_RATE = 44800
VALID_RATIO = 0.2
# Model
BATCH_LEN = 19
DIM = 192
N = 2
NUM_HEADS = 12
DROPOUT = 0.15
CONV_NUM = 2
# Train
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
TRAIN_BATCHES_LEN = 400
VALID_BATCHES_LEN = 100
INITIAL_LEARNING_RATE = 5e-2
DECAY_RATE = 0.9
EPOCH = 300
BATCH_SIZE = 256
CKPT_DIR = make_dir()


''' Functions '''
def loss_function(y_real, y_pred):
    y_real_root, y_real_quality = y_real[:, :13], y_real[:, 13:]
    y_pred_root, y_pred_quality = y_pred[:, :13], y_pred[:, 13:]
    root_loss = root_loss_object(y_real_root, y_pred_root)
    quality_loss = quality_loss_object(y_real_quality, y_pred_quality)
    return tf.reduce_sum(root_loss*quality_loss)


def accuracy_function(y_real, y_pred):
    accuracies = tf.equal(tf.cast(y_real, dtype=tf.int64), tf.argmax(y_pred, axis=-1))
    mask = tf.math.equal(y_real, 0)
    accuracies = tf.cast(mask, dtype=tf.uint8) + tf.cast(accuracies, dtype=tf.uint8)
    accuracies = tf.reduce_min(accuracies, axis=-1)
    return tf.reduce_sum(accuracies)/len(accuracies)


@tf.function
def train_step(x, y_real):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_function(y_real, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    # train_acc(accuracy_function(y_real, y_pred))
    return y_pred


def valid_step(x, y_real):
    y_pred = model(x)
    loss = loss_function(y_real, y_pred)
    valid_loss(loss)
    # valid_acc(accuracy_function(y_real, y_pred))
    return y_pred


def process_chords(chords):
    qualities = [ mir_eval.chord.split(chord)[1] for chord in chords ]
    for i, quality in enumerate(qualities):
        if quality in mapping_quality2id.keys(): qualities[i] = mapping_quality2id[quality]
        else: qualities[i] = mapping_quality2id['']
    qualities = np.eye(len(mapping_quality2id))[qualities].tolist()
    root_ids, quality_bitmaps, _ = mir_eval.chord.encode_many(chords)
    root_ids = [ id+1 for id in root_ids ]
    root_ids = np.eye(13)[root_ids].tolist()
    # return np.concatenate([root_ids, quality_bitmaps], axis=-1).tolist()
    return np.concatenate([root_ids, qualities], axis=-1).tolist()


def read_data(path, song_amount, hop_len):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{SAMPLE_RATE}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_chords(chords))
    return x_dataset, y_dataset


def main():

    x_dataset, y_dataset = read_data(r"../customized_data/CE200", 50, 4480)
    x_dataset_train = x_dataset[:int(len(x_dataset)*(1-VALID_RATIO))]
    y_dataset_train = y_dataset[:int(len(y_dataset)*(1-VALID_RATIO))]
    x_dataset_valid = x_dataset[int(len(x_dataset)*(1-VALID_RATIO)):]
    y_dataset_valid = y_dataset[int(len(y_dataset)*(1-VALID_RATIO)):]

    x_train, y_train = [], []
    x_valid, y_valid = [], []
    bar = tqdm(
        range(len(x_dataset_train)), desc="Processing train data",
        total=len(x_dataset_train), ascii=True
    )
    for i in bar:  # song num
        for j in range(len(x_dataset_train[i])-BATCH_LEN+1):
            x_train.append(x_dataset_train[i][j:j+BATCH_LEN])
            y_train.append(y_dataset_train[i][j:j+BATCH_LEN])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    bar = tqdm(
        range(len(x_dataset_valid)), desc="Processing valid data",
        total=len(x_dataset_valid), ascii=True
    )
    for i in bar:
        for j in range(len(x_dataset_valid[i])-BATCH_LEN+1):
            x_valid.append(x_dataset_valid[i][j:j+BATCH_LEN])
            y_valid.append(y_dataset_valid[i][j:j+BATCH_LEN])
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    global root_loss_object, quality_loss_object, optimizer
    root_loss_object = tf.keras.losses.CategoricalCrossentropy()
    quality_loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(  # lr_schedule
            initial_learning_rate=INITIAL_LEARNING_RATE,
            decay_steps=EPOCH,
            decay_rate=DECAY_RATE)
    )

    global train_loss, train_acc, valid_loss, valid_acc
    train_loss = tf.keras.metrics.Mean()
    # train_acc = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    # valid_acc = tf.keras.metrics.Mean()

    global model
    model = MyModel(BATCH_LEN, DIM, N, NUM_HEADS, DROPOUT, CONV_NUM)
    # model.compile(
    #     optimizer='adam', loss='mse', metrics=['accuracy'],
    #     # run_eagerly=True,
    # )
    # model.fit(
    #     x=x_train, y=y_train,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCH,
    #     verbose=1,
    #     callbacks=None,
    #     validation_data=(x_valid, y_valid),
    #     shuffle=True,
    # )


    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(EPOCH):

        train_batches = []
        random_train_idx = np.arange(len(x_train)-BATCH_SIZE)
        np.random.shuffle(random_train_idx)
        random_train_idx = random_train_idx[:TRAIN_BATCHES_LEN]
        pbar = tqdm(random_train_idx, total=len(random_train_idx), ascii=True,
                    desc=f"Sampling random train batches ({TRAIN_BATCHES_LEN/(len(x_train)-BATCH_SIZE)*100:.3f}%)")
        for i in pbar:
            train_batches.append((x_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]))

        valid_batches = []
        random_valid_idx = np.arange(len(x_valid)-BATCH_SIZE)
        np.random.shuffle(random_valid_idx)
        random_valid_idx = random_valid_idx[:VALID_BATCHES_LEN]
        pbar = tqdm(random_valid_idx, total=len(random_valid_idx), ascii=True,
                    desc=f"Sampling random valid batches ({VALID_BATCHES_LEN/(len(x_valid)-BATCH_SIZE)*100:.3f}%)")
        for i in pbar:
            valid_batches.append((x_valid[i:i+BATCH_SIZE], y_valid[i:i+BATCH_SIZE]))

        train_losses = []
        # train_accs  = []
        valid_losses = []
        # valid_accs  = []

        train_loss.reset_states()
        # train_acc.reset_states()
        valid_loss.reset_states()
        # valid_acc.reset_states()

        print('')
        pbar = tqdm(train_batches, desc=f"Epoch {epoch+1:3d}", total=len(train_batches), ascii=True)
        for (x_input, y_real) in pbar:
            y_pred = train_step(x_input, y_real)
            train_losses.append(train_loss.result())
            pbar.set_description(f"Epoch {epoch+1:3d}: train loss = {tf.math.reduce_mean(train_losses):.5f} | current learning rate: {optimizer._decayed_lr('float32').numpy():.8f} | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        avg_train_losses.append(np.mean(train_losses))

        os.system('cls')

        print('')
        pbar = tqdm(valid_batches, desc=f"Epoch {epoch+1:3d}", total=len(valid_batches), ascii=True)
        for (x_input, y_real) in pbar:
            y_pred = valid_step(x_input, y_real)
            valid_losses.append(valid_loss.result())
            pbar.set_description(f"Epoch {epoch+1:3d}: valid loss = {tf.math.reduce_mean(valid_losses):.5f} | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        avg_valid_losses.append(np.mean(valid_losses))

        np.set_printoptions(precision=6, linewidth=272, suppress=True)
        print('\ny_pred[0]:', y_pred[0].numpy().shape)
        print(y_pred[0].numpy())
        print('\nProcessed y_pred[0]:', y_pred[0].numpy().shape)
        print(np.concatenate([np.concatenate([np.eye(13)[np.argmax(y_pred[0, :, :13].numpy(), axis=-1)], np.eye(6)[np.argmax(y_pred[0, :, 13:].numpy(), axis=-1)]], axis=-1), ((np.arange(19)+1)/1000000)[np.newaxis, :]], axis=-0))
        print('\ny_real[0]:', y_real[0].shape)
        print(np.concatenate([y_real[0], ((np.arange(19)+1)/1000000)[np.newaxis, :]], axis=-0))
        print('\n')

        plot_history(CKPT_DIR, {
            'train_loss': avg_train_losses,
            'valid_loss': avg_valid_losses,
        })

    return


''' Exection '''
if __name__ == '__main__':
    main()