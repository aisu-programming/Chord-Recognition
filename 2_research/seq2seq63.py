''' Libraries '''
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from process_chord import process_chords_v1, process_chords_v2
from btc63 import MyModel
from make_dir_and_plot import make_dir, plot_history, plot_loss_lr


''' Parameters '''
# Data
SAMPLE_RATE = 44800
VALID_RATIO = 0.2
# Model
BATCH_LEN = 25
DIM = 192
N = 1
NUM_HEADS = 4
DROPOUT = 0.2
CONV_NUM = 2
# Train
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
TRAIN_BATCHES_LEN = 800
VALID_BATCHES_LEN = 200
INITIAL_LEARNING_RATE = 1e-2
DECAY_RATE = 0.963
# INITIAL_LEARNING_RATE = 1e-3
# DECAY_RATE = 1.02
EPOCH = 300
BATCH_SIZE = 256
CKPT_DIR = make_dir()


''' Functions '''
def loss_function(y_real, y_pred):
    losses = loss_object(y_real, y_pred)
    return tf.reduce_sum(losses)
    # return tf.reduce_sum(losses)/len(losses)


def accuracy_function(y_real, y_pred):
    y_real_ids = tf.argmax(y_real, axis=-1)
    y_pred_ids = tf.argmax(y_pred, axis=-1)
    accs = tf.cast(tf.equal(y_real_ids, y_pred_ids), dtype=tf.float32)

    # tf.print("")
    # tf.print("y_real_root_ids[0]   : ", y_real_root_ids[0], summarize=-1)
    # tf.print("y_pred_root_ids[0]   : ", y_pred_root_ids[0], summarize=-1)
    # tf.print("root_accs[0]         : ", root_accs[0], summarize=-1)
    # tf.print("")
    # tf.print("y_real_quality_ids[0]: ", y_real_quality_ids[0], summarize=-1)
    # tf.print("y_pred_quality_ids[0]: ", y_pred_quality_ids[0], summarize=-1)
    # tf.print("quality_accs[0]      : ", quality_accs[0], summarize=-1)
    # tf.print("")
    # tf.print("accs[0]              : ", accs[0], summarize=-1)
    # tf.print("")

    return tf.reduce_sum(accs)/len(accs)


@tf.function
def train_step(x, y_real):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_function(y_real, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(accuracy_function(y_real, y_pred))
    return y_pred


def valid_step(x, y_real):
    y_pred = model(x)
    valid_loss(loss_function(y_real, y_pred))
    valid_acc(accuracy_function(y_real, y_pred))
    return y_pred


def save_details():
    with open(f"{CKPT_DIR}/details.txt", mode='w') as f:
        f.write(f"# Program name\n")
        f.write(f"{__file__}.py\n\n")

        f.write(f"# Data\n")
        f.write(f"SAMPLE_RATE: {SAMPLE_RATE}\n")
        f.write(f"VALID_RATIO: {VALID_RATIO}\n\n")

        f.write(f"# Model\n")
        f.write(f"BATCH_LEN: {BATCH_LEN}\n")
        f.write(f"DIM: {DIM}\n")
        f.write(f"N: {N}\n")
        f.write(f"NUM_HEADS: {NUM_HEADS}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"CONV_NUM: {CONV_NUM}\n\n")

        f.write(f"# Train\n")
        f.write(f"RANDOM_SEED: {RANDOM_SEED}\n")
        f.write(f"TRAIN_BATCHES_LEN: {TRAIN_BATCHES_LEN}\n")
        f.write(f"VALID_BATCHES_LEN: {VALID_BATCHES_LEN}\n")
        f.write(f"INITIAL_LEARNING_RATE: {INITIAL_LEARNING_RATE}\n")
        f.write(f"DECAY_RATE: {DECAY_RATE}\n")
        f.write(f"EPOCH: {EPOCH}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")


def read_data(path, song_amount, hop_len):
    x_dataset, y_dataset = [], []
    for i in tqdm(range(song_amount), desc="Reading data", total=song_amount, ascii=True):
        try: data = pd.read_csv(f"{path}/{i+1}/data_{SAMPLE_RATE}_{hop_len}.csv", index_col=0).values
        except: continue
        cqts   = data[:, :-1].tolist()
        chords = [ chord.strip() for chord in data[:, -1].tolist() ]
        x_dataset.append(cqts)
        y_dataset.append(process_chords_v2(chords))
    return x_dataset, y_dataset


def main():

    save_details()

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

    global loss_object, optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    optimizer = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(  # lr_schedule
            initial_learning_rate=INITIAL_LEARNING_RATE,
            decay_steps=EPOCH,
            decay_rate=DECAY_RATE)
    )

    global train_loss, train_acc, valid_loss, valid_acc
    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    valid_acc = tf.keras.metrics.Mean()

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
    avg_train_accs = []
    avg_valid_losses = []
    avg_valid_accs = []
    for epoch in range(EPOCH):

        train_batches = []
        random_train_idx = np.arange(len(x_train)-BATCH_SIZE)
        np.random.shuffle(random_train_idx)
        random_train_idx = random_train_idx[:TRAIN_BATCHES_LEN]
        pbar = tqdm(random_train_idx, total=len(random_train_idx), ascii=True,
                    desc=f"Sampling random train batches ({TRAIN_BATCHES_LEN}/{len(x_train)-BATCH_SIZE}={TRAIN_BATCHES_LEN/(len(x_train)-BATCH_SIZE)*100:.3f}%)")
        for i in pbar:
            train_batches.append((x_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]))

        valid_batches = []
        random_valid_idx = np.arange(len(x_valid)-BATCH_SIZE)
        np.random.shuffle(random_valid_idx)
        random_valid_idx = random_valid_idx[:VALID_BATCHES_LEN]
        pbar = tqdm(random_valid_idx, total=len(random_valid_idx), ascii=True,
                    desc=f"Sampling random valid batches ({VALID_BATCHES_LEN}/{len(x_valid)-BATCH_SIZE}={VALID_BATCHES_LEN/(len(x_valid)-BATCH_SIZE)*100:.3f}%)")
        for i in pbar:
            valid_batches.append((x_valid[i:i+BATCH_SIZE], y_valid[i:i+BATCH_SIZE]))

        train_losses = []
        train_accs  = []
        valid_losses = []
        valid_accs  = []

        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()


        # Test
        testing_losses = []
        lrs = []


        print('')
        pbar = tqdm(enumerate(train_batches), desc=f"Epoch {epoch+1:3d}", total=len(train_batches), ascii=True)
        for (i, (x_input, y_real)) in pbar:
            y_pred = train_step(x_input, y_real)
            train_losses.append(train_loss.result())
            train_accs.append(train_acc.result())
            pbar.set_description(f"Epoch {epoch+1:3d}: train loss = {tf.math.reduce_mean(train_losses):.5f}, train accuracy = {tf.math.reduce_mean(train_accs):.3f}% | current learning rate: {optimizer._decayed_lr('float32').numpy():.8f} | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
            # if i % 50 == 0:
            #     testing_losses.append(train_loss.result())
            #     lrs.append(optimizer._decayed_lr('float32').numpy())
            #     plot_loss_lr(CKPT_DIR, {
            #         'loss': testing_losses,
            #         'lr': lrs,
            #     })
        avg_train_losses.append(np.mean(train_losses))
        avg_train_accs.append(np.mean(train_accs))

        os.system('cls')

        print('')
        pbar = tqdm(valid_batches, desc=f"Epoch {epoch+1:3d}", total=len(valid_batches), ascii=True)
        for (x_input, y_real) in pbar:
            y_pred = valid_step(x_input, y_real)
            valid_losses.append(valid_loss.result())
            valid_accs.append(valid_acc.result())
            pbar.set_description(f"Epoch {epoch+1:3d}: valid loss = {tf.math.reduce_mean(valid_losses):.5f}, valid accuracy = {tf.math.reduce_mean(valid_accs):.3f}% | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        avg_valid_losses.append(np.mean(valid_losses))
        avg_valid_accs.append(np.mean(valid_accs))

        np.set_printoptions(precision=5, edgeitems=9, linewidth=272, suppress=True)
        print("\nSample prediction: Shape =", y_pred[0].numpy().shape)
        print(y_pred[0].numpy())
        print("\nProcessed sample prediction: Shape =", y_pred[0].numpy().shape)
        print(np.concatenate([np.eye(63)[np.argmax(y_pred[0].numpy(), axis=-1)], ((np.arange(63)+1)/100000)[np.newaxis, :]], axis=-0))
        print("\nGround truth: Shape =", y_real[0].shape)
        print(np.concatenate([y_real[0], ((np.arange(63)+1)/100000)[np.newaxis, :]], axis=-0))
        print('\n')

        plot_history(CKPT_DIR, {
            'train_loss': avg_train_losses,
            'train_acc': avg_train_accs,
            'valid_loss': avg_valid_losses,
            'valid_acc': avg_valid_accs,
        })

    return


''' Exection '''
if __name__ == '__main__':
    main()