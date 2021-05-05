''' Libraries '''
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from btc import MyModel
from utils import make_dir, plot_history, plot_loss_lr, plot_attns



''' Data parameters '''
LOAD_SONG_AMOUNT = 200
SAMPLE_RATE = 44800
HOP_LENGTH = 4480
VALID_RATIO = 0.2

''' Model parameters '''
MODEL_MODE = 'seq2seq'
# MODEL_MODE = 'seq2one'
OUTPUT_MODE = '63'
# OUTPUT_MODE = '13+6'
BATCH_LEN = 101
DIM = 192
QKV_DIM = 128
N = 8
NUM_HEADS = 4
DROPOUT = 0.2
CONV_NUM = 2
CONV_DIM = 128

''' Preprocess parameters '''
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
DATASET_HOP = 70
TRAIN_BATCH_LEN = 2000
VALID_BATCH_LEN = None

''' Training parameters '''
INITIAL_LR = 1e-4
WARMUP_STEPS = 10000
DECAY_STEPS = 2000
DECAY_RATE = 0.99
EPOCH = 300
BATCH_SIZE = 128
CKPT_DIR = make_dir()



''' Model library '''
if   MODEL_MODE == 'seq2seq' and OUTPUT_MODE == '63'  : from seq2seq_63 import loss_function, accuracy_function, read_data, show_pred_and_truth
elif MODEL_MODE == 'seq2seq' and OUTPUT_MODE == '13+6': from seq2seq_16_3 import loss_function, accuracy_function, read_data, show_pred_and_truth
elif MODEL_MODE == 'seq2one' and OUTPUT_MODE == '63'  : from seq2one_63 import loss_function, accuracy_function, read_data, show_pred_and_truth
elif MODEL_MODE == 'seq2one' and OUTPUT_MODE == '13+6': from seq2one_16_3 import loss_function, accuracy_function, read_data, show_pred_and_truth



''' Function '''
# Save the parameters into "details.txt"
def save_details():
    with open(f"{CKPT_DIR}/details.txt", mode='w') as f:
        f.write(f"# Data parameters\n")
        f.write(f"LOAD_SONG_AMOUNT: {LOAD_SONG_AMOUNT}\n")
        f.write(f"SAMPLE_RATE     : {SAMPLE_RATE}\n")
        f.write(f"HOP_LENGTH      : {HOP_LENGTH}\n")
        f.write(f"VALID_RATIO     : {VALID_RATIO}\n\n")

        f.write(f"# Model parameters\n")
        f.write(f"MODEL_MODE : {MODEL_MODE}\n")
        f.write(f"OUTPUT_MODE: {OUTPUT_MODE}\n")
        f.write(f"BATCH_LEN  : {BATCH_LEN}\n")
        f.write(f"DIM        : {DIM}\n")
        f.write(f"QKV_DIM    : {QKV_DIM}\n")
        f.write(f"N          : {N}\n")
        f.write(f"NUM_HEADS  : {NUM_HEADS}\n")
        f.write(f"DROPOUT    : {DROPOUT}\n")
        f.write(f"CONV_NUM   : {CONV_NUM}\n")
        f.write(f"CONV_DIM   : {CONV_DIM}\n\n")

        f.write(f"# Training parameters\n")
        f.write(f"RANDOM_SEED    : {RANDOM_SEED}\n")
        f.write(f"DATASET_HOP    : {DATASET_HOP}\n")
        f.write(f"TRAIN_BATCH_LEN: {TRAIN_BATCH_LEN}\n")
        f.write(f"VALID_BATCH_LEN: {VALID_BATCH_LEN}\n")
        f.write(f"INITIAL_LR     : {INITIAL_LR}\n")
        f.write(f"WARMUP_STEPS   : {WARMUP_STEPS}\n")
        f.write(f"DECAY_STEPS    : {DECAY_STEPS}\n")
        f.write(f"DECAY_RATE     : {DECAY_RATE}\n")
        f.write(f"EPOCH          : {EPOCH}\n")
        f.write(f"BATCH_SIZE     : {BATCH_SIZE}\n")


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate, min_lr):
        super(CustomSchedule, self).__init__()
        self.ilr = initial_lr
        self.ws  = warmup_steps
        self.ds  = decay_steps
        self.dr  = decay_rate
        self.mlr = min_lr
    def __call__(self, step):
        arg1 = step / self.ws
        arg2 = self.dr ** ((step - self.ws) / self.ds)
        return tf.math.maximum(self.ilr * tf.math.minimum(arg1, arg2), self.mlr)


# Customized training step
@tf.function
def train_step(x, y_real):
    with tf.GradientTape() as tape:
        y_pred, attns_forward, attns_backward = model(x, training=True)
        loss = loss_function(y_real, y_pred, loss_objects, BATCH_LEN//2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(accuracy_function(y_real, y_pred, BATCH_LEN//2))
    return y_pred, attns_forward, attns_backward


# Customized validating step
def valid_step(x, y_real):
    y_pred, attns_forward, attns_backward = model(x)
    valid_loss(loss_function(y_real, y_pred, loss_objects, BATCH_LEN//2))
    valid_acc(accuracy_function(y_real, y_pred, BATCH_LEN//2))
    return y_pred


# Main process
def main():


    # Save parameters first for records
    save_details()


    # Load dataset
    x_dataset, y_dataset = read_data(
        r"../customized_data/CE200",
        LOAD_SONG_AMOUNT, SAMPLE_RATE, HOP_LENGTH
    )
    # Split loaded datasets into training and validating purpose
    x_dataset_train = x_dataset[:int(len(x_dataset)*(1-VALID_RATIO))]
    y_dataset_train = y_dataset[:int(len(y_dataset)*(1-VALID_RATIO))]
    x_dataset_valid = x_dataset[int(len(x_dataset)*(1-VALID_RATIO)):]
    y_dataset_valid = y_dataset[int(len(y_dataset)*(1-VALID_RATIO)):]


    # Process datasets to data with input format
    x_train, y_train = [], []
    x_valid, y_valid = [], []

    progress_bar = tqdm(
        range(len(x_dataset_train)), desc="Processing train data",
        total=len(x_dataset_train), ascii=True
    )
    # 'i' for song no.
    for i in progress_bar:
        # 'j' for frame no. in a song
        for j in range(0, len(x_dataset_train[i])-BATCH_LEN+1, DATASET_HOP):
            x_train.append(x_dataset_train[i][j:j+BATCH_LEN])
            y_train.append(y_dataset_train[i][j:j+BATCH_LEN])

    progress_bar = tqdm(
        range(len(x_dataset_valid)), desc="Processing valid data",
        total=len(x_dataset_valid), ascii=True
    )
    for i in progress_bar:
        for j in range(0, len(x_dataset_valid[i])-BATCH_LEN+1, DATASET_HOP):
            x_valid.append(x_dataset_valid[i][j:j+BATCH_LEN])
            y_valid.append(y_dataset_valid[i][j:j+BATCH_LEN])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)


    # Global variable 'loss_objects' for both training and validating step
    # Global variable 'optimizer', 'train_loss', and 'train_acc' for training step
    # Global variable 'valid_loss', and 'valid_acc' for validating step
    global loss_objects, optimizer, train_loss, train_acc, valid_loss, valid_acc
    loss_objects = [
        tf.keras.losses.CategoricalCrossentropy(reduction='none'),
        tf.keras.losses.CategoricalCrossentropy(reduction='none'),
    ]
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(  # lr_schedule
        initial_lr=INITIAL_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
    ))
    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    valid_acc = tf.keras.metrics.Mean()


    # Define the model
    global model
    model = MyModel(OUTPUT_MODE, BATCH_LEN, DIM, QKV_DIM, N, NUM_HEADS, DROPOUT, CONV_NUM, CONV_DIM)


    # Average losses, accuracies and learning rate per epoch for plotting figures
    avg_train_losses = []
    avg_train_accs = []
    avg_valid_losses = []
    avg_valid_accs = []
    learning_rate = []
    for epoch in range(EPOCH):


        # According to limited memory space (RAM),
        #   train the model based on little data each epoch.
        train_batches = []
        random_train_idx = np.arange(len(x_train)-BATCH_SIZE)
        np.random.shuffle(random_train_idx)
        global TRAIN_BATCH_LEN
        if TRAIN_BATCH_LEN is None: TRAIN_BATCH_LEN = len(random_train_idx)
        else: random_train_idx = random_train_idx[:TRAIN_BATCH_LEN]
        progress_bar = tqdm(
            random_train_idx, total=len(random_train_idx), ascii=True,
            desc=f"Sampling random train batches " +
                 f"({TRAIN_BATCH_LEN}/{len(x_train)-BATCH_SIZE}={TRAIN_BATCH_LEN/(len(x_train)-BATCH_SIZE)*100:.3f}%)"
        )
        for i in progress_bar:
            train_batches.append((x_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]))

        valid_batches = []
        random_valid_idx = np.arange(len(x_valid)-BATCH_SIZE)
        np.random.shuffle(random_valid_idx)
        global VALID_BATCH_LEN
        if VALID_BATCH_LEN is None: VALID_BATCH_LEN = len(random_valid_idx)
        else: random_valid_idx = random_valid_idx[:VALID_BATCH_LEN]
        progress_bar = tqdm(
            random_valid_idx, total=len(random_valid_idx), ascii=True,
            desc=f"Sampling random valid batches " +
                 f"({VALID_BATCH_LEN}/{len(x_valid)-BATCH_SIZE}={VALID_BATCH_LEN/(len(x_valid)-BATCH_SIZE)*100:.3f}%)"
        )
        for i in progress_bar:
            valid_batches.append((x_valid[i:i+BATCH_SIZE], y_valid[i:i+BATCH_SIZE]))


        # Record every losses and accuracies per batch in one epoch
        train_losses = []
        train_accs  = []
        valid_losses = []
        valid_accs  = []

        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()


        # Training
        print('')
        progress_bar = tqdm(enumerate(train_batches), desc=f"Epoch {epoch+1:3d}", total=len(train_batches), ascii=True)
        for (i, (x_input, y_real)) in progress_bar:
            y_pred, attns_forward, attns_backward = train_step(x_input, y_real)
            train_losses.append(train_loss.result())
            train_accs.append(train_acc.result())
            progress_bar.set_description(
                f"Epoch {epoch+1:3d}: " +
                f"train loss = {np.mean(train_losses):.5f}, train accuracy = {np.mean(train_accs):.3f}% | " +
                f"current learning rate: {optimizer._decayed_lr('float32').numpy():.8f} | " +
                f"BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")

        avg_train_losses.append(np.mean(train_losses))
        avg_train_accs.append(np.mean(train_accs))
        learning_rate.append(optimizer._decayed_lr('float32').numpy())

        os.system('cls')

        # Validating
        print('')
        progress_bar = tqdm(valid_batches, desc=f"Epoch {epoch+1:3d}", total=len(valid_batches), ascii=True)
        for (x_input, y_real) in progress_bar:
            y_pred = valid_step(x_input, y_real)
            valid_losses.append(valid_loss.result())
            valid_accs.append(valid_acc.result())
            progress_bar.set_description(
                f"Epoch {epoch+1:3d}: " +
                f"valid loss = {np.mean(valid_losses):.5f}, valid accuracy = {np.mean(valid_accs):.3f}% | " +
                f"BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        avg_valid_losses.append(np.mean(valid_losses))
        avg_valid_accs.append(np.mean(valid_accs))

        show_pred_and_truth(y_real, y_pred, BATCH_LEN//2)

        plot_history(CKPT_DIR, {
            'train_loss': avg_train_losses,
            'train_acc' : avg_train_accs,
            'valid_loss': avg_valid_losses,
            'valid_acc' : avg_valid_accs,
            'lr'        : learning_rate,
        })

        plot_attns(CKPT_DIR, attns_forward, attns_backward)

    return



''' Exection '''
if __name__ == '__main__':
    main()