''' Libraries '''
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from btc import MyModel
from plot import plot_history, plot_loss_lr, plot_attns
from utils import make_dir, loss_function, accuracy_function, read_data, show_pred_and_truth

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



''' Data parameters '''
LOAD_SONG_AMOUNT = 200
SAMPLE_RATE = 44800
HOP_LENGTH = 4480
VALID_RATIO = 0.6

''' Model parameters '''
MODEL_TARGET = 'seventh'
# MODEL_TARGET = 'majmin'
MODEL_MODE = 'seq2seq'
# MODEL_MODE = 'seq2one'
PRED_MODE = 'integrate'
# PRED_MODE = 'separate'
# PRED_MODE = 'root'
# PRED_MODE = 'quality'
# PRED_MODE = 'quality_bitmap'
BATCH_LEN = 51
N = 4
DIM = 192
NUM_HEADS = 32
QKV_DIM = NUM_HEADS * 64
DROPOUT = 0.3
CONV_NUM = 2
CONV_DIM = DIM * 1.5  # 128

''' Preprocess parameters '''
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
DATASET_HOP = BATCH_LEN
TRAIN_BATCH_LEN = 1000
VALID_BATCH_LEN = 3000

''' Training parameters '''
INITIAL_LR = 2e-4
WARMUP_STEPS = TRAIN_BATCH_LEN
DECAY_STEPS = TRAIN_BATCH_LEN
DECAY_RATE = 0.88
MIN_LR = 2e-5
EPOCH = 20
BATCH_SIZE = 128
CKPT_DIR = make_dir()



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
        f.write(f"MODEL_TARGET: {MODEL_TARGET}\n")
        f.write(f"MODEL_MODE  : {MODEL_MODE}\n")
        f.write(f"PRED_MODE   : {PRED_MODE}\n")
        f.write(f"BATCH_LEN   : {BATCH_LEN}\n")
        f.write(f"N           : {N}\n")
        f.write(f"DIM         : {DIM}\n")
        f.write(f"NUM_HEADS   : {NUM_HEADS}\n")
        f.write(f"QKV_DIM     : {QKV_DIM}\n")
        f.write(f"DROPOUT     : {DROPOUT}\n")
        f.write(f"CONV_NUM    : {CONV_NUM}\n")
        f.write(f"CONV_DIM    : {CONV_DIM}\n\n")

        f.write(f"# Preprocess parameters\n")
        f.write(f"RANDOM_SEED    : {RANDOM_SEED}\n")
        f.write(f"DATASET_HOP    : {DATASET_HOP}\n")
        f.write(f"TRAIN_BATCH_LEN: {TRAIN_BATCH_LEN}\n")
        f.write(f"VALID_BATCH_LEN: {VALID_BATCH_LEN}\n\n")

        f.write(f"# Training parameters\n")
        f.write(f"INITIAL_LR  : {INITIAL_LR}\n")
        f.write(f"WARMUP_STEPS: {WARMUP_STEPS}\n")
        f.write(f"DECAY_STEPS : {DECAY_STEPS}\n")
        f.write(f"DECAY_RATE  : {DECAY_RATE}\n")
        f.write(f"MIN_LR      : {MIN_LR}\n")
        f.write(f"EPOCH       : {EPOCH}\n")
        f.write(f"BATCH_SIZE  : {BATCH_SIZE}\n")


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate, min_lr):
        super(CustomSchedule, self).__init__()
        self.ilr = initial_lr
        self.ws  = warmup_steps
        self.ds  = decay_steps
        self.dr  = decay_rate
        self.mlr = min_lr
    def __call__(self, step):
        if self.ws == 0 or self.ws == None:
            return tf.math.maximum(self.ilr * self.dr ** (step / self.ds), self.mlr)
        else:
            arg1 = step / self.ws
            arg2 = self.dr ** ((step - self.ws) / self.ds)
            return tf.math.maximum(self.ilr * tf.math.minimum(arg1, arg2), self.mlr)


# Customized training step
# @tf.function
def train_step(x, y_real):
    
    with tf.GradientTape() as tape:
        y_pred, attns_forward, attns_backward = model(x, training=True)
        all_loss, mid_loss = loss_function(y_real, y_pred, PRED_MODE, loss_criterion)
    all_acc, mid_acc = accuracy_function(y_real, y_pred, PRED_MODE)

    if   MODEL_MODE == 'seq2seq': gradients = tape.gradient(all_loss, model.trainable_variables)
    elif MODEL_MODE == 'seq2one': gradients = tape.gradient(mid_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if MODEL_MODE == 'seq2seq': train_all_loss(all_loss)
    train_mid_loss(mid_loss)

    if MODEL_MODE == 'seq2seq': train_all_acc(all_acc)
    train_mid_acc(mid_acc)

    return y_pred, attns_forward, attns_backward


# Customized validating step
def valid_step(x, y_real):

    y_pred, _, _ = model(x)
    all_loss, mid_loss = loss_function(y_real, y_pred, PRED_MODE, loss_criterion)
    all_acc, mid_acc = accuracy_function(y_real, y_pred, PRED_MODE)

    if MODEL_MODE == 'seq2seq': valid_all_loss(all_loss)
    valid_mid_loss(mid_loss)

    if MODEL_MODE == 'seq2seq': valid_all_acc(all_acc)
    valid_mid_acc(mid_acc)
    
    return y_pred


# Main process
def main():


    # Save parameters first for records
    save_details()


    # Load dataset
    x_dataset, y_dataset = read_data(
        r"../customized_data/CE200",
        LOAD_SONG_AMOUNT, SAMPLE_RATE, HOP_LENGTH, MODEL_TARGET, PRED_MODE
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


    # Global variable 'loss_criterion' for both training and validating step
    # Global variable 'optimizer', 'train_all_loss', 'train_mid_loss', 'train_all_acc' and 'train_mid_acc' for training step
    # Global variable 'valid_all_loss', 'valid_mid_loss', 'valid_all_acc' and 'valid_mid_acc' for validating step
    global loss_criterion, optimizer
    global train_all_loss, train_mid_loss, train_all_acc, train_mid_acc
    global valid_all_loss, valid_mid_loss, valid_all_acc, valid_mid_acc
    if PRED_MODE == 'quality_bitmap': 
        loss_criterion = tf.keras.losses.MeanSquaredError(reduction='none')
    else:
        loss_criterion = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(  # lr_schedule
        initial_lr=INITIAL_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        min_lr=MIN_LR,
    ))
    train_all_loss = tf.keras.metrics.Mean()
    train_mid_loss = tf.keras.metrics.Mean()
    train_all_acc  = tf.keras.metrics.Mean()
    train_mid_acc  = tf.keras.metrics.Mean()
    valid_all_loss = tf.keras.metrics.Mean()
    valid_mid_loss = tf.keras.metrics.Mean()
    valid_all_acc  = tf.keras.metrics.Mean()
    valid_mid_acc  = tf.keras.metrics.Mean()


    # Define the model & checkpoint
    global model
    model = MyModel(MODEL_TARGET, PRED_MODE, BATCH_LEN, DIM, DROPOUT, QKV_DIM, N, NUM_HEADS, CONV_NUM, CONV_DIM)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    min_all_loss_ckpt_manager = tf.train.CheckpointManager(ckpt, f"{CKPT_DIR}/min_all_loss/", max_to_keep=1)
    min_mid_loss_ckpt_manager = tf.train.CheckpointManager(ckpt, f"{CKPT_DIR}/min_mid_loss/", max_to_keep=1)
    max_all_acc_ckpt_manager  = tf.train.CheckpointManager(ckpt, f"{CKPT_DIR}/max_all_acc/", max_to_keep=1)
    max_mid_acc_ckpt_manager  = tf.train.CheckpointManager(ckpt, f"{CKPT_DIR}/max_mid_acc/", max_to_keep=1)


    # Average losses, accuracies and learning rate per epoch for plotting figures
    avg_train_all_losses = []
    avg_train_mid_losses = []
    avg_train_all_accs   = []
    avg_train_mid_accs   = []
    avg_valid_all_losses = []
    avg_valid_mid_losses = []
    avg_valid_all_accs   = []
    avg_valid_mid_accs   = []
    learning_rates       = []
    print('')
    for epoch in range(EPOCH):


        # According to limited memory space (RAM),
        #   train the model based on part of data each epoch. (TRAIN_BATCH_LEN / total)
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

        # According to limited memory space (RAM),
        #   validate the model based on part of data each epoch. (VALID_BATCH_LEN / total)
        valid_batches = []
        random_valid_idx = np.arange(len(x_valid)-BATCH_SIZE)
        np.random.shuffle(random_valid_idx)
        global VALID_BATCH_LEN
        if VALID_BATCH_LEN is None or VALID_BATCH_LEN > len(random_valid_idx):
            VALID_BATCH_LEN = len(random_valid_idx)
        else:
            random_valid_idx = random_valid_idx[:VALID_BATCH_LEN]
        progress_bar = tqdm(
            random_valid_idx, total=len(random_valid_idx), ascii=True,
            desc=f"Sampling random valid batches " +
                 f"({VALID_BATCH_LEN}/{len(x_valid)-BATCH_SIZE}={VALID_BATCH_LEN/(len(x_valid)-BATCH_SIZE)*100:.3f}%)"
        )
        for i in progress_bar:
            valid_batches.append((x_valid[i:i+BATCH_SIZE], y_valid[i:i+BATCH_SIZE]))


        # Reset metrics
        train_all_loss.reset_states()
        train_mid_loss.reset_states()
        train_all_acc.reset_states()
        train_mid_acc.reset_states()
        valid_all_loss.reset_states()
        valid_mid_loss.reset_states()
        valid_all_acc.reset_states()
        valid_mid_acc.reset_states()


        # Train model
        print('')
        print(f"EPOCH {epoch+1:2d}/{EPOCH} [TRAIN]:")
        progress_bar = tqdm(enumerate(train_batches), desc=f"{epoch+1:2d}/{EPOCH}", total=len(train_batches), ascii=True)
        for (i, (x_input, y_real)) in progress_bar:
            y_pred, attns_forward, attns_backward = train_step(x_input, y_real)
            if MODEL_MODE == 'seq2seq':
                progress_bar.set_description(
                    f"AL {train_all_loss.result():.4f} ML {train_mid_loss.result():.4f} " +  # all loss & mid loss
                    f"AA {train_all_acc.result():.2f}% MA {train_mid_acc.result():.2f}% " +  # all acc  & mid acc
                    f"LR {optimizer._decayed_lr('float32').numpy():.6f}")
                    # f" | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
            else:
                progress_bar.set_description(
                    f"{epoch+1:2d}/{EPOCH} TRAIN | " +
                    f"mid loss: {train_mid_loss.result():.4f} | " +
                    f"mid acc: {train_mid_acc.result():.2f}% | " +
                    f"lr: {optimizer._decayed_lr('float32').numpy():.10f}")  # | ")  # +
                    # f"BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        if MODEL_MODE == 'seq2seq': avg_train_all_losses.append(train_all_loss.result())
        avg_train_mid_losses.append(train_mid_loss.result())
        if MODEL_MODE == 'seq2seq': avg_train_all_accs.append(train_all_acc.result())
        avg_train_mid_accs.append(train_mid_acc.result())
        learning_rates.append(optimizer._decayed_lr('float32').numpy())


        # Validate model
        # os.system('cls')
        # print('')
        print(f"EPOCH {epoch+1:2d}/{EPOCH} [VALID]:")
        progress_bar = tqdm(valid_batches, desc=f"{epoch+1:2d}/{EPOCH}", total=len(valid_batches), ascii=True)
        for (x_input, y_real) in progress_bar:
            y_pred = valid_step(x_input, y_real)
            if MODEL_MODE == 'seq2seq':
                progress_bar.set_description(
                    f"AL {valid_all_loss.result():.4f} ML {valid_mid_loss.result():.4f} " +  # all loss & mid loss
                    f"AA {valid_all_acc.result():.2f}% MA {valid_mid_acc.result():.2f}%")    # all acc  & mid acc
                    # f" | BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
            else:
                progress_bar.set_description(
                    f"{epoch+1:2d}/{EPOCH} VALID | " +
                    f"mid loss: {valid_mid_loss.result():.4f} | " +
                    f"mid acc: {valid_mid_acc.result():.2f}%")  #  | " +
                    # f"BATCH_LEN: {BATCH_LEN} | BATCH_SIZE: {BATCH_SIZE}")
        if MODEL_MODE == 'seq2seq': avg_valid_all_losses.append(valid_all_loss.result())
        avg_valid_mid_losses.append(valid_mid_loss.result())
        if MODEL_MODE == 'seq2seq': avg_valid_all_accs.append(valid_all_acc.result())
        avg_valid_mid_accs.append(valid_mid_acc.result())


        # Save the model checkpoint if this is the best one (with minimum loss or maximum accuracy).
        print('')
        if MODEL_MODE == 'seq2seq' and valid_all_loss.result() <= min(avg_valid_all_losses):
            print("Minimum all loss: improved! Model saved.")
            min_all_loss_ckpt_manager.save()
        else: print("Minimum all loss: did not improve, model not saved.")
        if valid_mid_loss.result() <= min(avg_valid_mid_losses):
            print("Minimum mid loss: improved! Model saved.")
            min_mid_loss_ckpt_manager.save()
        else: print("Minimum mid loss: did not improve, model not saved.")
        if MODEL_MODE == 'seq2seq' and valid_all_acc.result() >= max(avg_valid_all_accs):
            print("Maximum all acc : improved! Model saved.")
            max_all_acc_ckpt_manager.save()
        else: print("Maximum all acc : did not improve, model not saved.")
        if valid_mid_acc.result() >= max(avg_valid_mid_accs):
            print("Maximum mid acc : improved! Model saved.")
            max_mid_acc_ckpt_manager.save()
        else: print("Maximum mid acc : did not improve, model not saved.")
        print('')

        show_pred_and_truth(y_real, y_pred, PRED_MODE)

        print("Plotting history figure... ", end='')
        plot_history(CKPT_DIR, {
            'train_all_loss': avg_train_all_losses,
            'train_mid_loss': avg_train_mid_losses,
            'train_all_acc' : avg_train_all_accs,
            'train_mid_acc' : avg_train_mid_accs,
            'valid_all_loss': avg_valid_all_losses,
            'valid_mid_loss': avg_valid_mid_losses,
            'valid_all_acc' : avg_valid_all_accs,
            'valid_mid_acc' : avg_valid_mid_accs,
            'lr'            : learning_rates,
        })
        print("Done.")

        print("Plotting attention figures... ", end='')
        plot_attns(CKPT_DIR, attns_forward, attns_backward)
        print("Done.\n")

    return



''' Exection '''
if __name__ == '__main__':
    main()