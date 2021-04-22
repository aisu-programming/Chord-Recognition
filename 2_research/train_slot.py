''' Libraries from sample code '''
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from tqdm import trange
from dataset_slot import SeqClsDataset
from utils_slot import Vocab


''' Libraries added by me '''
import os, time
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from torch.utils.data import DataLoader
from tf_model_slot import SeqClassifier
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


''' Parameters from sample code '''
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


''' Functions '''
def parse_args() -> Namespace:

    now_time = time.strftime('%m%d_%H%M%S', time.localtime())

    parser = ArgumentParser()

    # data
    parser.add_argument("--max_len", type=int, default=37)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # optimizer
    # parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--decay_rate", type=float, default=0.996)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--num_epoch", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=5)

    # path
    parser.add_argument(
        "--data_dir", type=Path, help="Directory to the dataset.",
        default="./data/slot")
    parser.add_argument(
        "--cache_dir", type=Path, help="Directory to the preprocessed caches.",
        default="./cache/slot")
    parser.add_argument(
        "--ckpt_dir", type=Path, help="Directory to save the model file.",
        default=f"./ckpt/slot/{now_time}")
    parser.add_argument(
        "--logs_dir", type=Path, help="Directory to save the logs file.",
        default=f"./logs/slot/{now_time}")

    args = parser.parse_args()
    return args


def save_args(args):
    d = vars(args)
    with open(f"{args.ckpt_dir}/args.txt", mode='w') as f:
        for key in d.keys():
            f.write(f"{key:20}: {d[key]}\n")
    return
    

def get_data(cache_dir) -> Dict[str, DataLoader]:

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(
            split_dataset,
            batch_size = 99999,
            shuffle = False,
            collate_fn = split_dataset.collate_fn
        ) for split, split_dataset in datasets.items()
    }
    return datasets, data_loaders


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):

    # accuracies = tf.equal(tf.cast(real, dtype=tf.int64), tf.argmax(pred, axis=2))

    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    # accuracies = tf.math.logical_and(mask, accuracies)

    # tf.print(mask, summarize=-1)
    # tf.print(accuracies, summarize=-1)

    # accuracies = tf.cast(accuracies, dtype=tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)

    # return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    accuracies = tf.equal(tf.cast(real, dtype=tf.int64), tf.argmax(pred, axis=-1))

    mask = tf.math.equal(real, 0)
    accuracies = tf.cast(mask, dtype=tf.uint8) + tf.cast(accuracies, dtype=tf.uint8)
    accuracies = tf.reduce_min(accuracies, axis=-1)
    
    return tf.reduce_sum(accuracies)/len(accuracies)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(X, Y):
    enc_padding_mask = create_padding_mask(X)
    dec_padding_mask = create_padding_mask(X)

    dec_target_padding_mask = create_padding_mask(Y)
    look_ahead_mask = create_look_ahead_mask(tf.shape(Y)[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


@tf.function
def train_step(model, x, y):
    y_input = y[:, :-1]
    y_real  = y[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y_input)

    with tf.GradientTape() as tape:
        predictions = model(x, y_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
        
        loss = loss_function(y_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(accuracy_function(y_real, predictions))

    return predictions


def valid_step(model, data_loader, text_len):

    for d in data_loader:
        valid_X = tf.convert_to_tensor(d['encoded_tokens'], dtype=tf.float32)
        valid_Y = tf.convert_to_tensor(d['tags'], dtype=tf.float32)

    valid_batches = []
    for i in range(len(valid_X)//args.batch_size+1):
        if i == len(valid_X)//args.batch_size:
            valid_batches.append((valid_X[i*args.batch_size:], valid_Y[i*args.batch_size:]))
        else:
            valid_batches.append((valid_X[i*args.batch_size:(i+1)*args.batch_size], valid_Y[i*args.batch_size:(i+1)*args.batch_size]))
    
    pred_Y = tf.convert_to_tensor(np.empty((0, 37, 12), dtype=np.float32))
    pbar = tqdm(enumerate(valid_batches), desc=f"Valid   ", total=len(valid_batches))
    for (_, (x, _)) in pbar:
        output_id = tf.convert_to_tensor(np.concatenate([np.array([0], dtype=np.float32)[np.newaxis, :]]*x.shape[0]))
        output = tf.convert_to_tensor(np.concatenate([np.array(
            [ [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ] ]
        , dtype=np.float32)[np.newaxis, :]]*x.shape[0]))
        for i in range(text_len-1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, output_id)

            predictions = model(x, output_id, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            predicted_id = tf.cast(predicted_id, dtype=tf.float32)

            output    = tf.concat([output, predictions], axis=-2)
            output_id = tf.concat([output_id, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end 
            # token
            # if predicted_id == end:
            # break

        pred_Y = tf.concat([pred_Y, output], axis=0)
    
    valid_loss(loss_function(valid_Y, pred_Y))
    valid_acc(accuracy_function(valid_Y, pred_Y))
    print(f"\tValid Loss = {valid_loss.result():.8f} | Valid Accuracy = {valid_acc.result()*100:.3f}%\n")
    return


def main(args):
    
    datasets, data_loaders = get_data(args.cache_dir)

    for d in data_loaders[TRAIN]:
        train_X = tf.convert_to_tensor(d['encoded_tokens'], dtype=tf.float32)
        train_Y = tf.convert_to_tensor(d['tags'], dtype=tf.float32)

    embeddings = tf.convert_to_tensor(torch.load(args.cache_dir / "embeddings.pt"))
    model = SeqClassifier(
        embeddings=embeddings,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_class=datasets[TRAIN].num_classes,
    )

    # os.system('clear')

    # print args
    print("\n")
    d = vars(args)
    for key in d.keys():
        print(f"{key:20}: {d[key]}")
    print("\n")

    global loss_object, optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    optimizer = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(  # lr_schedule
            initial_learning_rate=0.1,
            decay_steps=10000,
            decay_rate=0.8
    ))

    global train_loss, train_acc, valid_loss, valid_acc
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Mean(name='train_acc')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.Mean(name='valid_acc')

    train_batches = []
    for i in range(len(train_X)//args.batch_size+1):
        if i == len(train_X)//args.batch_size:
            train_batches.append((train_X[i*args.batch_size:], train_Y[i*args.batch_size:]))
        else:
            train_batches.append((train_X[i*args.batch_size:(i+1)*args.batch_size], train_Y[i*args.batch_size:(i+1)*args.batch_size]))

    for epoch in range(args.num_epoch):

        average_loss = []
        average_acc  = []
        train_loss.reset_states()
        train_acc.reset_states()

        # pred_Y = tf.convert_to_tensor(np.empty((0, 36), dtype=np.int64))
        pbar = tqdm(enumerate(train_batches), desc=f"Epoch {epoch+1:2d}", total=len(train_batches))
        for (_, (x, y)) in pbar:
            predictions = train_step(model=model, x=x, y=y)
            average_loss.append(train_loss.result())
            average_acc.append(train_acc.result())
            pbar.set_description(f"Epoch {epoch+1:2d}: train loss = {tf.math.reduce_mean(average_loss):.5f} | train acc = {tf.math.reduce_mean(average_acc)*100:.3f}%")
            # pred_Y = tf.concat([pred_Y, tf.argmax(predictions, axis=-1)], axis=-2)
        print(f"\tTrain Loss = {train_loss.result():.8f} | Train Accuracy = {train_acc.result()*100:.3f}%")

        tf.print("")
        tf.print("x[0]       : ", train_batches[-1][0][0], summarize=-1)
        # tf.print("y_input[0] : ", y_input[0], summarize=-1)
        tf.print("y[0]       : ", train_batches[-1][1][0, 1:], summarize=-1)
        tf.print("predictions: ", tf.argmax(predictions, axis=-1)[0], summarize=-1)
        tf.print("")

        # valid_step(model=model, data_loader=data_loaders[DEV], text_len=args.max_len)
        model.save_weights(filepath=f"{args.ckpt_dir}/best.h5")

        # print(pred_Y.shape)

    # classification_report(train_batches[-1][1][:, 1:], predictions, mode='strict', scheme=IOB2)

    return


''' Execution '''
if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir = Path(f"{args.ckpt_dir}_numlayers={args.num_layers}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    # args.logs_dir = Path(f"{args.logs_dir}_numlayers={args.num_layers}")
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)
    main(args)