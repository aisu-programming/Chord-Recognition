# usage: python trainer.py <train_dataset> <test_dataset> <batch_size> <num_epochs> <model_path>
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import numpy as np
import time
import sys
import os
import librosa

from net import Net
from data_utils.dp_post_processing import dp_post_processing

def frame_loss(Y, out):
    loss = nn.BCEWithLogitsLoss()
    return loss(out, Y)

def to_probability(raw_output):
    sm = nn.Softmax(dim=0)
    for idx in range(raw_output.shape[1]):
        raw_output[:13, idx] = sm(raw_output[:13, idx])
        raw_output[13:, idx] = sm(raw_output[13:, idx])
    return raw_output.numpy()


class AudioDataset(Dataset):
    def __init__(self, files_path):
        audio_path = os.path.join(files_path, "data")
        lab_path = os.path.join(files_path, "lab")

        self.audio_paths = sorted([os.path.join(audio_path, f) for f in os.listdir(audio_path)], key=lambda x: (len(x), x))
        self.lab_paths = sorted([os.path.join(lab_path, f) for f in os.listdir(lab_path)], key=lambda x: (len(x), x))

        # Get max length
        self.max_length = 0
        for idx in range(len(self.audio_paths)):
            tmp_lab = np.load(self.lab_paths[idx])
            if tmp_lab.shape[1] > self.max_length:
                self.max_length = tmp_lab.shape[1]

    def __getitem__(self, idx):
        # Pad the song to the maximum length
        raw_X = np.load(self.audio_paths[idx])
        raw_Y = np.load(self.lab_paths[idx])
        frame_num = raw_X.shape[1]

        X = np.zeros((raw_X.shape[0], self.max_length))
        X -= 80
        X[:, :frame_num] = raw_X

        Y = np.zeros((raw_Y.shape[0], self.max_length))
        Y[12][frame_num:] = np.ones(self.max_length - frame_num)
        Y[:, :frame_num] = raw_Y
        Y = np.expand_dims(Y, axis=0)

        return torch.from_numpy(X), torch.from_numpy(Y)

    def __len__(self):
        return len(self.audio_paths)


class Trainer:
    def __init__(self, train_path, test_path, bs, ep, model_path=""):
        # Set params
        self.train_path = train_path
        self.test_path = test_path
        self.bs = bs
        self.ep = ep

        # Create model
        self.model = nn.DataParallel(Net(1).cuda())
        if model_path != "":
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")["state_dict"]
            )

        # Set data loaders
        self.train_loader = DataLoader(
            AudioDataset(train_path),
            batch_size=self.bs,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            AudioDataset(test_path), batch_size=1, num_workers=0, pin_memory=True
        )

    def show_dataset_model_params(self):
        summary(self.model, (192, 2000))

    def chord_symbol_recall(self, X, Y):
        X = X.T
        Y = Y.T
        right_count = 0
        total_length = X.shape[0]
        for idx in range(total_length):
            colx = X[idx]
            coly = Y[idx]
            root = np.argmax(colx[:13])
            quality = np.argmax(colx[13:]) + 13
            real_root = np.argmax(coly[:13])
            real_quality = np.argmax(coly[13:]) + 13

            # Set confusion matrix
            if coly[12] != 1:
                self.confusion_matrix[root][real_root] += 1
                self.confusion_matrix[quality][real_quality] += 1

            # Ignore N
            if coly[12] == 1:
                total_length -= 1
                continue

            # Quality
            if coly[root] == 1 and coly[quality] == 1:
                right_count += 1

        return (0, 0) if total_length == 0 else (right_count / total_length, total_length)

    def test(self, model_path=""):
        # Load existing model if path is given
        if model_path != "":
            self.model = nn.DataParallel(Net(1).cuda())
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")["state_dict"]
            )

        start_time = time.time()
        self.model.eval()
        total_csr = 0
        total_length = 0

        self.confusion_matrix = np.zeros((22, 22))
        for batch_idx, (X, Y) in enumerate(self.test_loader):
            X, Y = Variable(X.float().cuda()), Variable(Y.float().cuda())
            estimation = self.model(X).data.cpu()[0][0]
            estimation = to_probability(estimation)

            # Post-processing
            estimation = dp_post_processing(estimation, alpha=0.25)

            # Calculate CSR
            csr, song_length = self.chord_symbol_recall(estimation, Y.data.cpu().numpy()[0][0])

            total_csr += csr * song_length
            total_length += song_length

        np.save("confusion_matrix.npy", self.confusion_matrix)
        wcsr = total_csr / total_length
        print("model: %s, Test WCSR: %f Time: %f" % (os.path.basename(model_path), wcsr, time.time() - start_time))
        return wcsr

    def fit(self):
        start_time = time.time()
        save_dict = {}
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        loss_list = []
        for epoch in range(1, self.ep + 1):
            # Training
            total_loss = 0
            for batch_idx, (X, Y) in enumerate(self.train_loader):
                X, Y = Variable(X.float().cuda()), Variable(Y.float().cuda())

                out = self.model(X)
                loss = frame_loss(Y, out)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Show training message
                print(
                    "\r| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d |"
                    % (
                        epoch,
                        self.ep,
                        batch_idx + 1,
                        len(self.train_loader),
                        loss.item(),
                        time.time() - start_time,
                    ),
                    end="",
                )
            print()

            # Save loss list
            loss_list.append(total_loss / len(self.train_loader))
            np.save("train_loss", np.array(loss_list))

            # Save model
            save_dict["state_dict"] = self.model.state_dict()
            directory = "models/%s" % (self.model.module.model_name)
            print("Saved to: " + directory + "/e_%d" % (epoch))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(save_dict, directory + "/e_%d" % (epoch))
            print(
                "Training Epoch: %d, Finish time: %1f\n"
                % (epoch, time.time() - start_time)
            )

if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        sys.exit("usage: python trainer <train_path> <test_path> <batch_size> <epoch> <model_path>")

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    batch_size = int(sys.argv[3]) #8
    num_epoch = int(sys.argv[4])  #100
    model_path = sys.argv[5] if len(sys.argv) == 6 else "" #./model

    trainer = Trainer(train_path, test_path, batch_size, num_epoch, model_path)
    trainer.fit()
    trainer.test()
