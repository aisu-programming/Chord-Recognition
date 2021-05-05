''' Libraries '''
import time, os
import numpy as np
import matplotlib.pyplot as plt


''' Function '''
def make_dir():
    ckpt_dir = f"ckpt/{time.strftime('%Y.%m.%d_%H.%M', time.localtime())}"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    return ckpt_dir


def plot_history(ckpt_dir, history):
    
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    valid_loss = history['valid_loss']
    valid_acc = history['valid_acc']
    lr = history['lr']
    epochs_length = range(1, 1+len(train_loss))

    fig, axs = plt.subplots(3)
    fig.set_size_inches(14, 20)
    fig.suptitle('History')
    plt.xlabel('Epochs')

    axs[0].set_ylabel('Loss')
    axs[0].plot(epochs_length, train_loss, "b-", label='Training')
    if valid_loss != []: axs[0].plot(epochs_length, valid_loss, "r-", label='Validation')
    axs[0].legend()

    axs[1].set_ylabel('Accuracy')
    axs[1].plot(epochs_length, train_acc, "b-", label='Training')
    if valid_acc != []: axs[1].plot(epochs_length, valid_acc, "r-", label='Validation')
    axs[1].legend()

    axs[2].set_ylabel('Learning Rate')
    axs[2].plot(epochs_length, lr, "b-")

    # plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/history.png", dpi=200)
    return


def plot_loss_lr(ckpt_dir, history):

    raise NotImplementedError
    
    loss = history['loss'][-20:]
    lr = history['lr'][-20:]
    epochs_length = range(1, min(len(loss)+1, 21))

    fig, axs = plt.subplots(2)
    fig.set_size_inches(12, 16)
    fig.suptitle('Loss LR Comparition')
    plt.xlabel('Steps')
    axs[0].plot(epochs_length, loss, "b-", label='Loss')
    axs[0].legend()
    axs[1].plot(epochs_length, lr, "b-", label='Learning Rate')
    axs[1].legend()

    plt.savefig(f"{ckpt_dir}/Loss_LR.png", dpi=200)
    return


def plot_attns(ckpt_dir, attns_forward, attns_backward):

    attns_forward  = np.squeeze(np.array(attns_forward)[:, -1])
    attns_backward = np.squeeze(np.array(attns_backward)[:, -1])

    for i in range(attns_forward.shape[0]):

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(20, 23)
        fig.suptitle(f"Forward attentions (N={i+1})", fontsize=20)
        for j in range(attns_forward.shape[1]):
            axs[j//4][j%4].set_title(j+1)
            axs[j//4][j%4].matshow(attns_forward[i, j])
            # axs[j//4][j%4].axis('off')
        plt.tight_layout()
        plt.savefig(f"{ckpt_dir}/attns_forward_N={i+1}.png", dpi=200)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(20, 23)
        fig.suptitle(f"Backward attentions (N={i+1})", fontsize=20)
        for j in range(attns_backward.shape[1]):
            axs[j//4][j%4].set_title(j+1)
            axs[j//4][j%4].matshow(attns_backward[i, j])
            # axs[j//4][j%4].axis('off')
        plt.tight_layout()
        plt.savefig(f"{ckpt_dir}/attns_backward_N={i+1}.png", dpi=200)
        plt.close('all')

    return