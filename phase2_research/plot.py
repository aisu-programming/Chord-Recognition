''' Libraries '''
import numpy as np
import matplotlib.pyplot as plt


''' Function '''
def plot_history(ckpt_dir, history):

    train_all_loss = history['train_all_loss']
    train_mid_loss = history['train_mid_loss']
    valid_all_loss = history['valid_all_loss']
    valid_mid_loss = history['valid_mid_loss']

    train_all_acc  = history['train_all_acc']
    train_mid_acc  = history['train_mid_acc']
    valid_all_acc  = history['valid_all_acc']
    valid_mid_acc  = history['valid_mid_acc']

    lr             = history['lr']

    epochs_length  = range(1, 1+len(train_all_loss))

    fig, axs = plt.subplots(3)
    fig.set_size_inches(14, 20)
    fig.suptitle('History')
    plt.xlabel('Epochs')

    axs[0].set_xticks(np.arange(0, 100.1, 5))
    axs[0].set_xticks(np.arange(0, 100.1, 1), minor=True)
    axs[0].set_ylabel('Loss')
    axs[0].set_yticks(np.arange(0, 101, 1)/2)
    axs[0].set_yticks(np.arange(0, 101, 1)/10, minor=True)
    axs[0].grid()
    axs[0].grid(which='minor', alpha=0.3)
    if train_all_loss != []: axs[0].plot(epochs_length, train_all_loss, 'b-', label="Training (All)")
    if train_mid_loss != []: axs[0].plot(epochs_length, train_mid_loss, 'c-', label="Training (Middle)")
    if valid_all_loss != []: axs[0].plot(epochs_length, valid_all_loss, 'r-', label="Validation (All)")
    if valid_mid_loss != []: axs[0].plot(epochs_length, valid_mid_loss, 'm-', label="Validation (Middle)")
    axs[0].legend()

    axs[1].set_xticks(np.arange(0, 100.1, 5))
    axs[1].set_xticks(np.arange(0, 100.1, 1), minor=True)
    axs[1].set_ylabel('Accuracy')
    axs[1].set_yticks(np.arange(0, 100.1, 5))
    axs[1].set_yticks(np.arange(0, 100.1, 1), minor=True)
    axs[1].grid()
    axs[1].grid(which='minor', alpha=0.3)
    if train_all_acc != []: axs[1].plot(epochs_length, train_all_acc, 'b-', label="Training (All)")
    if train_mid_acc != []: axs[1].plot(epochs_length, train_mid_acc, 'c-', label="Training (Middle)")
    if valid_all_acc != []: axs[1].plot(epochs_length, valid_all_acc, 'r-', label="Validation (All)")
    if valid_mid_acc != []: axs[1].plot(epochs_length, valid_mid_acc, 'm-', label="Validation (Middle)")
    axs[1].legend()

    axs[2].set_xticks(np.arange(0, 100.1, 5))
    axs[2].set_xticks(np.arange(0, 100.1, 1), minor=True)
    axs[2].set_ylabel('Learning Rate')
    axs[2].plot(epochs_length, lr, 'b-')
    axs[2].grid()
    axs[2].grid(which='minor', alpha=0.3)

    plt.tight_layout()
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

    col_num = 8
    plot_width = 36  # 4 -> 18 ( 1 : 4.5 )

    row_num = 4
    plot_height = 20  # 4 -> 20 (1 : 5)

    for i in range(attns_forward.shape[0]):

        fig, axs = plt.subplots(row_num, col_num)
        fig.set_size_inches(plot_width, plot_height)
        fig.suptitle(f"Forward attentions (N={i+1})", fontsize=20)
        for j in range(attns_forward.shape[1]):
            axs[j//col_num][j%col_num].set_title(j+1)
            axs[j//col_num][j%col_num].matshow(attns_forward[i, j])
        plt.tight_layout()
        plt.savefig(f"{ckpt_dir}/attns_forward_N={i+1}.png", dpi=200)

        fig, axs = plt.subplots(row_num, col_num)
        fig.set_size_inches(plot_width, plot_height)
        fig.suptitle(f"Backward attentions (N={i+1})", fontsize=20)
        for j in range(attns_backward.shape[1]):
            axs[j//col_num][j%col_num].set_title(j+1)
            axs[j//col_num][j%col_num].matshow(attns_backward[i, j])
        plt.tight_layout()
        plt.savefig(f"{ckpt_dir}/attns_backward_N={i+1}.png", dpi=200)
        plt.close('all')

    return