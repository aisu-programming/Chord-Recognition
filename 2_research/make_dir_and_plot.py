''' Libraries '''
import time, os
import matplotlib.pyplot as plt


''' Function '''
def make_dir():
    ckpt_dir = f"ckpt/{time.strftime('%Y.%m.%d_%H.%M', time.localtime())}"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    return ckpt_dir


def plot_history(ckpt_dir, history):
    
    train_loss = history['train_loss']
    # train_acc = history['train_acc']
    valid_loss = history['valid_loss']
    # valid_acc = history['valid_acc']
    epochs_length = range(1, len(train_loss)+1)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(12, 16) # 3:4
    fig.suptitle('Training & Validation Comparition')
    plt.xlabel('Epochs')
    axs[0].plot(epochs_length, train_loss, "b-", label='Training Loss')
    axs[0].plot(epochs_length, valid_loss, "r-", label='Validation Loss')
    axs[0].legend()
    # axs[1].plot(epochs_length, train_acc, "b-", label='Training Accuracy')
    # axs[1].plot(epochs_length, valid_acc, "r-", label='Validation Accuracy')
    # axs[1].legend()

    plt.savefig(f"{ckpt_dir}/history.png", dpi=200)
    # plt.show()

    return