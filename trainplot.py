import matplotlib.pyplot as plt

def make_training_plot(name='single_layer'):
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle(name + ' Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    return fig, axes

def update_training_plot(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    train_loss = [s[3] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0].legend(['Validation', 'Train'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[1].legend(['Validation', 'Train'])
    plt.pause(0.00001)

def savetraining_plot(fig, name='single_layer'):
    """
    Saves the training plot to a file
    """
    fig.savefig(name + '_training_plot.png', dpi=200)


