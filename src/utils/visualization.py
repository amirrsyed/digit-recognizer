import pandas as pd
import matplotlib.pyplot as plt

def plot_history(history):
    history_frame = pd.DataFrame(history.history)

    history_frame[['loss', 'val_loss']].plot()
    plt.title("Loss")
    plt.savefig("outputs/figures/loss.png")

    history_frame[['accuracy', 'val_accuracy']].plot()
    plt.title("Accuracy")
    plt.savefig("outputs/figures/accuracy.png")
