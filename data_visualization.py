import os
import matplotlib.pyplot as plt

def plot_train_val_acc(train_acc, val_acc, save_dir="plots", filename="train_val_accuracy.png"):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")