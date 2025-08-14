import os
import wfdb
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from data_visualization import plot_train_val_acc

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device:{device}")

class EMGCNN(nn.Module):
    """
    Convolutional Neural Network for EMG signal classification.

    This model processes multichannel EMG time-series data using
    1D convolution layers along the time axis, progressively extracting
    temporal features from all channels and producing gesture predictions.

    Architecture:
        Input:  (batch_size, 32, 10240)  # 32 EMG channels, 10240 time steps
        Conv1:  64 filters, kernel size 11, stride 1
        ReLU
        MaxPool1d: kernel size 2 (halves time length)
        Conv2:  128 filters, kernel size 5, stride 1
        ReLU
        MaxPool1d: kernel size 2
        AdaptiveAvgPool1d: output length 1 (global average pooling)
        Flatten
        Fully Connected (256 units) + ReLU + Dropout(0.1)
        Fully Connected (17 units) --> gesture class logits

    Output:
        torch.FloatTensor of shape (batch_size, 17) containing raw logits
        for each gesture class.
    """

    def __init__(self):
        """Initialize EMGCNN layers"""

        super().__init__()
        # 64 filters scanning over 32 channels with a window of 11
        self.conv1 = nn.Conv1d(32, 64, kernel_size=11)
        # keeps positive vals & sets negative vals to 0
        self.relu = nn.ReLU()
        # keeps max value in each window of 2
        self.pool = nn.MaxPool1d(2)
        # 128 filters over XX samples with a window of 5
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)  # final_length depends on input size
        self.dropout = nn.Dropout(0.1)
        # final classification layer
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        """
        Forward pass of the EMGCNN.

        Args:
            x (torch.FloatTensor): Input tensor of shape (batch_size, 32, time_steps).

        Returns:
            torch.FloatTensor: Output logits of shape (batch_size, 17) for classification.
        """

        x = self.pool(self.relu(self.conv1(x)))  # (B, 64, ~)88
        x = self.pool(self.relu(self.conv2(x)))  # (B, 128, ~)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class EMGData(Dataset):
    """
    Loads EMG data into torch.Tensors.

    Has methods to:
        - Load data from a given base folder
        - Return the sample number at a specific id
        - Return the length of samples

    Args:
        base_folder (str): Folder name in the current working directory containing the GrabMyo data files.
                           This base folder will be organized into session and participant folders:
                           "./base_folder
                             /Session{i}
                             /session{i}_participant{j}
                             /session{i}_participant{j}_gesture{k}_trial{l}"
    """

    def __init__(self, base_folder):

        self.base_folder = base_folder
        self.samples_as_tensor, self.labels_as_tensor = self.load_data()

    def load_data(self):
        """
        Loads samples and their labels as torch.Tensors.

        Returns:
            samples_as_tensor (torch.Tensor): Tensor containing physical signals of each sample
                                              of shape (15351, 10240, 32) = (# samples, # time steps, # channels)
            labels_as_tensor (torch.Tensor): Tensor containing gesture labels of each sample
                                             of shape (15351) = (# samples). 
                                             Labels are 0-16 for gestures 1-17.
        """

        num_samples = 15351 # 3 sessions x 43 participants x 17 gestures x 7 trials
        samples_np = np.empty((num_samples, 10240, 32), dtype=np.float32)
        labels_np = np.empty((num_samples,), dtype=np.int64)

        idx = 0

        for session in range(1, 4):

            for participant in range(1, 44):

                for gesture in range(1, 18):

                    for trial in range(1, 8):

                        session_folder = f"Session{session}"
                        participant_folder = f"session{session}_participant{participant}"

                        # ensure that the record path does not have .head or .dat extension
                        file_name_wo_extension = f"{participant_folder}_gesture{gesture}_trial{trial}"

                        record_path = os.path.join(self.base_folder, 
                                                session_folder, 
                                                participant_folder, 
                                                file_name_wo_extension)
                        
                        # update our np arrays with this record's signal data and gesture label
                        try:
                            # read the record
                            record = wfdb.rdrecord(record_path)

                            # extract physical signal of this record as a 2D numpy array
                            signal_data = record.p_signal  # Shape: (samples, channels) --> (10240, 32)

                            samples_np[idx] = signal_data.astype(np.float32)

                            # labels will range 0-16 for gestures 1-17
                            labels_np[idx] = gesture - 1

                            idx += 1

                        except Exception as e:
                            print(f"Error processing {file_name_wo_extension}: {e}")
        
        # convert our np arrays of samples and labels to tensors
        samples_as_tensor = torch.from_numpy(samples_np[:idx])
        labels_as_tensor = torch.from_numpy(labels_np[:idx])

        return samples_as_tensor, labels_as_tensor

    def __len__(self):
        return len(self.samples_as_tensor)

    def __getitem__(self, idx):
        """
        Get sample and label at a specific idx
        
        Args:
            idx: ID of the sample to return

        Returns:
            sample (torch.FloatTensor): Sample data as a tensor.
            label (long): Gesture label of the given sample.
        """

        sample = self.samples_as_tensor[idx].float()
        label = self.labels_as_tensor[idx].long()
        return sample, label

def train(dataloader, model, loss_fn, optimizer, device):
    """
    Trains model for one epoch using provided dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing training batches.
        model (nn.Module): Neural network model to train.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
        device (torch.device): Device to run the computations on (ex: cuda or cpu).

    Returns:
        train_loss (float): Average loss over the training epoch.
        train_acc (float): Training accuracy as a percent.
    """

    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.permute(0, 2, 1)  # [batch_size, channels, signal_length]

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(dataloader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate(dataloader, model, loss_fn, device):
    """
    Evaluates the model on validation set without updating parameters.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing validation batches.
        model (nn.Module): Neural network model to evaluate.
        loss_fn (nn.Module): Loss function for calculating validation loss.
        device (torch.device): Device to run the computations on (ex: cuda or cpu).

    Returns:
        val_loss (float): Average loss over the validation dataset.
        val_acc (float): Validation accuracy as a percentage.
    """

    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 2, 1)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def main():

    # folder organized into session and participant folders containing each record
    base_folder = "./physionet.org/files/grabmyo/1.1.0"
    dataset = EMGData(base_folder)

    # set random seed for reproducibility
    random_seed = 4

    # define 80/10/10 split for training, validation, and test data
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = EMGCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(model)
    best_acc = float("-inf")

    train_acc_list = []

    train_acc_list = []
    val_acc_list = []

    num_epochs = 100

    for epoch in range(num_epochs + 1):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")

        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = validate(val_loader, model, loss_fn, device)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")

        # save model with best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc

            model_save_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(model_save_dir, exist_ok=True)

            model_file_path = os.path.join(model_save_dir, f"best_emg_cnn_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_file_path)

            print(f"Saved Best Model with Accuracy: {best_acc:.2f}%\n")

    plot_train_val_acc(train_acc_list, val_acc_list)

    print("=== Final Evaluation on Test Set ===")
    final_test_loss, final_test_acc = validate(test_loader, model, loss_fn, device)
    print(f"Test Accuracy: {final_test_acc:.2f}%, Test Loss: {final_test_loss:.4f}")

if __name__ == "__main__":
    main()