# cnn-for-emg-signal-processing

EMG (electromyography) measures the electrical activity of muscles. Raw EMG signals require processing to extract relevant information, allowing us to find patterns that help in tasks like gesture recognition. This project trains a Convolutional Neural Network on the GRABMyo dataset of EMG records (https://physionet.org/content/grabmyo/1.1.0/) for classification of signals into one of 17 gestures.

### Project Structure
```
cnn-for-emg-signal-processing/
├── models/ # Directory to save best CNN model's .pth files
├── plots/ # Directory to which plots will be saved visualizing training accuracy over time
    ├── train_val_accuracy_example.png # Shows an example training/validation accuracy plot
├── physionet.org/files/grabmyo/1.1.0/ # Directory containing session and participant subdirectories with EMG records
    ├── Session1
        ├──session1_participant1
            ├──session1_participant1_gesture1_trial_1.hea # Actual EMG signals in binary format
            ├──session1_participant1_gesture1_trial_1.dat # Metadata for .dat record
            .
            .
            .
        .
        .
        .
    .
    .
    .
├── cnn_signal_processing.py # Trains a CNN on the EMG dataset
├── data_visualization.py # Provides methods to visualize CNN training accuracy over epochs
├── requirements.txt # Python dependencies
└── README.md
```

### GRABMyo dataset
GRABMyo contains 15,351 EMG records collected from forearm and wrist muscles. Download it in the terminal using
```
wget -r -N -c -np https://physionet.org/files/grabmyo/1.1.0/
```

## Installation & Running

1. **Clone this repository**:
```bash
git clone https://github.com/nataliekozlowskii/cnn-for-emg-signal-processing.git
cd cnn-for-emg-signal-processing
```
2. **Install dependencies:**
```
pip install -r requirements.txt
```
3. **Run the script:**
```
python cnn_signal_processing.py
```

### Citations
Jiang, N., Pradhan, A., & He, J. (2024). Gesture Recognition and Biometrics ElectroMyogram (GRABMyo) (version 1.1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/89dm-f662

Pradhan, A., He, J. & Jiang, N. Multi-day dataset of forearm and wrist electromyogram for hand gesture recognition and biometrics. Sci Data 9, 733 (2022).

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
