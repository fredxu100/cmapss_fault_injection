from pathlib import Path
import numpy as np
import torch

from cmapps_rnn.parse_data import process_cmapss_data
from cmapps_rnn.RNN import RNN
from cmapps_rnn.train import train

MAX_RUL = 125
CURRENT_DIR = Path(__file__).resolve().parent

#PARAMETERS TO SWEEP: hidden_dim (rnn), n_layers (rnn), seq_len (parse_data), learning_rate (train)
#other sweepable params: epochs (train), batch_size (train), regularization (train?)

def create_rnn(model_name="rul.pt", hidden_dim=32, n_layers=1, epochs=30, batch_size=256, learning_rate=1e-3, seq_len=50):
    train_dir = CURRENT_DIR / 'cmapps_data' / 'train'

    # Find all four training files
    train_files = sorted(train_dir.glob('train_FD*.txt'))

    #1D lsit of 3D input/target outputs of process_cmapps_data
    all_inputs = [] #all input windows
    all_targets = [] #all target windows
    
    for file in train_files:
        train_input, train_target = process_cmapss_data(train_file=file, seq_len=seq_len,  is_train=True)
        all_inputs.append(train_input)
        all_targets.append(train_target)

    #concats the sub arrays of all_inputs/all_targets into a 3d array
    train_input = np.concatenate(all_inputs, axis=0)
    train_target = np.concatenate(all_targets, axis=0)

    # Standardize using combined training statistics
    mean = train_input.reshape(-1, train_input.shape[-1]).mean(axis=0)
    std = train_input.reshape(-1, train_input.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    train_input = (train_input - mean) / std

    # create the rnn with constructor inputs
    rnn = RNN(input_size=21, output_size=1, hidden_dim=hidden_dim, n_layers=n_layers)

    # Train rnn
    train(rnn, train_input, train_target, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    # Save model and normalization statistics
    model_dir = CURRENT_DIR / 'models'
    save_path = model_dir / model_name

    torch.save({
        "model": rnn.state_dict(), #saves all learned params
        "mean": mean,
        "std": std,
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seq_len": seq_len
        }
    }, save_path)