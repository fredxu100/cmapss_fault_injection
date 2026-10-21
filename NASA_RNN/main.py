from pathlib import Path
import numpy as np
import torch

from dataset import process_cmapss_data
from RNN import RNN
from train import train, evaluate

MAX_RUL = 125

def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    train_file = CURRENT_DIR / 'cmapps_data' / 'train_FD001.txt'
    test_file = CURRENT_DIR / 'cmapps_data' / 'test_FD001.txt'
    rul_file = CURRENT_DIR / 'cmapps_data' / 'RUL_FD001.txt'

    train_input, train_target = process_cmapss_data(train_file, max_rul=MAX_RUL, is_train=True)
    test_input, test_target = process_cmapss_data(test_file, rul_path=rul_file,
                                                  max_rul=MAX_RUL, is_train=False)
    test_target = np.clip(test_target, 0, MAX_RUL)  # same cap as training labels

    # Standardize per sensor using TRAIN statistics only
    mean = train_input.reshape(-1, train_input.shape[-1]).mean(axis=0)
    std = train_input.reshape(-1, train_input.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0  # constant sensors
    train_input = (train_input - mean) / std
    test_input = (test_input - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training windows: {len(train_input)}, device: {device}")

    rnn = RNN(input_size=21, output_size=1, hidden_dim=32, n_layers=1)
    train(rnn, train_input, train_target, epochs=30, device=device, max_rul=MAX_RUL)

    rmse, preds = evaluate(rnn, test_input, test_target, device=device, max_rul=MAX_RUL)
    print(f"Test RMSE: {rmse:.2f} cycles")
    for i in range(5):
        print(f"Engine {i+1}: predicted {preds[i]:.1f}, actual {test_target[i]:.0f}")

    torch.save({"model": rnn.state_dict(), "mean": mean, "std": std}, "rnn_rul.pt")

if __name__ == '__main__':
    main()