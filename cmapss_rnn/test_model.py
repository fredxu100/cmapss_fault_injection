from pathlib import Path
import numpy as np
import torch

from cmapss_rnn.parse_data import process_cmapss_data
from cmapss_rnn.RNN import RNN
from cmapss_rnn.evaluate import evaluate

MAX_RUL = 125
CURRENT_DIR = Path(__file__).resolve().parent


def test_model(model_name):
    # Load saved model and normalization statistics
    if model_name is None:
        print("ERROR: no model provided")
        return

    model_path = CURRENT_DIR /'models'/ model_name
    
    #loads previous .pt into checkpoint, dict of the torch.save
    checkpoint = torch.load(model_path, weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    hp = checkpoint["hyperparameters"]
    seq_len = hp["seq_len"]

    # Reconstruct the trained RNN
    rnn = RNN(input_size=21, output_size=1, hidden_dim=hp["hidden_dim"], n_layers=hp["n_layers"])
    rnn.load_state_dict(checkpoint['model'])
    rnn.eval()
    results = {}

    # Test all 4 datasets
    for i in range(1, 5):
        fd = f'FD00{i}'

        test_file = CURRENT_DIR / 'cmapss_data' / f'test_{fd}.txt'
        rul_file = CURRENT_DIR / 'cmapss_data' / f'RUL_{fd}.txt'

        #loads input/target windows to test
        test_input, test_target = process_cmapss_data(test_file=test_file, rul_file=rul_file, is_train=False, seq_len = seq_len)

        test_input = (test_input - mean) / std
        test_target = np.clip(test_target, 0, MAX_RUL)

        rmse, predictions = evaluate(rnn, test_input, test_target)

        results[fd] = {
            'rmse': rmse,
            'predictions': predictions,
            'actual': test_target
        }

    rmse_average = np.mean([results[fd]['rmse'] for fd in results])
    #return average rsme and results dict of dicts
    return rmse_average, results