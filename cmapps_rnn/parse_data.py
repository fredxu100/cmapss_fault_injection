import numpy as np
import pandas as pd

MAX_RUL = 125 #normalization value
COLS = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + [ f'sensor_{i}' for i in range(1, 22)]
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]

def process_cmapss_data(train_file=None, test_file=None, rul_file=None, seq_len=50, is_train=True):
    if is_train:
        train_file = pd.read_csv(train_file, sep=r'\s+', header=None, names=COLS)
        return parse_train(train_file, seq_len)
    else: 
        test_file = pd.read_csv(test_file, sep=r'\s+', header=None, names=COLS)
        return parse_test(test_file, rul_file, seq_len)


def parse_train(df, seq_len):
    input_windows = []
    target_windows = []

    for unit_id, group in df.groupby('unit_id'):
        max_cycle = group['cycle'].max()

        # Calculate capped RUL for every cycle
        rul = np.clip(max_cycle - group['cycle'], 0, MAX_RUL)

        data_values = group[SENSOR_COLS].values
        labels = rul.values

        # Generate sliding windows
        for i in range(len(group) - seq_len + 1):
            window = data_values[i:i + seq_len]
            target = labels[i + seq_len - 1]

            input_windows.append(window)
            target_windows.append(target)

    return np.array(input_windows), np.array(target_windows)


def parse_test(df, rul_file, seq_len):
    input_windows = []
    target_windows = []

    # Load actual RUL values
    rul_df = pd.read_csv(rul_file, header=None, names=['rul'])

    for idx, (unit_id, group) in enumerate(df.groupby('unit_id')):
        data_values = group[SENSOR_COLS].values

        # Extract final window, padding if necessary
        if len(data_values) >= seq_len:
            window = data_values[-seq_len:]
        else:
            pad_width = seq_len - len(data_values)
            window = np.pad(
                data_values,
                ((pad_width, 0), (0, 0)),
                mode='edge'
            )

        input_windows.append(window)
        target_windows.append(rul_df.iloc[idx]['rul'])

    return np.array(input_windows), np.array(target_windows)