import numpy as np
import pandas as pd

#PROCESSING TXT FILES INTO TENSORS
def process_cmapss_data(file_path, rul_path=None, seq_len=50, max_rul=125, is_train=True):
    cols = ['unit_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)] #all columns
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)] #sensor columns
    # Load raw space-separated file
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=cols)

    all_inputs = [] #ALL INPUT WINDOWS
    all_targets = [] #ALL TARGET WINDOWS

    # USE FOR TRAINING DATA FILES
    if is_train: 
        for unit_id, group in df.groupby('unit_id'):
            max_cycle = group['cycle'].max()
            
            # Compute RUL for each cycle, capped at max_rul (piecewise linear degradation model)
            rul = max_cycle - group['cycle'] #RUL = number cycles - current cycle
            rul = np.clip(rul, 0, max_rul)
            
            data_values = group[sensor_cols].values
            labels = rul.values
            
            # Slide a window of length `seq_len` over this engine's data
            for i in range(len(group) - seq_len + 1):
                window = data_values[i : i + seq_len]
                target = labels[i + seq_len - 1] # RUL at the end of the window
                
                all_inputs.append(window)
                all_targets.append(target)
                
    # USE FOR TESTING DATA FILES
    else: 
        # TEST DATA: Engines are cut off mid-life. RUL file provides the exact target per engine.
        rul_df = pd.read_csv(rul_path, header=None, names=['rul'])
        
        for idx, (unit_id, group) in enumerate(df.groupby('unit_id')):
            data_values = group[sensor_cols].values
            
            # For testing, we only care about the *final* available window of the engine
            if len(data_values) >= seq_len:
                window = data_values[-seq_len:]
            else:
                # If an engine has fewer cycles than seq_len, pad or handle appropriately
                pad_width = seq_len - len(data_values)
                window = np.pad(data_values, ((pad_width, 0), (0, 0)), mode='edge')
                
            all_inputs.append(window)
            all_targets.append(rul_df.iloc[idx]['rul'])
            
    return np.array(all_inputs), np.array(all_targets)
