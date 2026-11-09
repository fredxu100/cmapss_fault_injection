import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MAX_RUL = 125

def train(rnn, input_windows, target_windows, epochs=30, batch_size=256, learning_rate=1e-3):
    input_tensors = torch.tensor(input_windows, dtype=torch.float32)
    target_tensors = torch.tensor(target_windows / MAX_RUL, dtype=torch.float32).unsqueeze(1)  # scale to [0, 1]
    loader = DataLoader(TensorDataset(input_tensors, target_tensors), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)
    rnn.train()
    
    for epoch in range(epochs): 
        total_loss, n = 0.0, 0
        for input_batch, target_batch in loader:
            pred, _ = rnn(input_batch, None)
            loss = criterion(pred, target_batch)

            optimizer.zero_grad()   # clear gradients from the previous batch
            loss.backward()         # backprop: compute d(loss)/d(weights)
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)  # cap exploding gradients (common in RNNs)
            optimizer.step()        # update weights

            total_loss += loss.item() * len(input_batch)
            n += len(input_batch)

        rmse_cycles = (total_loss / n) ** 0.5 * MAX_RUL
        print(f"Epoch {epoch+1:02d}/{epochs}  train RMSE: {rmse_cycles:.2f} cycles")