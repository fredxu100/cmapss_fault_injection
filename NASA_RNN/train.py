import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train(rnn, train_input, train_target, epochs=30, batch_size=256, max_rul=125, device="cpu"):
    x = torch.tensor(train_input, dtype=torch.float32)
    y = torch.tensor(train_target / max_rul, dtype=torch.float32).unsqueeze(1)  # scale to [0, 1]
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    rnn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

    for epoch in range(epochs):
        rnn.train()
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred, _ = rnn(xb, None)
            loss = criterion(pred, yb)

            optimizer.zero_grad()   # clear gradients from the previous batch
            loss.backward()         # backprop: compute d(loss)/d(weights)
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)  # cap exploding gradients (common in RNNs)
            optimizer.step()        # update weights

            total_loss += loss.item() * len(xb)
            n += len(xb)

        rmse_cycles = (total_loss / n) ** 0.5 * max_rul
        print(f"Epoch {epoch+1:02d}/{epochs}  train RMSE: {rmse_cycles:.2f} cycles")

@torch.no_grad()
def evaluate(rnn, inputs, targets, device="cpu", max_rul=125):
    rnn.eval()
    rnn.to(device)
    x = torch.tensor(inputs, dtype=torch.float32).to(device)
    pred, _ = rnn(x, None)
    pred = pred.squeeze(1).cpu().numpy() * max_rul
    rmse = ((pred - targets) ** 2).mean() ** 0.5
    return rmse, pred