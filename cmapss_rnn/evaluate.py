import torch

MAX_RUL = 125

@torch.no_grad()
def evaluate(rnn, input_windows, target_windows):
    rnn.eval()
    input_tensor = torch.tensor(input_windows, dtype=torch.float32)
    pred, _ = rnn(input_tensor, None)
    pred = pred.squeeze(1).numpy() * MAX_RUL #converts 2D torch tensor into 1D numpy array
    rmse = ((pred - target_windows) ** 2).mean() ** 0.5
    return rmse, pred
