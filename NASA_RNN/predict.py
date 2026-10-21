from pathlib import Path
import numpy as np
import torch

from dataset import process_cmapss_data
from RNN import RNN

DATA = Path(__file__).resolve().parent / "cmapps_data"
MAX_RUL = 125

# load model + the train-set normalization stats
ckpt = torch.load("rnn_rul.pt", map_location="cpu", weights_only=False)
model = RNN(input_size=21, output_size=1, hidden_dim=32, n_layers=1)
model.load_state_dict(ckpt["model"])
model.eval()

# last 50-cycle window of each test engine, shape (100, 50, 21)
x, actual = process_cmapss_data(DATA / "test_FD001.txt",
                                rul_path=DATA / "RUL_FD001.txt",
                                is_train=False)
x = (x - ckpt["mean"]) / ckpt["std"]

with torch.no_grad():
    out, _ = model(torch.tensor(x, dtype=torch.float32))
preds = out.squeeze(1).numpy() * MAX_RUL

for i in range(10):
    print(f"Engine {i+1}: predicted {preds[i]:.1f}, actual {min(actual[i], MAX_RUL):.0f}")