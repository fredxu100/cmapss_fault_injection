"""
FGSM / BIM attack on models/rul_rnn1.pt  (repo: fredxu100/cmapss_fault_injection)

Place this file in the fgsm_attack/ folder of the repo (a sibling of cmapss_rnn/,
cmapss_data/, models/). Run it from fgsm_attack/ or from the repo root:
    python fgsm_attack.py
    python fgsm_attack.py --mode over --last-k 10 --eps 0.05 0.1 0.2
    python fgsm_attack\\fgsm_attack.py          (from the repo root, PowerShell)

What it matches from your code:
  * RNN: cmapss_rnn.RNN.RNN (tanh nn.RNN, batch_first, last-timestep head)
  * model output is RUL / 125 -> loss is computed on the raw RUL scale
  * test set: last seq_len cycles of each test engine, labels clipped to [0, 125]
  * input to model: (raw_sensors - mean) / std  -> attack lives in this normalised
    space, so eps is in units of each sensor's std (eps=0.1 -> 0.1 std)

Attack modes (--mode):
  any   : maximise squared RUL error (default)
  over  : push predicted RUL up   (engine looks healthier than it is - the unsafe direction)
  under : push predicted RUL down (false alarms)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of fgsm_attack/)
sys.path.insert(0, str(ROOT))                   # so `cmapss_rnn` is importable

import numpy as np
import torch

from cmapss_rnn.parse_data import process_cmapss_data
from cmapss_rnn.RNN import RNN

MAX_RUL = 125

#takes model from model folder and loads it
#RETURNS: loaded rnn with frozen params, mean/std of datasets, seq_len
def load_model(name="rul_rnn1.pt"):
    ck = torch.load(ROOT / "models" / name, weights_only=False) #loads dict from torch.save
    hp = ck["hyperparameters"] #hyperparams dict

    net = RNN(input_size=21, output_size=1, hidden_dim=hp["hidden_dim"], n_layers=hp["n_layers"]) #creates RNN shape
    net.load_state_dict(ck["model"]) #load weights/biases

    net.eval() #set to eval mode

    #Sets RNN so that NO params will be changed
    for p in net.parameters(): #all learnable parameters 
        p.requires_grad_(False) #don't calculate gradient

    #saves 21 sensor mean and std of each training dataset into tensors
    mean = torch.as_tensor(ck["mean"], dtype=torch.float32)
    std = torch.as_tensor(ck["std"], dtype=torch.float32)
    
    return net, mean, std, hp["seq_len"]

#loads RNN and input, then returns unnormalized prediction
def predict_rul(net, x):
    """x: (B, T, 21) normalised -> RUL in cycles, shape (B,)."""
    pred, _ = net(x, None)
    return pred.squeeze(1) * MAX_RUL

#defines what FGSM attack is tryna acomplish (overshoot, undershoot, max error)
#pred: predicted RNN output
#y: actual RUL value from CMAPSS dataset
def objective(pred, y, mode):
    if mode == "any":
        return ((pred - y) ** 2).sum()
    if mode == "over":
        return pred.sum()
    if mode == "under":
        return -pred.sum()
    else:
        return 0

#blackbox: select which values you wanan perturb
def build_mask(last_k, n_features=21, seq_len=40, sensors=None):
     # Allow perturbations only in the last K timesteps
    m = torch.zeros(1, seq_len, n_features)
    m[:, -last_k:, :] = 1.0

    # Optionally restrict perturbations to selected sensors
    if sensors is not None:
        sm = torch.zeros(n_features)
        sm[[s - 1 for s in sensors]] = 1.0
        m *= sm.view(1, 1, -1) #reshapes tensor into 3D

    return m #3D tensor, 1s and 0s of masking for selected timestamps and sensors


def fgsm(net, x, y, eps, mode="any", mask=None, bounds=None):
    x_adv = x.clone().requires_grad_(True)
    loss = objective(predict_rul(net, x_adv), y, mode)
    g, = torch.autograd.grad(loss, x_adv)
    step = g.sign()
    if mask is not None:
        step = step * mask
    x_adv = x + eps * step
    if bounds is not None:
        x_adv = torch.max(torch.min(x_adv, bounds[1]), bounds[0])
    return x_adv.detach()


def bim(net, x, y, eps, steps=20, mode="any", mask=None, bounds=None):
    """Iterative FGSM (PGD, L-inf ball of radius eps around x)."""
    alpha = 2.5 * eps / steps
    x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = objective(predict_rul(net, x_adv), y, mode)
        g, = torch.autograd.grad(loss, x_adv)
        step = g.sign()
        if mask is not None:
            step = step * mask
        x_adv = x_adv.detach() + alpha * step
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        if bounds is not None:
            x_adv = torch.max(torch.min(x_adv, bounds[1]), bounds[0])
    return x_adv.detach()


@torch.no_grad()
def stats(net, x, y):
    p = predict_rul(net, x)
    err = p - y
    return (err ** 2).mean().sqrt().item(), err.mean().item()


def train_bounds(fd, mean, std):
    """Per-sensor min/max of the FD's training data, in normalised space."""
    X, _ = process_cmapss_data(train_file=ROOT / "cmapss_data" / "train" / f"train_{fd}.txt",
                               is_train=True, seq_len=1)
    X = torch.as_tensor(X.reshape(-1, 21), dtype=torch.float32)
    Xn = (X - mean) / std
    return Xn.min(0).values.view(1, 1, -1), Xn.max(0).values.view(1, 1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rul_rnn1.pt")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2])
    ap.add_argument("--mode", default="any", choices=["any", "over", "under"])
    ap.add_argument("--last-k", type=int, default=None, help="only perturb last k timesteps")
    ap.add_argument("--sensors", nargs="+", type=int, default=None, help="1-indexed sensors to perturb")
    ap.add_argument("--bim-steps", type=int, default=20)
    ap.add_argument("--clamp", action="store_true", help="clip to training min/max per sensor")
    a = ap.parse_args()

    net, mean, std, T = load_model(a.model)
    mask = None
    if a.last_k is not None or a.sensors is not None:
        mask = build_mask(a.last_k or T, 21, T, a.sensors)

    for fd in a.fds:
        Xr, y = process_cmapss_data(test_file=ROOT / "cmapss_data" / f"test_{fd}.txt",
                                    rul_file=ROOT / "cmapss_data" / f"RUL_{fd}.txt",
                                    is_train=False, seq_len=T)
        x = (torch.as_tensor(Xr, dtype=torch.float32) - mean) / std
        y = torch.as_tensor(np.clip(y, 0, MAX_RUL), dtype=torch.float32)
        bounds = train_bounds(fd, mean, std) if a.clamp else None

        r0, b0 = stats(net, x, y)
        print(f"\n{fd}  mode={a.mode}  n={len(y)}   clean RMSE {r0:6.2f}  bias {b0:+6.2f}")
        for eps in a.eps:
            xf = fgsm(net, x, y, eps, a.mode, mask, bounds)
            xb = bim(net, x, y, eps, a.bim_steps, a.mode, mask, bounds)
            rf, bf = stats(net, xf, y)
            rb, bb = stats(net, xb, y)
            print(f"  eps={eps:<6} FGSM RMSE {rf:7.2f} (bias {bf:+7.2f})   "
                  f"BIM-{a.bim_steps} RMSE {rb:7.2f} (bias {bb:+7.2f})")


if __name__ == "__main__":
    main()