"""
Hyperparameter sweep for the CMAPSS RUL RNN.

Run from the folder that contains the NASA_RNN package:
    python -m NASA_RNN.sweep_rnn

Output:
    models/rul_rnn1.pt ... rul_rnn5.pt   top-5 models (1 = lowest average test RMSE)
    models/sweep_results.csv             one row per trial (all params + RMSE)
"""
import contextlib
import csv
import io
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch

from create_rnn import create_rnn
from test_model import test_model

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR / 'models'

# =====================================================================
#                         SWEEP CONFIGURATION
#            (everything you normally need to edit is here)
# =====================================================================

# Each key must match a create_rnn() argument. Add/remove values freely.
# Total grid size = product of the list lengths (3 * 2 * 2 * 3 = 36 runs here).

SWEEP_GRID = {
    # Model capacity
    "hidden_dim": [16, 32, 48, 64, 96, 128],

    # Recurrent depth
    "n_layers": [1, 2, 3],

    # Input sequence length
    "seq_len": [20, 30, 40, 50, 60],

    # Optimizer learning rate
    "learning_rate": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
}

# create_rnn() arguments held constant for every trial.
# Move any of these into SWEEP_GRID to sweep them too.
FIXED_PARAMS = {"epochs": 2, "batch_size": 256}

SEARCH_MODE = "grid"        # "grid" = every combination, "random" = random subset of the grid
N_RANDOM_TRIALS = 20        # only used when SEARCH_MODE == "random"

TOP_K = 5                   # number of models kept as rul_rnn1.pt ... rul_rnn{TOP_K}.pt
SEED = 42                   # same seed before every trial -> differences come from the hyperparameters
SUPPRESS_TRAIN_OUTPUT = True  # hide the per-epoch prints from train() during the sweep

RESULTS_CSV = MODEL_DIR / 'sweep_results.csv'
FD_NAMES = ['FD001', 'FD002', 'FD003', 'FD004']

# =====================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextlib.contextmanager
def quiet(enabled):
    if enabled:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def build_trials():
    """Expands SWEEP_GRID into a list of hyperparameter dicts."""
    keys = list(SWEEP_GRID.keys())
    trials = [dict(zip(keys, values)) for values in itertools.product(*SWEEP_GRID.values())]

    if SEARCH_MODE == "random":
        random.Random(SEED).shuffle(trials)
        trials = trials[:N_RANDOM_TRIALS]
    elif SEARCH_MODE != "grid":
        raise ValueError(f"SEARCH_MODE must be 'grid' or 'random', got {SEARCH_MODE!r}")
    return trials


def final_name(rank):
    return f"rul_rnn{rank}.pt"


def finalize_leaderboard(leaderboard):
    """Renames the surviving temp files to rul_rnn1.pt ... in rank order.
    Returns {temp_name: final_name}. Replaces any rul_rnn*.pt from a previous sweep."""
    if not leaderboard:
        return {}  # nothing succeeded: leave any existing rul_rnn*.pt alone

    for rank in range(1, TOP_K + 1):
        (MODEL_DIR / final_name(rank)).unlink(missing_ok=True)

    mapping = {}
    for rank, entry in enumerate(leaderboard, start=1):
        (MODEL_DIR / entry["file"]).replace(MODEL_DIR / final_name(rank))
        mapping[entry["file"]] = final_name(rank)
    return mapping


def write_csv(log, mapping):
    fields = (["trial"] + list(SWEEP_GRID.keys()) + ["avg_rmse"] + [f"rmse_{fd}" for fd in FD_NAMES]
              + ["seconds", "status", "saved_as"])
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in log:
            row = dict(row)
            row["saved_as"] = mapping.get(row.pop("file"), "")
            writer.writerow(row)


def run_sweep():
    """Runs the sweep. Returns (all_rmse, leaderboard):
       all_rmse    - list with the average test RMSE of every trial (nan if a trial failed),
                     in the same order as the trials
       leaderboard - best-to-worst list of the TOP_K entries {rmse, params, file}
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # clear leftovers from an interrupted earlier sweep
    for stale in MODEL_DIR.glob('_sweep_trial_*.pt'):
        stale.unlink()

    trials = build_trials()
    all_rmse = []
    log = []
    leaderboard = []   # sorted best -> worst, never longer than TOP_K
    sweep_start = time.time()

    print(f"Sweeping {len(trials)} configurations ({SEARCH_MODE} search), keeping top {TOP_K}\n")

    try:
        for idx, params in enumerate(trials, start=1):
            tmp_name = f"_sweep_trial_{idx:03d}.pt"
            row = {"trial": idx, **params, "file": tmp_name}
            t0 = time.time()

            set_seed(SEED)
            try:
                with quiet(SUPPRESS_TRAIN_OUTPUT):
                    create_rnn(model_name=tmp_name, **FIXED_PARAMS, **params)
                    rmse, results = test_model(tmp_name)
                rmse = float(rmse)
                if not np.isfinite(rmse):
                    raise ValueError("RMSE is NaN/inf (training diverged)")
            except Exception as e:
                (MODEL_DIR / tmp_name).unlink(missing_ok=True)
                all_rmse.append(float('nan'))
                row.update(avg_rmse="", seconds=round(time.time() - t0, 1), status=f"failed: {e}")
                log.append(row)
                print(f"[{idx}/{len(trials)}] {params}  FAILED: {e}")
                continue

            all_rmse.append(rmse)
            row.update(avg_rmse=round(rmse, 3), seconds=round(time.time() - t0, 1), status="ok")
            for fd in FD_NAMES:
                row[f"rmse_{fd}"] = round(float(results[fd]['rmse']), 3)
            log.append(row)

            # keep only the best TOP_K model files on disk
            if len(leaderboard) < TOP_K or rmse < leaderboard[-1]["rmse"]:
                leaderboard.append({"rmse": rmse, "params": params, "file": tmp_name})
                leaderboard.sort(key=lambda e: e["rmse"])
                if len(leaderboard) > TOP_K:
                    dropped = leaderboard.pop()
                    (MODEL_DIR / dropped["file"]).unlink(missing_ok=True)
            else:
                (MODEL_DIR / tmp_name).unlink(missing_ok=True)

            elapsed = time.time() - sweep_start
            eta = elapsed / idx * (len(trials) - idx)
            print(f"[{idx}/{len(trials)}] {params}  avg RMSE: {rmse:.2f}  "
                  f"(best so far: {leaderboard[0]['rmse']:.2f}, ETA {eta / 60:.1f} min)")

    except KeyboardInterrupt:
        print("\nSweep interrupted - saving what has been found so far.")

    finally:
        # runs on normal exit, Ctrl+C, or a crash, so the top-K files are always named correctly
        mapping = finalize_leaderboard(leaderboard)
        if log:
            write_csv(log, mapping)

    print("\n================ TOP MODELS ================")
    for rank, entry in enumerate(leaderboard, start=1):
        print(f"{final_name(rank)}  RMSE {entry['rmse']:.2f}  {entry['params']}")
    print(f"\nFull results: {RESULTS_CSV}")

    return all_rmse, leaderboard


if __name__ == "__main__":
    run_sweep()