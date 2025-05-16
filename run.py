from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model import DCRNNModel_classification

import wandb
import utils


"""# The data

We model *segments* of brain activity, which correspond to windows of a longer *session* of EEG recording.

These segments, and their labels, are described in the `segments.parquet` files, which can be directly loaded with `pandas`.
"""

# You might need to change this according to where you store the data folder
# Inside your data folder, you should have the following structure:
# data
# ├── train
# │   ├── signals/
# │   ├── segments.parquet
# │-- test
#     ├── signals/
#     ├── segments.parquet

data_path = "/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/data/train"

DATA_ROOT = Path(data_path)
# DATA_ROOT_TEST = Path("/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/data/test")

clips_tr = pd.read_parquet(DATA_ROOT / "train/segments_train.parquet")
clips_va = pd.read_parquet(DATA_ROOT / "train/segments_val.parquet")
# clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
# clips_te = pd.read_parquet(DATA_ROOT_TEST / "test/segments.parquet")


"""## Loading the signals

For convenience, the `EEGDataset class` provides functionality for loading each segment and its label as `numpy` arrays.

You can provide an optional `signal_transform` function to preprocess the signals. In the example below, we have two bandpass filtering functions, which extract frequencies between 0.5Hz and 30Hz which are used in seizure analysis literature:

The `EEGDataset` class also allows to load all data in memory, instead of reading it from disk at every iteration. If your compute allows it, you can use `prefetch=True`.
"""
bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)

def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()

def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))
    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

# You can change the signal_transform, or remove it completely
dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    signal_transform=fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)
dataset_va = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT / "train",
    signal_transform=fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)

"""## Compatibility with PyTorch

The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
"""
def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1)
batch_size = 128
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

INCLUDED_CHANNELS = [
    'FP1',
    'FP2',
    'F3',
    'F4',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'F7',
    'F8',
    'T3',
    'T4',
    'T5',
    'T6',
    'FZ',
    'CZ',
    'PZ'
]

thresh = 0.9
dist_df = pd.read_csv('/home/chaurasi/networkml/eeg-gnn-ssl/distances_3d.csv')
A, sensor_id_to_ind = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

def _compute_supports(adj_mat, filter_type):
    """
    Compute supports
    """
    supports = []
    supports_mat = []
    if filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(
            utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif filter_type == "random_walk":  # Forward random walk
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
    elif filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(
            utils.calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        supports.append(torch.FloatTensor(support.toarray()))
    return supports

# --------------------------
# Model, Loss, Optimizer
# --------------------------
num_nodes = 19
rnn_units = 64
num_rnn_layers = 2
input_dim = 1
num_classes = 1
max_diffusion_step = 2
dcgru_activation = 'tanh'
filter_type = 'laplacian'
dropout = 0.2
lr = 1e-4
epochs = 100

supports = _compute_supports(A, filter_type)
supports = [support.to(device) for support in supports]

# Initialize Weights and Biases
wandb.init(project="diffusion-gnn", config={
    "num_nodes": num_nodes,
    "input_dim": input_dim,
    "rnn_units": rnn_units,
    "num_rnn_layers": num_rnn_layers,
    "num_classes": num_classes,
    "max_diffusion_step": max_diffusion_step,
    "dcgru_activation": dcgru_activation,
    "filter_type": filter_type,
    "dropout": dropout,
    "lr": lr,
    "epochs": epochs,
    "batch_size": batch_size
})

model = DCRNNModel_classification(
    input_dim=input_dim,
    num_nodes=num_nodes,
    num_classes=num_classes,
    num_rnn_layers=num_rnn_layers,
    rnn_units=rnn_units,
    max_diffusion_step=max_diffusion_step,
    dcgru_activation=dcgru_activation,
    filter_type=filter_type,
    dropout=dropout,
    device=device
).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

criterion = nn.BCEWithLogitsLoss()

# --------------------------
# Training Loop
# --------------------------
train_losses = []
ckpt_path = os.path.join(wandb.run.dir, "best_diffusion_gnn_model.pth")

global_step = 0
best_acc = 0.0

for epoch in tqdm(range(epochs), desc="Training"):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader_tr:
        seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
        x_batch = x_batch.float().unsqueeze(-1).to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        logits = model(x_batch, seq_lengths, supports)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"loss": loss.item(), "global_step": global_step})
        print(f"Step {global_step}, Loss: {loss.item():.4f}")
        global_step += 1

    avg_loss = running_loss / len(loader_tr)
    train_losses.append(avg_loss)

    # Evaluation phase for train accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader_va:
            x_batch = x_batch.float().unsqueeze(-1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
            logits = model(x_batch, seq_lengths, supports)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch.bool()).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    wandb.log({"loss": avg_loss, "accuracy": acc, "epoch": epoch + 1})
    
    # Save model if best accuracy so far
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best model saved with accuracy: {acc:.4f}")

    scheduler.step()
    
wandb.finish()
