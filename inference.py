from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import pandas as pd
from pathlib import Path
import re
from model.model import DCRNNModel_classification
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_TEST = Path("/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/data/test")
clips_te = pd.read_parquet(DATA_ROOT_TEST / "test/segments.parquet")

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

# Create test dataset
dataset_te = EEGDataset(
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT_TEST / "test",  # Update this path if your test signals are stored elsewhere
    signal_transform=fft_filtering,  # You can change or remove the signal_transform as needed
    prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
)

# Create DataLoader for the test dataset
loader_te = DataLoader(dataset_te, batch_size=64, shuffle=False)

# Define node list (in order, matching your image)
nodes = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Define edge list (bidirectional edges for undirected graph)
edges = [
    ('Fp1', 'F7'), ('Fp1', 'F3'), ('Fp1', 'Fp2'),
    ('Fp2', 'F4'), ('Fp2', 'F8'),
    ('F7', 'F3'), ('F3', 'Fz'), ('Fz', 'F4'), ('F4', 'F8'),
    ('F7', 'T3'), ('F3', 'C3'), ('Fz', 'Cz'), ('F4', 'C4'), ('F8', 'T4'),
    ('T3', 'C3'), ('C3', 'Cz'), ('Cz', 'C4'), ('C4', 'T4'),
    ('T3', 'T5'), ('C3', 'P3'), ('Cz', 'Pz'), ('C4', 'P4'), ('T4', 'T6'),
    ('T5', 'P3'), ('P3', 'Pz'), ('Pz', 'P4'), ('P4', 'T6'),
    ('T5', 'O1'), ('P3', 'O1'), ('Pz', 'O1'), ('Pz', 'O2'), ('P4', 'O2'), ('T6', 'O2')
]

# Create a mapping from node names to indices
node_idx = {node: i for i, node in enumerate(nodes)}

# Convert edge list to index tensors
edge_index = torch.tensor([[node_idx[u], node_idx[v]] for u, v in edges] +
                          [[node_idx[v], node_idx[u]] for u, v in edges], dtype=torch.long).t().to(device)

A = torch.zeros((len(nodes), len(nodes)), dtype=torch.float32)
for u, v in edges:
    i, j = node_idx[u], node_idx[v]
    A[i, j] = 1
    A[j, i] = 1  # undirected

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
rnn_units = 128
num_rnn_layers = 2
input_dim = 1
num_classes = 1
max_diffusion_step = 2
dcgru_activation = 'tanh'
filter_type = 'dual_random_walk'
dropout = 0.2

supports = _compute_supports(A, filter_type)
supports = [support.to(device) for support in supports]

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
)

model.load_state_dict(torch.load("/work/cvlab/students/bhagavan/eeg-gnn-ssl/wandb/run-20250504_190416-fsdmqeew/files/best_diffusion_gnn_model.pth", map_location=device))  # Load the trained model weights
model.to(device)
model.eval()

# Lists to store sample IDs and predictions
all_predictions = []
all_ids = []

def clean_underscores(s):
    # Step 1: Replace all double underscores (or more) with a temporary marker
    s = re.sub(r'__+', lambda m: '<<UND>>' * (len(m.group()) // 2), s)

    # Step 2: Remove remaining single underscores
    s = re.sub(r'_', '', s)

    # Step 3: Replace all temporary markers back with a single underscore each
    s = s.replace('<<UND>>', '_')

    return s

# Disable gradient computation for inference
with torch.no_grad():
    for batch in loader_te:
        # Assume each batch returns a tuple (x_batch, sample_id)
        # If your dataset does not provide IDs, you can generate them based on the batch index.
        x_batch, x_ids = batch
        actual_x_ids = x_ids
        x_ids = [clean_underscores(x_id) for x_id in x_ids]  # Clean the IDs

        # Move the input data to the device (GPU or CPU)
        x_batch = x_batch.float().to(device)

        # Perform the forward pass to get the model's output logits
        seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
        logits = model(x_batch, seq_lengths, supports)

        # Convert logits to predictions.
        # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

        # Append predictions and corresponding IDs to the lists
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(actual_x_ids))

# Create a DataFrame for Kaggle submission with the required format: "id,label"
submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

# Save the DataFrame to a CSV file without an index
submission_df.to_csv("submission.csv", index=False)
print("Kaggle submission file generated: submission.csv")
