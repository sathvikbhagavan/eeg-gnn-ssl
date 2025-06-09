from pathlib import Path
import numpy as np
import pandas as pd
from seiz_eeg.dataset import EEGDataset
import os
import random
import argparse
import torch
from torch.utils.data import DataLoader
import utils
from model.model import DCRNNModel_classification
from constants import INCLUDED_CHANNELS

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

def main(args):

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Seed everything for reproduction of results
    seed_everything(1)

    # You might need to change this according to where you store the data folder
    # Inside your data folder, you should have the following structure:
    # data
    # ├── train
    # │   ├── signals/
    # │   ├── segments.parquet
    # │-- test
    #     ├── signals/
    #     ├── segments.parquet

    DATA_ROOT = Path(args.data_path)

    # Load the test split.
    clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")

    dataset_te = EEGDataset(
        clips_te,  # Your test clips variable
        signals_root=DATA_ROOT / "test",  # Update this path if your test signals are stored elsewhere
        signal_transform=utils.fft_filtering,  # Frequency domain transformation
        prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )    

    # Initialize the data loaders
    batch_size = args.batch_size
    loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

    # Best Configuration of both Correlation and Distance Graph defined below.
    # Initialize the model parameters.
    num_nodes = 19 # Number of electrodes.
    rnn_units = 64 # The hidden variable dimension of gru.
    num_rnn_layers = 2 # Number of gru layers.
    input_dim = 1 # Each electrode has one dimension.
    num_classes = 1 # Seizure or not.
    max_diffusion_step = 2 # How deep the diffusion walk should be.
    dcgru_activation = 'tanh'
    dropout = 0 # Amount of dropout.
    lr = 1e-3 # Learning rate.

    # Distance Graphs are undirected, hence we chose 'laplacian' for diffusion steps.
    if args.graph_type == 'distance':
        filter_type = 'laplacian'
    # Correlation Graphs are directed, hence we chose 'bidirectional random walk' for diffusion steps.
    elif args.graph_type == 'correlation':
        filter_type = 'dual_random_walk'

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

    # Load the best model from args.model_ckpt_path
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))  # Load the trained model weights
    model.to(device)

    # Generate the model's predictions on test set.
    if args.graph_type == 'distance':
        inference_distance_graph(model, loader_te, args, device)
    elif args.graph_type == 'correlation':
        inference_correlation_graph(model, loader_te, args, device)

def inference_distance_graph(model, loader_te, args, device):
    
    # Turn on the model's evaluation mode.
    model.eval()

    # Compute the Distance Adjacency Matrix and make it sparse using threshold 0.9
    thresh = 0.9
    dist_df = pd.read_csv('distances_3d.csv')
    A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

    # Compute the supports (ChebNet graph conv)
    filter_type = 'laplacian'
    supports = utils.compute_supports(A, filter_type)
    supports = [support.to(device) for support in supports]

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch

            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().to(device)

            # Perform the forward pass to get the model's output logits
            seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
            logits = model(x_batch, seq_lengths, supports)

            # Convert logits to predictions.
            # 0.5 threshold used for binary classification.
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission.csv", index=False)
    print("Kaggle submission file generated: submission.csv")

def inference_correlation_graph(model, loader_te, args, device):
    
    # Turn on the model's evaluation mode.
    model.eval()
    
    filter_type = 'dual_random_walk'

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch

            # Compute on the fly Correlation Adjacency matrix and supports (Bidirectional random walk)
            A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
            supports = utils.compute_supports(A, filter_type)
            supports = [support.to(device) for support in supports]

            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().to(device)

            # Perform the forward pass to get the model's output logits
            seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
            logits = model(x_batch, seq_lengths, supports)

            # Convert logits to predictions.
            # 0.5 threshold used for binary classification.
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission.csv", index=False)
    print("Kaggle submission file generated: submission.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')

    parser.add_argument('--data_path', type=str, default='') # directory where the eeg test and train data is present.
    parser.add_argument('--batch_size', type=int, default=128) # Batch size to be used.
    parser.add_argument('--graph_type', type=str, default='distance') # What type of adjacency matrix you want.
    parser.add_argument('--model_ckpt_path', type=str, default='') # Directory from where you want to load your pretrained model checkpoint.

    args = parser.parse_args()
    main(args)
