from pathlib import Path
import numpy as np
import pandas as pd
from seiz_eeg.dataset import EEGDataset
import os
import random
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    # Load the train, validation split.
    clips_tr = pd.read_parquet("segments_train.parquet")
    clips_va = pd.read_parquet("segments_val.parquet")

    dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    signal_transform=utils.fft_filtering, # Frequency domain transformation
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
    )

    dataset_va = EEGDataset(
        clips_va,
        signals_root=DATA_ROOT / "train",
        signal_transform=utils.fft_filtering, # Frequency domain transformation
        prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
    )  

    # Initialize the data loaders
    batch_size = args.batch_size
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)

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

    # Whether you want to log your plots on wandb.
    if args.wandblog:
        # Initialize Weights and Biases
        wandb.init(project="diffusion-gnn", config={
            "num_nodes": num_nodes,
            "input_dim": input_dim,
            "rnn_units": rnn_units,
            "num_rnn_layers": num_rnn_layers,
            "num_classes": num_classes,
            "max_diffusion_step": max_diffusion_step,
            "dcgru_activation": dcgru_activation,
            "adjancency_graph": args.graph_type,
            "dropout": dropout,
            "lr": lr,
            "epochs": args.num_epochs,
            "batch_size": batch_size
        })

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

    # Different training modules for distance adjacency matrix vs correlation adjacency matrix
    if args.graph_type == 'distance':
        train_distance_graph(model, loader_tr, loader_va, args, lr, device)
    elif args.graph_type == 'correlation':
        train_correlation_graph(model, loader_tr, loader_va, args, lr, device)

def train_distance_graph(model, loader_tr, loader_va, args, lr, device):

    epochs = args.num_epochs # Total epochs.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam Optimizer used.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) # Learning rate decay.
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss.

    train_losses = []
    best_ckpt_path = args.best_ckpt_path
    global_step = 0
    best_macrof1 = 0.0 # Store the best Macro F1 score.

    # Compute the Distance Adjacency Matrix and make it sparse using threshold 0.9
    thresh = 0.9
    dist_df = pd.read_csv('distances_3d.csv')
    A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

    # Compute the supports (ChebNet graph conv)
    filter_type = 'laplacian'
    supports = utils.compute_supports(A, filter_type)
    supports = [support.to(device) for support in supports]

    for epoch in tqdm(range(epochs), desc="Training"):

        # Turn on the model's training mode.
        model.train()
        running_loss = 0.0

        # Per epoch training loop
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
            # Log the loss to wandb.
            if args.wandblog:
                wandb.log({"loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        avg_loss = running_loss / len(loader_tr)
        train_losses.append(avg_loss)

        # Evaluation phase on validation data.

        # Turn on the model's evaluation mode.
        model.eval()

        with torch.no_grad():
            y_pred_all = []
            y_true_all = []
            for x_batch, y_batch in loader_va:
                x_batch = x_batch.float().unsqueeze(-1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
                logits = model(x_batch, seq_lengths, supports)

                # 0.5 threshold used for binary classification.
                preds = torch.sigmoid(logits) >= 0.5
                y_pred_all.append(preds)
                y_true_all.append(y_batch.bool())

        # Calculate the MacroF1 score on the validation set.
        y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
        y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))
        macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')

        # Track the validation macrof1 to know overfitting or underfitting.
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")

        if args.wandblog:
            wandb.log({"loss": avg_loss, "Macrof1": macrof1, "epoch": epoch + 1})
        
        # Save model if best accuracy so far
        if macrof1 > best_macrof1:
            best_macrof1 = macrof1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

        # Learning Rate Decay Step.
        scheduler.step()

    if args.wandblog:  
        wandb.finish()

def train_correlation_graph(model, loader_tr, loader_va, args, lr, device):
    
    epochs = args.num_epochs # Total epochs.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam Optimizer used.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) # Learning rate decay.
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss.

    train_losses = []
    best_ckpt_path = args.best_ckpt_path
    global_step = 0
    best_macrof1 = 0.0 # Store the best Macro F1 score.

    filter_type = 'dual_random_walk'

    for epoch in tqdm(range(epochs), desc="Training"):

        # Turn on the model's training mode.
        model.train()
        running_loss = 0.0

        # Per epoch training loop
        for x_batch, y_batch in loader_tr:

            # Compute on the fly Correlation Adjacency matrix and supports (Bidirectional random walk)
            A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
            supports = utils.compute_supports(A, filter_type)
            supports = [support.to(device) for support in supports]
            
            seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
            x_batch = x_batch.float().unsqueeze(-1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            logits = model(x_batch, seq_lengths, supports)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log the loss to wandb.
            if args.wandblog:
                wandb.log({"loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        avg_loss = running_loss / len(loader_tr)
        train_losses.append(avg_loss)

        # Evaluation phase on validation data.

        # Turn on the model's evaluation mode.
        model.eval()

        with torch.no_grad():
            y_pred_all = []
            y_true_all = []
            for x_batch, y_batch in loader_va:

                # Compute on the fly Correlation Adjacency matrix and supports (Bidirectional random walk)
                A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
                supports = utils.compute_supports(A, filter_type)
                supports = [support.to(device) for support in supports]
            
                x_batch = x_batch.float().unsqueeze(-1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
                logits = model(x_batch, seq_lengths, supports)

                # 0.5 threshold used for binary classification.
                preds = torch.sigmoid(logits) >= 0.5
                y_pred_all.append(preds)
                y_true_all.append(y_batch.bool())

        # Calculate the MacroF1 score on the validation set.
        y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
        y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))
        macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')

        # Track the validation macrof1 to know overfitting or underfitting.
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")
        
        if args.wandblog:
            wandb.log({"loss": avg_loss, "Macrof1": macrof1, "epoch": epoch + 1})
        
        # Save model if best accuracy so far
        if macrof1 > best_macrof1:
            best_macrof1 = macrof1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

        # Learning Rate Decay Step.
        scheduler.step()

    if args.wandblog:  
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')

    parser.add_argument('--data_path', type=str, default='') # directory where the eeg test and train data is present.
    parser.add_argument('--batch_size', type=int, default=128) # Batch size to be used.
    parser.add_argument('--wandblog', type=int, default=0) # Whether you want wandb logging or not.
    parser.add_argument('--graph_type', type=str, default='distance') # What type of adjacency matrix you want.
    parser.add_argument('--num_epochs', type=int, default=100) # Total number of epochs, 100 was done for the best model.
    parser.add_argument('--best_ckpt_path', type=str, default='') # Directory where you want to save your best model checkpoint.

    args = parser.parse_args()
    main(args)
