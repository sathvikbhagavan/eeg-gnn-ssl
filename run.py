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

    clips_tr = pd.read_parquet("segments_train.parquet")
    clips_va = pd.read_parquet("segments_val.parquet")
    clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")

    dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    signal_transform=utils.fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
    )

    dataset_va = EEGDataset(
        clips_va,
        signals_root=DATA_ROOT / "train",
        signal_transform=utils.fft_filtering,
        prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
    )

    dataset_te = EEGDataset(
        clips_te,  # Your test clips variable
        signals_root=DATA_ROOT / "test",  # Update this path if your test signals are stored elsewhere
        signal_transform=utils.fft_filtering,  # You can change or remove the signal_transform as needed
        prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )    

    batch_size = args.batch_size
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

    num_nodes = 19
    rnn_units = 64
    num_rnn_layers = 2
    input_dim = 1
    num_classes = 1
    max_diffusion_step = 2
    dcgru_activation = 'tanh'
    dropout = 0.2

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
            "lr": args.lr_init,
            "epochs": args.num_epochs,
            "batch_size": batch_size
        })

    if args.graph_type == 'distance':
        filter_type = 'laplacian'
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

    if args.graph_type == 'distance':
        train_distance_graph(model, loader_tr, loader_va, args, device)
    elif args.graph_type == 'correlation':
        train_correlation_graph(model, loader_tr, loader_va, args, device)

    model.load_state_dict(torch.load(args.best_ckpt_path, map_location=device))  # Load the trained model weights
    model.to(device)

    if args.graph_type == 'distance':
        evaluate_distance_graph(model, loader_te, args, device)
    elif args.graph_type == 'correlation':
        evaluate_correlation_graph(model, loader_te, args, device)

def train_distance_graph(model, loader_tr, loader_va, args, device):

    epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    best_ckpt_path = args.best_ckpt_path
    global_step = 0
    best_macrof1 = 0.0

    thresh = 0.9
    dist_df = pd.read_csv('distances_3d.csv')
    A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)
    filter_type = 'laplacian'
    supports = utils.compute_supports(A, filter_type)
    supports = [support.to(device) for support in supports]

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
            if args.wandblog:
                wandb.log({"loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        avg_loss = running_loss / len(loader_tr)
        train_losses.append(avg_loss)

        # Evaluation phase for train accuracy
        model.eval()

        with torch.no_grad():
            y_pred_all = []
            y_true_all = []
            for x_batch, y_batch in loader_va:
                x_batch = x_batch.float().unsqueeze(-1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
                logits = model(x_batch, seq_lengths, supports)
                preds = torch.sigmoid(logits) >= 0.5
                y_pred_all.append(preds)
                y_true_all.append(y_batch.bool())

        y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
        y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))
        macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")

        if args.wandblog:
            wandb.log({"loss": avg_loss, "Macrof1": macrof1, "epoch": epoch + 1})
        
        # Save model if best accuracy so far
        if macrof1 > best_macrof1:
            best_macrof1 = macrof1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

        scheduler.step()

    if args.wandblog:  
        wandb.finish()

def train_correlation_graph(model, loader_tr, loader_va, args, device):
    
    epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    best_ckpt_path = args.best_ckpt_path
    global_step = 0
    best_macrof1 = 0.0

    filter_type = 'dual_random_walk'

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in loader_tr:

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
            if args.wandblog:
                wandb.log({"loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        avg_loss = running_loss / len(loader_tr)
        train_losses.append(avg_loss)

        # Evaluation phase for train accuracy
        model.eval()

        with torch.no_grad():
            y_pred_all = []
            y_true_all = []
            for x_batch, y_batch in loader_va:

                A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
                supports = utils.compute_supports(A, filter_type)
                supports = [support.to(device) for support in supports]
            
                x_batch = x_batch.float().unsqueeze(-1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
                logits = model(x_batch, seq_lengths, supports)
                preds = torch.sigmoid(logits) >= 0.5
                y_pred_all.append(preds)
                y_true_all.append(y_batch.bool())

        y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
        y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))
        macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")

        if args.wandblog:
            wandb.log({"loss": avg_loss, "Macrof1": macrof1, "epoch": epoch + 1})
        
        # Save model if best accuracy so far
        if macrof1 > best_macrof1:
            best_macrof1 = macrof1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

        scheduler.step()

    if args.wandblog:  
        wandb.finish()

def evaluate_distance_graph(model, loader_te, args, device):
    
    model.eval()

    thresh = 0.9
    dist_df = pd.read_csv('distances_3d.csv')
    A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)
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
            # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission.csv", index=False)
    print("Kaggle submission file generated: submission.csv")

def evaluate_correlation_graph(model, loader_te, args, device):
    
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

            A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
            supports = utils.compute_supports(A, filter_type)
            supports = [support.to(device) for support in supports]

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
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission.csv", index=False)
    print("Kaggle submission file generated: submission.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')

    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--wandblog', type=int, default=0)
    parser.add_argument('--graph_type', type=str, default='distance')
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--best_ckpt_path', type=str, default='')

    args = parser.parse_args()
    main(args)