from scipy.sparse import linalg
import torch
import numpy as np
import scipy.sparse as sp
from scipy.signal import correlate

def get_adjacency_matrix(distance_df, sensor_ids, dist_k=0.9):
    """
    Args:
        distance_df: data frame with three columns: [from, to, distance].
        sensor_ids: list of sensor ids.
        dist_k: threshold for graph sparsity
    Returns:
        adj_mx: adj
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.    
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[dist_mx > dist_k] = 0
   
    return adj_mx

def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array
        y: 1D array
        mode: 'valid', 'full' or 'same',
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = torch.sum(torch.absolute(x) ** 2)
    cyy0 = torch.sum(torch.absolute(y) ** 2)
    if normalize and (cxx0 != 0) and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr /= scale
    return xcorr

def get_indiv_graphs(eeg_clip, top_k = 3):
    """
    Compute adjacency matrix for correlation graph
    Args:
        eeg_clip: shape (seq_len, num_nodes, input_dim)
    Returns:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
    """
    num_sensors = 19
    adj_mat = np.eye(num_sensors, num_sensors,
                        dtype=np.float32)  # diagonal is 1

    # (num_nodes, seq_len, input_dim)
    eeg_clip = np.transpose(eeg_clip, (1, 0, 2))
    assert eeg_clip.shape[0] == num_sensors

    # (num_nodes, seq_len*input_dim)
    eeg_clip = eeg_clip.reshape((num_sensors, -1))

    for i in range(0, num_sensors):
        for j in range(i + 1, num_sensors):
            xcorr = comp_xcorr(
                eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
            adj_mat[i, j] = xcorr
            adj_mat[j, i] = xcorr

    adj_mat = abs(adj_mat)

    if (top_k is not None):
        adj_mat = keep_topk(adj_mat, top_k=top_k, directed=True)
    else:
        raise ValueError('Invalid top_k value!')

    return adj_mat



def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian



def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


def calculate_random_walk_matrix(adj_mx):
    """
    State transition matrix D_o^-1W in paper.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx



def keep_topk(adj_mat, top_k=3, directed=True):
    """ "
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors
    for each node.
    Args:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        top_k: int
        directed: whether or not a directed graph
    Returns:
        adj_mat: sparse adjacency matrix, directed graph
    """
    # Set values that are not of top-k neighbors to 0:
    adj_mat_noSelfEdge = adj_mat.copy()
    for i in range(adj_mat_noSelfEdge.shape[0]):
        adj_mat_noSelfEdge[i, i] = 0

    top_k_idx = (-adj_mat_noSelfEdge).argsort(axis=-1)[:, :top_k]

    mask = np.eye(adj_mat.shape[0], dtype=bool)
    for i in range(0, top_k_idx.shape[0]):
        for j in range(0, top_k_idx.shape[1]):
            mask[i, top_k_idx[i, j]] = 1
            if not directed:
                mask[top_k_idx[i, j], i] = 1  # symmetric

    adj_mat = mask * adj_mat
    return adj_mat


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))
    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def compute_supports(adj_mat, filter_type):
    """
    Compute supports
    """
    supports = []
    supports_mat = []
    if filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(
            calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(
            calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        supports.append(torch.FloatTensor(support.toarray()))
    return supports


def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output

