import torch
import networkx as nx
from torch import device
import torch.nn.functional as F

import numpy as np
# import tensorflow as tf
import torch.distributions as dist


from typing import Optional
from cl.utils import normalize
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter
from torch_geometric.transforms import GDC
from torch.distributions import Uniform, Beta, Binomial
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, add_self_loops
from torch.distributions.bernoulli import Bernoulli


def permute(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute node embeddings or features.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Embeddings or features resulting from permutation.
    """
    return x[torch.randperm(x.size(0))]


def get_mixup_idx(x: torch.Tensor) -> torch.Tensor:
    """
    Generate node IDs randomly for mixup; avoid mixup the same node.

    Args:
        x: The latent embedding or node feature.

    Returns:
        torch.Tensor: Random node IDs.
    """
    mixup_idx = torch.randint(x.size(0) - 1, [x.size(0)])
    mixup_self_mask = mixup_idx - torch.arange(x.size(0))
    mixup_self_mask = (mixup_self_mask == 0)
    mixup_idx += torch.ones(x.size(0), dtype=torch.int) * mixup_self_mask
    return mixup_idx


def mixup(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Randomly mixup node embeddings or features with other nodes'.

    Args:
        x: The latent embedding or node feature.
        alpha: The hyperparameter controlling the mixup coefficient.

    Returns:
        torch.Tensor: Embeddings or features resulting from mixup.
    """
    device = x.device
    mixup_idx = get_mixup_idx(x).to(device)
    lambda_ = Uniform(alpha, 1.).sample([1]).to(device)
    x = (1 - lambda_) * x + lambda_ * x[mixup_idx]
    return x


def multiinstance_mixup(x1: torch.Tensor, x2: torch.Tensor,
                        alpha: float, shuffle=False) -> (torch.Tensor, torch.Tensor):
    """
    Randomly mixup node embeddings or features with nodes from other views.

    Args:
        x1: The latent embedding or node feature from one view.
        x2: The latent embedding or node feature from the other view.
        alpha: The mixup coefficient `\lambda` follows `Beta(\alpha, \alpha)`.
        shuffle: Whether to use fixed negative samples.

    Returns:
        (torch.Tensor, torch.Tensor): Spurious positive samples and the mixup coefficient.
    """
    device = x1.device
    lambda_ = Beta(alpha, alpha).sample([1]).to(device)
    if shuffle:
        mixup_idx = get_mixup_idx(x1).to(device)
    else:
        mixup_idx = x1.size(0) - torch.arange(x1.size(0)) - 1
    x_spurious = (1 - lambda_) * x1 + lambda_ * x2[mixup_idx]

    return x_spurious, lambda_


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[drop_mask, :] = 0

    return x

def add_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    n = int(x.size(0) * drop_prob)
    inst = torch.rand(n, x.size(1)).to(device)

    x = x.clone()
    x1 = torch.cat([x, inst], dim=0) 

    return x1

def rand_instance(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device

    drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)

    n = torch.sum(drop_mask)
    inst = torch.rand(n, x.size(1)).to(device)

    x1 = x.clone()
    x1[drop_mask, :] = inst 

    return x1


def replace_feature_np(x: torch.Tensor, p_m: float) -> torch.Tensor:
    device = x.device

    no, dim = x.shape
    m = torch.Tensor(np.random.binomial(1, p_m, x.shape)).to(device)

    x_bar = torch.zeros((no, dim), dtype=torch.float32).to(device)
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]
    x_tilde = x * (1-m) + x_bar * m
    x_tilde = x_tilde.to(device)

    return x_tilde


def replace_feature_np_ori(x: torch.Tensor, drop_prob: float) -> torch.Tensor:

    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    m = np.random.berne(1, p_m, x.shape)

    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def shufflerow(tensor, axis):
    # device = tensor.device
    # get permutation indices
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).cuda()
    for _ in range(tensor.ndim-axis-1):
        row_perm.unsqueeze_(-1)
    # reformat this for the gather operation
    row_perm = row_perm.repeat(
        *[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))
    return tensor.gather(axis, row_perm)


def replace_feature(x: torch.Tensor, p_r: float) -> torch.Tensor:

    # Randomly (and column-wise) shuffle data

    # fast
    # indexes = torch.randperm(no)
    # x_bar = x[indexes]

    # fast
    x_bar = shufflerow(x, 1)

    # low
    # x_bar=x
    no, dim = x.shape
    for i in range(dim):
        idx = torch.randperm(no).cuda()
        x_bar[:, i] = x[idx, i]

    p = torch.full(x.shape, p_r).cuda()

    m = torch.bernoulli(p)

    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m

    return x_tilde


def dropout_feature(x: torch.FloatTensor, drop_prob: float) -> torch.FloatTensor:
    return F.dropout(x, p=1. - drop_prob)


class AugmentTopologyAttributes(object):
    def __init__(self, pe=0.5, pf=0.5):
        self.pe = pe
        self.pf = pf

    def __call__(self, x, edge_index):
        edge_index = dropout_adj(edge_index, p=self.pe)[0]
        x = drop_feature(x, self.pf)
        return x, edge_index


def get_feature_weights(x, centrality, sparse=True):
    if sparse:
        x = x.to(torch.bool).to(torch.float32)
    else:
        x = x.abs()
    w = x.t() @ centrality
    w = w.log()

    return normalize(w)


def drop_feature_by_weight(x, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold,
                            torch.ones_like(weights) * threshold)  # clip
    drop_mask = torch.bernoulli(weights).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def get_eigenvector_weights(data):
    def _eigenvector_centrality(data):
        graph = to_networkx(data)
        x = nx.eigenvector_centrality_numpy(graph)
        x = [x[i] for i in range(data.num_nodes)]
        return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

    evc = _eigenvector_centrality(data)
    scaled_evc = evc.where(evc > 0, torch.zeros_like(evc))
    scaled_evc = scaled_evc + 1e-8
    s = scaled_evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]

    return normalize(s_col), evc


def get_degree_weights(data):
    edge_index_ = to_undirected(data.edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[data.edge_index[1]].to(torch.float32)
    scaled_deg_col = torch.log(deg_col)

    return normalize(scaled_deg_col), deg


def get_pagerank_weights(data, aggr: str = 'sink', k: int = 10):
    def _compute_pagerank(edge_index, damp: float = 0.85, k: int = 10):
        num_nodes = edge_index.max().item() + 1
        deg_out = degree(edge_index[0])
        x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

        for i in range(k):
            edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
            agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

            x = (1 - damp) * x + damp * agg_msg

        return x

    pv = _compute_pagerank(data.edge_index, k=k)
    pv_row = pv[data.edge_index[0]].to(torch.float32)
    pv_col = pv[data.edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

    return normalize(s), pv


def drop_edge_by_weight(edge_index, weights, drop_prob: float, threshold: float = 0.7):
    weights = weights / weights.mean() * drop_prob
    weights = weights.where(weights < threshold,
                            torch.ones_like(weights) * threshold)
    drop_mask = torch.bernoulli(1. - weights).to(torch.bool)

    return edge_index[:, drop_mask]


class AdaptivelyAugmentTopologyAttributes(object):
    def __init__(self, edge_weights, feature_weights, pe=0.5, pf=0.5, threshold=0.7):
        self.edge_weights = edge_weights
        self.feature_weights = feature_weights
        self.pe = pe
        self.pf = pf
        self.threshold = threshold

    def __call__(self, x, edge_index):
        edge_index = drop_edge_by_weight(
            edge_index, self.edge_weights, self.pe, self.threshold)
        x = drop_feature_by_weight(
            x, self.feature_weights, self.pf, self.threshold)

        return x, edge_index


def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(
        diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight


def get_sparse_adj(edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                   add_self_loop: bool = True) -> torch.sparse.Tensor:
    num_nodes = edge_index.max().item() + 1
    num_edges = edge_index.size(1)

    if edge_weight is None:
        edge_weight = torch.ones(
            (num_edges,), dtype=torch.float32, device=edge_index.device)

    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=num_nodes)
        edge_index, edge_weight = coalesce(
            edge_index, edge_weight, num_nodes, num_nodes)

    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, num_nodes, normalization='sym')

    adj_t = torch.sparse_coo_tensor(
        edge_index, edge_weight, size=(num_nodes, num_nodes)).coalesce()

    return adj_t.t()


def compute_markov_diffusion(
        edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
        alpha: float = 0.1, degree: int = 10,
        sp_eps: float = 1e-3, add_self_loop: bool = True):
    adj = get_sparse_adj(edge_index, edge_weight, add_self_loop)

    z = adj.to_dense()
    t = adj.to_dense()
    for _ in range(degree):
        t = (1.0 - alpha) * torch.spmm(adj, t)
        z += t
    z /= degree
    z = z + alpha * adj

    adj_t = z.t()

    return GDC().sparsify_dense(adj_t, method='threshold', eps=sp_eps)


