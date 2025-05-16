from torch_geometric.data import HeteroData
import os
import json
import yaml
import pathlib
from src.utils import count_parameters, AVGMeter, Reporter, Timer
from src.oven import Oven
from loguru import logger
import torch.distributed as dist
from src.utils import set_random_seed, setup_distributed, setup_default_logging_wt_dir
import pprint
import torch
import torch.nn as nn
import argparse
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import Linear, ResGatedGraphConv, HeteroConv
import torch.nn.functional as F
from scipy.sparse.csgraph import floyd_warshall
from metrics import RMSE
import traceback
def calculate_gpri(batch_original, batch_perturbed, edge_scores, k=10):
    """
    Calculate Graph Perturbation Robustness Index (GPRI)
    
    Args:
        batch_original: Original batch data
        batch_perturbed: Perturbed batch data
        edge_scores: Edge importance scores
        k: Number of top connections to consider
        
    Returns:
        gpri: Graph Perturbation Robustness Index
    """
    gpri_values = []
    
    for edge_type in edge_scores:
        # Get top-k important edges in original graph
        scores_orig = edge_scores[edge_type]
        if len(scores_orig) == 0:
            continue
            
        _, top_indices_orig = torch.topk(scores_orig, min(k, len(scores_orig)))
        top_edges_orig = set(top_indices_orig.cpu().numpy())
        
        # Get corresponding edges in perturbed graph
        if edge_type in batch_perturbed.edge_index_dict:
            edge_index_perturbed = batch_perturbed.edge_index_dict[edge_type]
            
            # Calculate intersection size
            intersection_size = len(top_edges_orig.intersection(set(range(edge_index_perturbed.size(1)))))
            
            # Calculate GPRI for this edge type
            if len(top_edges_orig) > 0:
                gpri_values.append(intersection_size / len(top_edges_orig))
    
    # Average GPRI across all edge types
    if len(gpri_values) > 0:
        return sum(gpri_values) / len(gpri_values)
    else:
        return 0.0

def vm_va_matrix(batch: HeteroData, mode="train"):
    Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5
    Ybus = create_Ybus(batch)
    delta_p, delta_q = deltapq_loss(batch, Ybus)
    
    # Calculate RMSE metrics
    matrix = {
        f"{mode}/PQ_Vm_rmse": RMSE(batch['PQ'].x[:, Vm], batch['PQ'].y[:, Vm]),
        f"{mode}/PQ_Va_rmse": RMSE(batch['PQ'].x[:, Va], batch['PQ'].y[:, Va]),
        f"{mode}/PV_Va_rmse": RMSE(batch['PV'].x[:, Va], batch['PV'].y[:, Va]),
        f"{mode}/delta_p": delta_p.abs().mean().item(),
        f"{mode}/delta_q": delta_q.abs().mean().item(),
    }
    
    # Add GPRI if edge scores are available
    if hasattr(batch, 'edge_scores') and batch.edge_scores:
        try:
            # Create a perturbed version of the batch for GPRI calculation
            batch_perturbed = batch.clone()
            
            # Apply small perturbation to edge attributes (5% noise)
            for edge_type, edge_attr in batch_perturbed.edge_attr_dict.items():
                if edge_attr is not None and len(edge_attr) > 0:
                    noise = torch.randn_like(edge_attr) * 0.05 * edge_attr.abs()
                    batch_perturbed[edge_type].edge_attr = edge_attr + noise
            
            # Calculate GPRI
            gpri = calculate_gpri(batch, batch_perturbed, batch.edge_scores)
            matrix[f"{mode}/GPRI"] = gpri
        except Exception as e:
            # If GPRI calculation fails, log and continue
            print(f"GPRI calculation failed: {e}")
    
    return matrix

def bi_deltapq_loss(graph_data: HeteroData, need_clone=False,
                    filt_type=True, aggr='abs'):
    """compute deltapq loss

    Args:
        graph_data (Hetero Graph): Batched Hetero graph data
        preds (dict): preds results

    Returns:
        torch.float: deltapq loss
    """
    def inner_deltapq_loss(bus, branch, edge_index, device):
        # makeYbus, reference to pypower makeYbus
        nb = bus.shape[0]  # number of buses
        nl = edge_index.shape[1]  # number of branch

        # branch = homo_graph_data.edge_attr
        BR_R, BR_X, BR_B, TAP, SHIFT = 0, 1, 2, 3, 4
        # bus = homo_graph_data.x
        PD, QD, GS, BS, PG, QG, VM, VA = 0, 1, 2, 3, 4, 5, 6, 7

        Ys = 1.0 / (branch[:, BR_R] + 1j * branch[:, BR_X])
        Bc = branch[:, BR_B]
        tap = torch.ones(nl).to(device)
        i = torch.nonzero(branch[:, TAP])
        tap[i] = branch[i, TAP]
        tap = tap * torch.exp(1j * branch[:, SHIFT])

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * torch.conj(tap))
        Yft = - Ys / torch.conj(tap)
        Ytf = - Ys / tap

        Ysh = bus[:, GS] + 1j * bus[:, BS]

        # build connection matrices
        f = edge_index[0]
        t = edge_index[1]
        Cf = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), f]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)
        Ct = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), t]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)

        i_nl = torch.cat([torch.arange(nl), torch.arange(nl)], dim=0).to(device)
        i_ft = torch.cat([f, t], dim=0)

        Yf = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Yff, Yft], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Yt = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Ytf, Ytt], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Ysh_square = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nb), torch.arange(nb)]).to(device),
            Ysh,
            (nb, nb),
            dtype=torch.complex64
        )

        Ybus = torch.matmul(Cf.T.to(torch.complex64), Yf) +\
            torch.matmul(Ct.T.to(torch.complex64), Yt) + Ysh_square

        v = bus[:, VM] * torch.exp(1j * bus[:, VA])

        i = torch.matmul(Ybus, v)
        i = torch.conj(i)
        s = v * i
        pd = bus[:, PD] + 1j * bus[:, QD]
        pg = bus[:, PG] + 1j * bus[:, QG]
        s = s + pd - pg

        delta_p = torch.real(s)
        delta_q = torch.imag(s)
        return delta_p, delta_q

    # preprocess
    if need_clone:
        graph_data = graph_data.clone()
    device = graph_data['PQ'].x.device

    # PQ: PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['PQ'].x = torch.cat([
        graph_data['PQ'].supply,
        graph_data['PQ'].x[:, :2]],
        dim=1)
    # PV: PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['PV'].x = torch.cat([
        graph_data['PV'].supply,
        graph_data['PV'].x[:, :2]],
        dim=1)
    # Slack PD, QD, GS, BS, PG, QG, Vm, Va
    graph_data['Slack'].x = torch.cat([
        graph_data['Slack'].supply,
        graph_data['Slack'].x[:, :2]],
        dim=1)

    # convert to homo graph for computing Ybus loss
    homo_graph_data = graph_data.to_homogeneous()

    index_diff = homo_graph_data.edge_index[1, :] - homo_graph_data.edge_index[0, :]
    # to index bigger than from index
    edge_attr_1 = homo_graph_data.edge_attr[index_diff > 0, :]
    edge_index_1 = homo_graph_data.edge_index[:, index_diff > 0]
    delta_p_1, delta_q_1 = inner_deltapq_loss(homo_graph_data.x, edge_attr_1, edge_index_1, device)

    # from index bigger than to index
    edge_index_2 = homo_graph_data.edge_index[:, index_diff < 0]
    edge_attr_2 = homo_graph_data.edge_attr[index_diff < 0, :]
    delta_p_2, delta_q_2 = inner_deltapq_loss(homo_graph_data.x, edge_attr_2, edge_index_2, device)

    delta_p, delta_q = (delta_p_1 + delta_p_2) / 2.0, (delta_q_1 + delta_q_2) / 2.0

    if filt_type:
        PQ_mask = homo_graph_data['node_type'] == 0
        PV_mask = homo_graph_data['node_type'] == 1
        delta_p = delta_p[PQ_mask | PV_mask]
        delta_q = delta_q[PQ_mask]

    if aggr == "abs":
        loss = delta_p.abs().mean() + delta_q.abs().mean()
    elif aggr == "square":
        loss = (delta_p**2).mean() + (delta_q**2).mean()
    else:
        raise TypeError(f"no such aggr: {aggr}")
    return loss


def create_Ybus(batch: HeteroData):
    homo_batch = batch.to_homogeneous().detach()
    bus = homo_batch.x
    index_diff = homo_batch.edge_index[1, :] - homo_batch.edge_index[0, :]
    # to index bigger than from index
    edge_attr = homo_batch.edge_attr[index_diff > 0, :]
    edge_index_ori = homo_batch.edge_index[:, index_diff > 0]
    device = batch['PQ'].x.device
    with torch.no_grad():
        edge_mask = torch.isnan(edge_attr[:,0])
        edge_attr = edge_attr[~edge_mask]
        edge_index = torch.vstack([edge_index_ori[0][~edge_mask],edge_index_ori[1][~edge_mask]])
        # makeYbus, reference to pypower makeYbus
        nb = bus.shape[0]  # number of buses
        nl = edge_index.shape[1]  # number of edges
        Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5
        BR_R, BR_X, BR_B, TAP, SHIFT = 0, 1, 2, 3, 4

        Ys = 1.0 / (edge_attr[:, BR_R] + 1j * edge_attr[:, BR_X])
        Bc = edge_attr[:, BR_B]
        tap = torch.ones(nl).to(device)
        i = torch.nonzero(edge_attr[:, TAP])
        tap[i] = edge_attr[i, TAP]
        tap = tap * torch.exp(1j * edge_attr[:, SHIFT])

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * torch.conj(tap))
        Yft = - Ys / torch.conj(tap)
        Ytf = - Ys / tap

        Ysh = bus[:, Gs] + 1j * bus[:, Bs]

        # build connection matrices
        f = edge_index[0]
        t = edge_index[1]
        Cf = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), f]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)
        Ct = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nl).to(device), t]),
            torch.ones(nl).to(device),
            (nl, nb)
        ).to(torch.complex64)

        i_nl = torch.cat([torch.arange(nl), torch.arange(nl)], dim=0).to(device)
        i_ft = torch.cat([f, t], dim=0)

        Yf = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Yff, Yft], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Yt = torch.sparse_coo_tensor(
            torch.vstack([i_nl, i_ft]),
            torch.cat([Ytf, Ytt], dim=0),
            (nl, nb),
            dtype=torch.complex64
        )

        Ysh_square = torch.sparse_coo_tensor(
            torch.vstack([torch.arange(nb), torch.arange(nb)]).to(device),
            Ysh,
            (nb, nb),
            dtype=torch.complex64
        )

        Ybus = torch.matmul(Cf.T.to(torch.complex64), Yf) +\
                torch.matmul(Ct.T.to(torch.complex64), Yt) + Ysh_square
    return Ybus

def deltapq_loss(batch, Ybus):
    Vm, Va, P_net, Q_net = 0, 1, 2, 3
    bus = batch.to_homogeneous().x
    v = bus[:, Vm] * torch.exp(1j * bus[:, Va])
    i = torch.conj(torch.matmul(Ybus, v))
    s = v * i + bus[:, P_net] + 1j * bus[:, Q_net]

    delta_p = torch.real(s)
    delta_q = torch.imag(s)
    return delta_p, delta_q


# -------------------------- #
#     1. various modules     #
# -------------------------- #
def compute_shortest_path_distances(adj_matrix):
    distances = floyd_warshall(csgraph=adj_matrix, directed=False)
    return distances


def convert_x_to_tanhx(tensor_in):
    return torch.tanh(tensor_in)


# ----- Enhanced Edge-Node Hierarchical Pooling (EENHPool)
class EENHPool(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=None):
        super(EENHPool, self).__init__()
        hidden_dim = hidden_dim or in_dim
        
        # Node and edge scoring parameters
        self.W_h = nn.Linear(edge_dim, hidden_dim)
        self.W_n = nn.Linear(in_dim * 2, hidden_dim)
        self.w_e = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.w_e)
        
        # Feature transformation
        self.feature_transform = nn.Linear(in_dim, in_dim)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Compute hierarchical edge importance and lift local features
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Dictionary of edge attributes for each edge type
            
        Returns:
            local_features: Dictionary of lifted local features for each node type
            edge_scores: Dictionary of edge importance scores
        """
        local_features = {}
        edge_scores = {}
        
        # First pass: compute edge scores
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in edge_attr_dict or edge_index.size(1) == 0:
                # Skip if no edges or no attributes
                edge_scores[edge_type] = torch.tensor([], device=edge_index.device)
                continue
                
            src_type, _, dst_type = edge_type
            
            # Get node features
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            edge_attr = edge_attr_dict[edge_type]
            
            # Compute edge scores
            src_idx, dst_idx = edge_index
            node_features = torch.cat([x_src[src_idx], x_dst[dst_idx]], dim=1)
            
            # Enhanced edge importance calculation with attention mechanism
            edge_h = self.W_h(edge_attr)
            node_h = self.W_n(node_features)
            combined_h = F.relu(edge_h + node_h)
            scores = torch.matmul(combined_h, self.w_e).squeeze(-1)
            alpha = F.softmax(scores, dim=0)
            
            edge_scores[edge_type] = alpha
            
        # Second pass: compute local features with weighted aggregation
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in edge_attr_dict or edge_index.size(1) == 0:
                continue
                
            src_type, _, dst_type = edge_type
            src_idx, dst_idx = edge_index
            alpha = edge_scores[edge_type]
            
            # Initialize local features if not already done
            for node_type in [src_type, dst_type]:
                if node_type not in local_features:
                    local_features[node_type] = torch.zeros_like(x_dict[node_type])
            
            # Compute local features (graph lifting) with importance-weighted aggregation
            if src_type == dst_type:
                # Self-loops: special handling for self-connections
                local_features[src_type].index_add_(
                    0, src_idx, 
                    -alpha.unsqueeze(1) * x_dict[dst_type][dst_idx]
                )
            else:
                # Regular edges between different node types
                local_features[src_type].index_add_(
                    0, src_idx, 
                    -alpha.unsqueeze(1) * x_dict[dst_type][dst_idx]
                )
                
                local_features[dst_type].index_add_(
                    0, dst_idx,
                    -alpha.unsqueeze(1) * x_dict[src_type][src_idx]
                )
        
        # Add original features and apply feature transformation with residual connection
        for node_type in x_dict:
            if node_type in local_features:
                # u_i = x_i - sum(alpha_ij * x_j)
                local_features[node_type] = x_dict[node_type] + local_features[node_type]
                # Apply feature transformation with residual connection
                local_features[node_type] = local_features[node_type] + self.feature_transform(local_features[node_type])
            else:
                # If no neighbors, just use the original features
                local_features[node_type] = x_dict[node_type]
            
        return local_features, edge_scores

# ----- ca
class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.k_dim**0.5
        # print("s1", q1.shape, k2.shape, attn.shape)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3)
        # print("s2", output.shape)
        output= output.contiguous().view(batch_size, seq_len1, -1)
        # print("s3", output.shape)
        output = self.proj_o(output)
        # print("s4", output.shape)
    
        return output


# ------- ffn ---
class GLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout_ratio=0.1):
        # in A*2, hidden:A2, out:A
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x) * v
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 batch_size=100,
                 dropout_ratio=0.1):
        super(GatedFusion, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features * 2, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_ratio)
        self.batch_size = batch_size

    def forward(self, pq_features, slack_features):
        # get size
        BK, D = pq_features.size()
        B = self.batch_size
        K = BK // B
        pq_features = pq_features.view(B, K, D)  # (B, K, D)
        slack_expanded = slack_features.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
        combined = torch.cat([pq_features, slack_expanded], dim=-1)  # (B, K, 2D)

        x = self.fc1(combined)  # (B, K, 2 * hidden_features)
        x, v = x.chunk(2, dim=-1)  # (B, K, hidden_features) each
        x = self.act(x) * v  # (B, K, hidden_features)
        x = self.fc2(x)  # (B, K, D)
        x = self.drop(x)  # (B, K, D)

        return x.contiguous().view(B*K, D)


# -------------------------- #
#     2. various layers      #
# -------------------------- #
class GraphLayer(torch.nn.Module):
    def __init__(self, 
                 emb_dim, 
                 edge_dim,
                 num_heads,
                 batch_size,
                 with_norm,
                 act_layer=nn.ReLU,
                 gcn_layer_per_block=2):
        super().__init__()
        
        self.graph_layers = nn.ModuleList()
        for _ in range(gcn_layer_per_block):
            self.graph_layers.append(
                HeteroConv({
                        ('PQ', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PQ', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PQ', 'default', 'Slack'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('PV', 'default', 'Slack'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('Slack', 'default', 'PQ'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                        ('Slack', 'default', 'PV'): ResGatedGraphConv((emb_dim,emb_dim), emb_dim, edge_dim=edge_dim),
                    }, 
                    aggr='sum')
            )
        self.act_layer = act_layer()
        self.global_transform = nn.Linear(emb_dim, emb_dim)

        self.cross_attention = CrossAttention(in_dim1=emb_dim,
                                              in_dim2=emb_dim,
                                              k_dim=emb_dim//num_heads,
                                              v_dim=emb_dim//num_heads,
                                              num_heads=num_heads)

        self.norm = torch.nn.LayerNorm(emb_dim) if with_norm else nn.Identity()
        self.batch_size = batch_size


    def forward(self, batch: HeteroData):
        graph_x_dict = batch.x_dict

        # vitual global node
        pq_x = torch.stack(torch.chunk(graph_x_dict['PQ'], self.batch_size, dim=0), dim=0) # B, 29, D
        pv_x = torch.stack(torch.chunk(graph_x_dict['PV'], self.batch_size, dim=0), dim=0)
        slack_x = torch.stack(torch.chunk(graph_x_dict['Slack'], self.batch_size, dim=0), dim=0)
        global_feature = torch.cat((pq_x,pv_x,slack_x), dim=1) # B, (29+9+1), D
        global_feature = self.global_transform(global_feature)
        global_feature_mean = global_feature.mean(dim=1, keepdim=True)
        global_feature_max, _ = global_feature.max(dim=1, keepdim=True)

        # forward gcn
        for layer in self.graph_layers:
            graph_x_dict = layer(graph_x_dict, 
                                 batch.edge_index_dict,
                                 batch.edge_attr_dict)
            ## NEW: add non-linear
            graph_x_dict = {key: self.act_layer(x) for key, x in graph_x_dict.items()}

        global_node_feat = torch.cat([global_feature_mean, global_feature_max], dim=1)
        
        # cross attent the global feat.
        res = {}
        for key in ["PQ", "PV"]:
            # get size
            BN, K = batch[key].x.size()
            B = self.batch_size
            N = BN // B
            # ca
            graph_x_dict[key] = graph_x_dict[key] + self.cross_attention(graph_x_dict[key].view(B, N, K), global_node_feat).contiguous().view(B*N, K)
            # norm
            res[key] = self.norm(graph_x_dict[key])
        res["Slack"] = graph_x_dict["Slack"]

        return res


# ----- ffn layers
class FFNLayer(torch.nn.Module):

    def __init__(self, 
                embed_dim_in: int,
                embed_dim_hid: int,
                embed_dim_out: int, 
                mlp_dropout: float, 
                with_norm: bool,
                act_layer=nn.GELU):
        super().__init__()

        # in: embed_dim_out, hidden: embed_dim_hid*2, out: embed_dim_out
        self.mlp = GLUFFN(in_features=embed_dim_in, 
                          hidden_features=embed_dim_hid, 
                          out_features=embed_dim_out,
                          act_layer=act_layer,
                          dropout_ratio=mlp_dropout)

        self.norm = torch.nn.LayerNorm(embed_dim_out) if with_norm else nn.Identity()

    def forward(self, x):
        x = x + self.mlp(x)
        return self.norm(x)
    

class FFNFuseLayer(torch.nn.Module):

    def __init__(self, 
                embed_dim_in: int,
                embed_dim_hid: int,
                embed_dim_out: int, 
                mlp_dropout: float, 
                with_norm: bool,
                batch_size: int,
                act_layer=nn.GELU):
        super().__init__()
        self.mlp = GatedFusion(in_features=embed_dim_in, 
                          hidden_features=embed_dim_hid, 
                          out_features=embed_dim_out,
                          act_layer=act_layer, 
                          batch_size=batch_size,
                          dropout_ratio=mlp_dropout)

        self.norm = torch.nn.LayerNorm(embed_dim_out) if with_norm else nn.Identity()

    def forward(self, x, x_aux):
        x = x + self.mlp(x, x_aux)
        return self.norm(x)


# ----- Stability-Regularized Temporal Graph Transformer (SRT-GT)
class SRT_GT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_timesteps, dropout=0.1):
        super(SRT_GT, self).__init__()
        
        # Temporal parameters with better initialization values
        self.gamma = nn.Parameter(torch.Tensor(num_timesteps))
        self.eta = nn.Parameter(torch.Tensor(num_timesteps))
        # Initialize with small positive values for stability
        nn.init.constant_(self.gamma, 0.15)  # Slightly increased for better message passing
        nn.init.constant_(self.eta, 0.6)     # Slightly increased for better self-loop importance
        
        # Transformation matrices with layer normalization
        self.W_t = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim)
            ) for _ in range(num_timesteps)
        ])
        
        # Integration parameter for local features
        self.xi = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.xi, 0.2)  # Increased to give more weight to local features
        
        # Output projection for better feature integration
        self.output_proj = nn.Linear(in_dim, in_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        # Store temporal edge importances for regularization
        self.temporal_edge_importances = []
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, local_features, timestep):
        """
        Apply temporal graph transformer update with improved stability
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Dictionary of edge attributes for each edge type
            local_features: Dictionary of lifted local features from EENHPool
            timestep: Current timestep
            
        Returns:
            updated_x_dict: Updated node features
        """
        updated_x_dict = {}
        edge_importances = {}
        
        # First pass: compute messages for all edges
        messages_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(1) == 0:
                # Skip if no edges
                continue
                
            src_type, _, dst_type = edge_type
            
            # Get node features
            x_src = x_dict[src_type]
            
            # Compute attention scores for message passing
            src_idx, dst_idx = edge_index
            
            # Transform source node features
            messages = self.W_t[timestep](x_src[src_idx])
            
            # Apply temporal coefficient
            gamma_t = torch.sigmoid(self.gamma[timestep])
            
            # Store messages for aggregation
            if dst_type not in messages_dict:
                messages_dict[dst_type] = []
            
            # Store weighted messages and indices
            messages_dict[dst_type].append((dst_idx, gamma_t * messages))
            
            # Store edge importances for regularization
            edge_importances[edge_type] = gamma_t
        
        # Second pass: aggregate messages and apply self-loops
        for node_type in x_dict:
            # Initialize with original features (residual connection)
            if node_type not in updated_x_dict:
                updated_x_dict[node_type] = x_dict[node_type].clone()
            
            # Aggregate messages if any
            if node_type in messages_dict:
                for dst_idx, messages in messages_dict[node_type]:
                    updated_x_dict[node_type].index_add_(0, dst_idx, messages)
            
            # Apply self-loop with eta parameter (gating mechanism)
            eta_t = torch.sigmoid(self.eta[timestep])
            
            # Residual connection with gated self-loop
            updated_x_dict[node_type] = (1 - eta_t) * updated_x_dict[node_type] + eta_t * x_dict[node_type]
            
            # Integrate local features with xi parameter
            if node_type in local_features:
                # Adaptive integration of local features
                updated_x_dict[node_type] = updated_x_dict[node_type] + self.xi * local_features[node_type]
            
            # Apply non-linearity, projection and dropout
            updated_x_dict[node_type] = self.act(updated_x_dict[node_type])
            updated_x_dict[node_type] = self.output_proj(updated_x_dict[node_type]) + updated_x_dict[node_type]  # Residual connection
            updated_x_dict[node_type] = self.dropout(updated_x_dict[node_type])
        
        # Store edge importances for regularization loss
        self.temporal_edge_importances.append(edge_importances)
        
        return updated_x_dict
    
    def get_temporal_regularization_loss(self, lambda_reg=0.001):
        """
        Compute temporal regularization loss to enforce smoothness
        
        Args:
            lambda_reg: Regularization weight (reduced for better balance)
            
        Returns:
            reg_loss: Temporal regularization loss
        """
        if len(self.temporal_edge_importances) <= 1:
            return torch.tensor(0.0, device=self.gamma.device)
        
        reg_loss = torch.tensor(0.0, device=self.gamma.device)
        
        # Compute L2 difference between consecutive timesteps
        for t in range(len(self.temporal_edge_importances) - 1):
            for edge_type in self.temporal_edge_importances[t]:
                if edge_type in self.temporal_edge_importances[t+1]:
                    diff = self.temporal_edge_importances[t+1][edge_type] - self.temporal_edge_importances[t][edge_type]
                    reg_loss = reg_loss + torch.sum(diff ** 2)
        
        return lambda_reg * reg_loss
    
    def reset_temporal_importances(self):
        """Reset stored temporal edge importances"""
        self.temporal_edge_importances = []

# -------------------------- #
#     3. building block      #
# -------------------------- #
class HybridBlock(nn.Module):
    def __init__(self, 
                 emb_dim_in, 
                 emb_dim_out, 
                 with_norm, 
                 edge_dim, 
                 batch_size,
                 dropout_ratio=0.1,
                 layers_in_gcn=2,
                 heads_ca=4,
                 num_timesteps=3):
        super(HybridBlock, self).__init__()
        self.emb_dim_in = emb_dim_in
        self.with_norm = with_norm
        self.num_timesteps = num_timesteps

        # Enhanced Edge-Node Hierarchical Pooling
        self.eenhpool = EENHPool(in_dim=emb_dim_in, edge_dim=edge_dim)
        
        # Stability-Regularized Temporal Graph Transformer
        self.srt_gt = SRT_GT(
            in_dim=emb_dim_in,
            hidden_dim=emb_dim_in,
            num_timesteps=num_timesteps,
            dropout=dropout_ratio
        )
        
        # Keep the original graph layer as fallback
        self.branch_graph = GraphLayer(emb_dim=emb_dim_in,
                                       edge_dim=edge_dim, 
                                       num_heads=heads_ca, 
                                       batch_size=batch_size,
                                       with_norm=with_norm, 
                                       gcn_layer_per_block=layers_in_gcn)

        # ---- mlp: activation + increase dimension
        self.ffn = nn.ModuleDict()
        self.ffn['PQ'] = FFNFuseLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    batch_size=batch_size,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)
        self.ffn['PV'] = FFNFuseLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    batch_size=batch_size,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)
        self.ffn['Slack'] = FFNLayer(embed_dim_in=emb_dim_in, embed_dim_hid=emb_dim_out,
                                    embed_dim_out=emb_dim_out,
                                    mlp_dropout=dropout_ratio, 
                                    with_norm=with_norm)

    def forward(self, batch: HeteroData):
        # Store original features for residual connections
        original_features = {k: v.clone() for k, v in batch.x_dict.items()}
        
        # Apply the original graph layer first for better feature extraction
        res_graph = self.branch_graph(batch)
        
        # Update batch with graph layer results
        for key in res_graph:
            batch[key].x = res_graph[key]
        
        # Get local features using EENHPool
        local_features, edge_scores = self.eenhpool(
            batch.x_dict, 
            batch.edge_index_dict, 
            batch.edge_attr_dict
        )
        
        # Reset temporal importances at the beginning of each forward pass
        self.srt_gt.reset_temporal_importances()
        
        # Apply temporal graph transformer for multiple timesteps
        x_dict = batch.x_dict.copy()
        for t in range(self.num_timesteps):
            x_dict = self.srt_gt(
                x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                local_features,
                t
            )
        
        # Adaptive feature fusion with original features (global residual connection)
        for node_type, x in x_dict.items():
            # Weighted combination of transformed features and original features
            alpha = 0.7  # Weight for transformed features
            batch[node_type].x = alpha * x + (1 - alpha) * original_features[node_type]
            
        # Store edge scores for GPRI calculation
        # Use setattr to avoid attribute error
        setattr(batch, 'edge_scores', edge_scores)
        
        # Apply FFN layers
        feat_slack = batch["Slack"].x
        
        for key in batch.x_dict:
            x = batch[key].x
            if "slack" in key.lower():
                batch[key].x = self.ffn[key](x)
            else:
                batch[key].x = self.ffn[key](x, feat_slack)
        
        # Store temporal regularization loss for later use
        # Use setattr to avoid attribute error
        setattr(batch, 'temporal_reg_loss', self.srt_gt.get_temporal_regularization_loss())

        return batch

# -------------------------- #
#     4. powerflow net       #
# -------------------------- #
class PFNet(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 num_block, 
                 with_norm,  
                 batch_size,
                 dropout_ratio,
                 heads_ca, 
                 layers_per_graph=2,
                 flag_use_edge_feat=False,
                 num_timesteps=2,
                 lambda_reg=0.001):
        super(PFNet, self).__init__()

        # ---- parse params ----
        if isinstance(hidden_channels, list):
            hidden_block_layers = hidden_channels
            num_block = len(hidden_block_layers) - 1
        elif isinstance(hidden_channels, int):
            hidden_block_layers = [hidden_channels] * (num_block+1)
        else:
            raise TypeError("Unsupported type: {}".format(type(hidden_channels)))
        self.hidden_block_layers = hidden_block_layers
        self.flag_use_edge_feat = flag_use_edge_feat
        self.lambda_reg = lambda_reg

        # ---- edge encoder ----
        if self.flag_use_edge_feat:
            self.edge_encoder = Linear(5, hidden_channels)
            edge_dim = hidden_channels
        else:
            self.edge_encoder = None
            edge_dim = 5

        # ---- node encoder ----
        self.encoders = nn.ModuleDict()
        self.encoders['PQ'] = Linear(6, hidden_block_layers[0])
        self.encoders['PV'] = Linear(6, hidden_block_layers[0])
        self.encoders['Slack'] = Linear(6, hidden_block_layers[0])
        
        # ---- blocks ----
        self.blocks = nn.ModuleList()
        for channel_in, channel_out in zip(hidden_block_layers[:-1], hidden_block_layers[1:]):
            self.blocks.append(
                HybridBlock(emb_dim_in=channel_in, 
                    emb_dim_out=channel_out, 
                    with_norm=with_norm, 
                    edge_dim=edge_dim, 
                    batch_size=batch_size,
                    dropout_ratio=dropout_ratio,
                    layers_in_gcn=layers_per_graph,
                    heads_ca=heads_ca,
                    num_timesteps=num_timesteps)
            )
        self.num_blocks = len(self.blocks)
        
        # predictor        
        final_dim = sum(hidden_block_layers) - hidden_block_layers[0]
        self.predictor = nn.ModuleDict()
        self.predictor['PQ'] = Linear(final_dim, 6)
        self.predictor['PV'] = Linear(final_dim, 6)
        

    def forward(self, batch):
        # construct edge feats if neccessary
        if self.flag_use_edge_feat:
            for key in batch.edge_attr_dict:
                cur_edge_attr = batch.edge_attr_dict[key]
                r, x = cur_edge_attr[:, 0], cur_edge_attr[:, 1]
                cur_edge_attr[:, 0], cur_edge_attr[:, 1] = \
                    1.0 / torch.sqrt(r ** 2 + x ** 2), torch.arctan(r / x)
                # edge_attr_dict[key] = self.edge_encoder(cur_edge_attr)
                batch[key].edge_attr = self.edge_encoder(cur_edge_attr)
        
        # encoding
        for key, x in batch.x_dict.items():
            # print("="*20, key, "\t", x.shape)
            batch[key].x = self.encoders[key](x)

        # blocks and aspp
        multi_level_pq = []
        multi_level_pv = []
        for index, block in enumerate(self.blocks):
                batch = block(batch)
                multi_level_pq.append(batch["PQ"].x)
                multi_level_pv.append(batch["PV"].x)

        output = {
            'PQ': self.predictor['PQ'](torch.cat(multi_level_pq, dim=1)),
            'PV': self.predictor['PV'](torch.cat(multi_level_pv, dim=1))
        }
        return output

# -------------------------- #
#     5. iterative pf       #
# -------------------------- #
class IterGCN(nn.Module):
    def __init__(self, 
                 hidden_channels, 
                 num_block, 
                 with_norm,
                 num_loops_train, 
                 scaling_factor_vm, 
                 scaling_factor_va, 
                 loss_type,
                 batch_size, **kwargs):
        super(IterGCN, self).__init__()
        # param
        self.scaling_factor_vm = scaling_factor_vm
        self.scaling_factor_va = scaling_factor_va
        self.num_loops = num_loops_train

        # model
        self.net = PFNet(hidden_channels=hidden_channels, 
                         num_block=num_block, 
                         with_norm=with_norm, 
                         batch_size=batch_size, 
                         dropout_ratio=kwargs.get("dropout_ratio", 0.1), 
                         heads_ca=kwargs.get("heads_ca", 4),
                         layers_per_graph=kwargs.get("layers_per_graph", 2),
                         flag_use_edge_feat=kwargs.get("flag_use_edge_feat", False),
                         num_timesteps=kwargs.get("num_timesteps", 2),
                         lambda_reg=kwargs.get("lambda_reg", 0.001)
                    )
        
        # include a ema model for better I/O
        self.ema_warmup_epoch = kwargs.get("ema_warmup_epoch", 0)
        self.ema_decay_param = kwargs.get("ema_decay_param", 0.99)
        self.flag_use_ema = kwargs.get("flag_use_ema", False)
        if self.flag_use_ema:
            # Ensure EMA model has the same parameters as the main model
            self.ema_model = PFNet(hidden_channels=hidden_channels, 
                            num_block=num_block, 
                            with_norm=with_norm, 
                            batch_size=batch_size, 
                            dropout_ratio=kwargs.get("dropout_ratio", 0.1), 
                            heads_ca=kwargs.get("heads_ca", 4),
                            layers_per_graph=kwargs.get("layers_per_graph", 2),
                            flag_use_edge_feat=kwargs.get("flag_use_edge_feat", False),
                            num_timesteps=kwargs.get("num_timesteps", 2),
                            lambda_reg=kwargs.get("lambda_reg", 0.001)
                        )

            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None

        # loss
        if loss_type == 'l1':
            self.critien = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.critien = nn.SmoothL1Loss()
        elif loss_type == 'l2':
            self.critien = nn.MSELoss()
        elif loss_type == 'l3':
            self.critien = nn.HuberLoss()   
        else:
            raise TypeError(f"no such loss type: {loss_type}")

        # loss weights
        self.flag_weighted_loss = kwargs.get("flag_weighted_loss", False)
        self.loss_weight_equ = kwargs.get("loss_weight_equ", 1.0)
        self.loss_weight_vm = kwargs.get("loss_weight_vm", 1.0)
        self.loss_weight_va = kwargs.get("loss_weight_va", 1.0)

    def update_ema_model(self, epoch, i_iter, len_loader):
        if not self.flag_use_ema:
            return 
        
        # update teacher model with EMA
        with torch.no_grad():
            if epoch > self.ema_warmup_epoch:
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len_loader * self.ema_warmup_epoch
                        + 1
                    ),
                    self.ema_decay_param,
                )
            else:
                ema_decay = 0.0

            # update weight with safety check for parameter shape mismatches
            for param_train, param_eval in zip(self.net.parameters(), self.ema_model.parameters()):
                # Skip if shapes don't match
                if param_train.data.shape != param_eval.data.shape:
                    print(f"Warning: Parameter shape mismatch - train: {param_train.data.shape}, ema: {param_eval.data.shape}")
                    continue
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            
            # update bn with safety check
            for buffer_train, buffer_eval in zip(self.net.buffers(), self.ema_model.buffers()):
                # Skip if shapes don't match
                if buffer_train.data.shape != buffer_eval.data.shape:
                    print(f"Warning: Buffer shape mismatch - train: {buffer_train.data.shape}, ema: {buffer_eval.data.shape}")
                    continue
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)


    def forward(self, batch, flag_return_losses=False, flag_use_ema_infer=False, num_loop_infer=0):
        # get size
        num_PQ = batch['PQ'].x.shape[0]
        num_PV = batch['PV'].x.shape[0]
        num_Slack = batch['Slack'].x.shape[0]
        Vm, Va, P_net, Q_net, Gs, Bs = 0, 1, 2, 3, 4, 5

        # use different loops during inference phase
        if num_loop_infer < 1:
            num_loops = self.num_loops
        else:
            num_loops = num_loop_infer
        
        # whether use ema model for inference
        if not self.flag_use_ema:
            flag_use_ema_infer = False

        # loss record
        loss = 0.0
        res_dict = {"loss_equ": 0.0, "loss_pq_vm": 0.0, "loss_pq_va": 0.0, "loss_pv_va": 0.0, "loss_temporal_reg": 0.0}
        Ybus = create_Ybus(batch.detach())
        delta_p, delta_q = deltapq_loss(batch, Ybus)

        # Initialize current_output before the loop
        current_output = None
        
        # iterative loops
        for i in range(num_loops):
            # ----------- updated input ------------
            cur_batch = batch.clone()

            # use ema for better iterative fittings
            if self.flag_use_ema and i > 0 and not flag_use_ema_infer and current_output is not None:
                # Store current batch for EMA model
                cur_batch_hist = cur_batch.clone().detach()
                
                self.ema_model.eval()
                with torch.no_grad():
                    output_ema = self.ema_model(cur_batch_hist)
                
                # Update current batch with EMA predictions
                cur_batch['PV'].x[:, Va] = cur_batch['PV'].x[:, Va] - current_output['PV'][:, Va] * self.scaling_factor_va + output_ema['PV'][:, Va] * self.scaling_factor_va
                cur_batch['PQ'].x[:, Vm] = cur_batch['PQ'].x[:, Vm] - current_output['PQ'][:, Vm] * self.scaling_factor_vm + output_ema['PQ'][:, Vm] * self.scaling_factor_vm
                cur_batch['PQ'].x[:, Va] = cur_batch['PQ'].x[:, Va] - current_output['PQ'][:, Va] * self.scaling_factor_va + output_ema['PQ'][:, Va] * self.scaling_factor_va

                delta_p, delta_q = deltapq_loss(cur_batch, Ybus)
                self.ema_model.train()

            # update the inputs --- use deltap and deltaq
            cur_batch['PQ'].x[:, P_net] = delta_p[:num_PQ]  # deltap
            cur_batch['PQ'].x[:, Q_net] = delta_q[:num_PQ]  # deltaq
            cur_batch['PV'].x[:, P_net] = delta_p[num_PQ:num_PQ+num_PV]
            cur_batch = cur_batch.detach()
            cur_batch_hist = cur_batch.clone().detach()
            
            # ----------- forward ------------
            if flag_use_ema_infer:
                output = self.ema_model(cur_batch)
            else:
                output = self.net(cur_batch)
                
            # Store output for next iteration's EMA update
            if self.flag_use_ema and not flag_use_ema_infer:
                # Save current output for next iteration
                current_output = {k: v.clone().detach() for k, v in output.items() if isinstance(v, torch.Tensor)}

            # --------------- update vm and va --------------
            batch['PV'].x[:, Va] += output['PV'][:, Va] * self.scaling_factor_va
            batch['PQ'].x[:, Vm] += output['PQ'][:, Vm] * self.scaling_factor_vm
            batch['PQ'].x[:, Va] += output['PQ'][:, Va] * self.scaling_factor_va

            # --------------- calculate loss --------------
            delta_p, delta_q = deltapq_loss(batch, Ybus)

            equ_loss = self.critien(delta_p[:num_PQ+num_PV],
                                    torch.zeros_like(delta_p[:num_PQ+num_PV]))\
                    + self.critien(delta_q[:num_PQ][batch['PQ'].q_mask],
                                    torch.zeros_like(delta_q[:num_PQ][batch['PQ'].q_mask]))
            
            pq_vm_loss = self.critien(batch['PQ'].x[:,Vm], batch['PQ'].y[:,Vm])
            pv_va_loss = self.critien(batch['PV'].x[:,Va], batch['PV'].y[:,Va])
            pq_va_loss = self.critien(batch['PQ'].x[:,Va], batch['PQ'].y[:,Va])
            
            # Add temporal regularization loss if available
            # Get device from one of the tensors in the batch
            device = batch['PQ'].x.device if 'PQ' in batch else next(iter(batch.x_dict.values())).device
            temporal_reg_loss = torch.tensor(0.0, device=device)
            if hasattr(cur_batch, 'temporal_reg_loss'):
                temporal_reg_loss = cur_batch.temporal_reg_loss

            if flag_return_losses:
                res_dict['loss_equ'] += equ_loss.cpu().item()
                res_dict['loss_pq_vm'] += pq_vm_loss.cpu().item()
                res_dict['loss_pq_va'] += pq_va_loss.cpu().item()
                res_dict['loss_pv_va'] += pv_va_loss.cpu().item()
                res_dict['loss_temporal_reg'] += temporal_reg_loss.cpu().item()
            
            if self.flag_weighted_loss:
                loss = loss + equ_loss * self.loss_weight_equ + pq_vm_loss * self.loss_weight_vm + (pv_va_loss + pq_va_loss) * self.loss_weight_va + temporal_reg_loss
            else:
                loss = loss + equ_loss + pq_vm_loss + pv_va_loss + pq_va_loss + temporal_reg_loss
            

        batch['PQ'].x[~batch['PQ'].q_mask, Q_net] = -delta_q[:num_PQ][~batch['PQ'].q_mask]
        batch['PV'].x[:, Q_net] = -delta_q[num_PQ:num_PQ+num_PV]
        batch['Slack'].x[:, P_net] = -delta_p[num_PQ+num_PV:num_PQ+num_PV+num_Slack]
        batch['Slack'].x[:, Q_net] = -delta_q[num_PQ+num_PV:num_PQ+num_PV+num_Slack]

        if flag_return_losses:
            return batch, loss, res_dict
        return batch, loss


# torch.autograd.set_detect_anomaly(True)
class SubclassOven(Oven):
    def __init__(self, cfg, log_dir):
        super(SubclassOven,self).__init__(cfg)
        self.cfg = cfg
        self.ngpus = cfg.get('ngpus', 1)
        if self.ngpus == 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
            self.reporter = Reporter(cfg, log_dir)
        self.matrix = self._init_matrix()
        self.train_loader, self.valid_loader = self._init_data()
        self.criterion = self._init_criterion()
        self.model = self._init_model()
        self.optim, self.scheduler = self._init_optim()
        checkpt_path = self.cfg['model'].get("resume_ckpt_path", "")
        # self.resume_training = True if os.path.exists(os.path.join(self.cfg['log_path'], 'ckpt_latest.pt')) else False
        self.resume_training = True if os.path.exists(checkpt_path) else False
        self.checkpt_path = checkpt_path
        # using ema info
        self.flag_use_ema_model = self.cfg['model'].get("flag_use_ema", False)
        
    def _init_matrix(self):
        if self.cfg['model']['matrix'] == 'vm_va':
            return vm_va_matrix
        else:
            raise TypeError(f"No such of matrix {self.cfg['model']['matrix']}")

    def _init_model(self):        
        model = IterGCN(**self.cfg['model'])
        model = model.to(self.device)
        return model
    
    def _init_criterion(self):
        if self.cfg['loss']['type'] == "deltapq_loss":
            return deltapq_loss
        elif self.cfg['loss']['type'] == "bi_deltapq_loss":
            return bi_deltapq_loss
        else:
            raise TypeError(f"No such of loss {self.cfg['loss']['type']}")
        
    def exec_epoch(self, epoch, flag, flag_infer_ema=False):
        flag_return_losses = self.cfg.get("flag_return_losses", False)
        if flag == 'train':
            if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                logger.info(f'-------------------- Epoch: {epoch+1} --------------------')
            self.model.train()
            if self.cfg['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            
            # record vars
            train_loss = AVGMeter()
            train_matrix = dict()
            total_batch = len(self.train_loader)
            print_period = self.cfg['train'].get('logs_freq', 8)
            print_freq = total_batch // print_period 
            print_freq_lst = [i * print_freq for i in range(1, print_period)] + [total_batch - 1]
            
            # start loops
            for batch_id, batch in enumerate(self.train_loader):
                # data
                batch.to(self.device, non_blocking=True)
                
                # forward
                self.optim.zero_grad()
                if flag_return_losses:
                    pred, loss, record_losses = self.model(batch, flag_return_losses=True)
                else:
                    pred, loss = self.model(batch)

                # records
                cur_matrix = self.matrix(pred)
                if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                    # logger.info(f"Iter:{batch_id}/{total_batch} - {str(cur_matrix)}")
                    # print(cur_matrix)
                    pass
                if batch_id == 0:
                    for key in cur_matrix:
                        train_matrix[key] = AVGMeter()

                for key in cur_matrix:
                    train_matrix[key].update(cur_matrix[key])
                
                # backwards
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                train_loss.update(loss.item())

                # update ema
                if self.flag_use_ema_model:
                    if self.cfg['distributed']:
                        self.model.module.update_ema_model(epoch, batch_id + epoch * total_batch, total_batch)
                    else:
                        self.model.update_ema_model(epoch, batch_id + epoch * total_batch, total_batch)

                # print stats
                if (batch_id in print_freq_lst) or ((batch_id + 1) == total_batch):
                    if self.cfg['distributed']:
                        if dist.get_rank() == 0:
                            if flag_return_losses:
                                ret_loss_str = " ".join(["{}:{:.5f}".format(x, y) for x,y in record_losses.items()])
                                logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}, {ret_loss_str}")
                            else:
                                logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}")
                    else:
                        if flag_return_losses:
                            ret_loss_str = " ".join(["{}:{:.5f}".format(x, y) for x,y in record_losses.items()])
                            logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}, {ret_loss_str}")
                        else:
                            logger.info(f"Epoch[{str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}], iter[{str(batch_id+1).zfill(3)}/{total_batch}], loss_total:{loss.item():.5f}")
            return train_loss, train_matrix
        elif flag == 'valid':
            n_loops_test = self.cfg['model'].get("num_loops_test", 1)
            self.model.eval()
            if self.cfg['distributed']:
                world_size = dist.get_world_size()
                self.valid_loader.sampler.set_epoch(epoch)

            valid_loss = AVGMeter()
            val_matrix = dict()
            # start data loops
            with torch.no_grad():
                for batch_id, batch in enumerate(self.valid_loader):
                    batch.to(self.device)
                    if self.flag_use_ema_model:
                        pred, loss = self.model(batch, num_loop_infer=n_loops_test, flag_use_ema_infer=flag_infer_ema)
                    else:
                        pred, loss = self.model(batch, num_loop_infer=n_loops_test)
                    cur_matrix = self.matrix(pred, mode='val')
                    # collect performance 1 --- matrix
                    if self.cfg['distributed']:
                        # get all res from multiple gpus 
                        for key in cur_matrix:
                            # tmp_value = cur_matrix[key].clone().detach().requires_grad_(False).cuda()
                            tmp_value = torch.tensor(cur_matrix[key]).cuda()
                            dist.all_reduce(tmp_value)
                            cur_matrix[key] = tmp_value.cpu().item() / world_size
                    if batch_id == 0: # record into val_matrix
                        for key in cur_matrix:
                            val_matrix[key] = AVGMeter()
                    for key in cur_matrix:
                            val_matrix[key].update(cur_matrix[key])
                    # collect performance 2 --- loss
                    if self.cfg['distributed']:
                        tmp_loss = loss.clone().detach()
                        dist.all_reduce(tmp_loss)
                        valid_loss.update(tmp_loss.cpu().item() / world_size)
                    else:
                        valid_loss.update(loss.cpu().item())
            
            return valid_loss, val_matrix
        else:
            raise ValueError(f'flag == {flag} not support, choice[train, valid]')

    
    def train(self):
        if self.ngpus > 1:
            dummy_batch_data = next(iter(self.train_loader))
            dummy_batch_data.to(self.device, non_blocking=True)
            with torch.no_grad():
                if self.flag_use_ema_model:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
                    _ = self.model(dummy_batch_data, num_loop_infer=1, flag_use_ema_infer=True)
                else:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
            
            if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
                logger.info(f'==================== Total number of parameters: {count_parameters(self.model):.3f}M')

            local_rank = int(os.environ["LOCAL_RANK"])
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                #  find_unused_parameters=False
            )
        else:
            dummy_batch_data = next(iter(self.train_loader))
            dummy_batch_data.to(self.device, non_blocking=True)
            with torch.no_grad():
                # _ = self.model(dummy_batch_data, num_loop_infer=1)
                if self.flag_use_ema_model:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
                    _ = self.model(dummy_batch_data, num_loop_infer=1, flag_use_ema_infer=True)
                else:
                    _ = self.model(dummy_batch_data, num_loop_infer=1)
            logger.info(f'==================== Total number of parameters: {count_parameters(self.model):.3f}M')

        
        if not self.resume_training:    
            self.perform_best = np.Infinity
            self.perform_best_ep = -1
            self.start_epoch = 0
            self.perform_best_metrics = {}
        else:
            self.perform_best, self.perform_best_ep, self.start_epoch, self.perform_best_metrics = self._init_training_wt_checkpoint(self.checkpt_path)
        
        local_best = self.perform_best
        local_best_ep = self.perform_best_ep
        local_best_metrics = self.perform_best_metrics
        if self.flag_use_ema_model:
            local_best_ema = self.perform_best
            local_best_ep_ema = self.perform_best_ep
            local_best_metrics_ema =self.perform_best_metrics
        for epoch in range(self.start_epoch, self.cfg['train']['epochs']):
            with Timer(rest_epochs=self.cfg['train']['epochs'] - (epoch + 1)) as timer:
                train_loss, train_matrix = self.exec_epoch(epoch, flag='train')
                valid_loss, val_matrix = self.exec_epoch(epoch, flag='valid')
                if self.flag_use_ema_model:
                    valid_loss_ema, valid_matrix_ema = self.exec_epoch(epoch, flag='valid', 
                                                             flag_infer_ema=True)
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(valid_loss.agg())
                    else:
                        self.scheduler.step()
            if self.flag_use_ema_model:
                local_best, local_best_ep, local_best_ema, local_best_ep_ema,local_best_metrics_ema = self.summary_epoch(epoch,
                                            train_loss, train_matrix,
                                            valid_loss, val_matrix,
                                            timer, local_best, local_best_ep, local_best_metrics,
                                            local_best_ema=local_best_ema, 
                                            local_best_ep_ema=local_best_ep_ema,
                                            local_best_metrics_ema = local_best_metrics_ema,
                                            valid_loss_ema=valid_loss_ema, 
                                            val_matrix_ema=valid_matrix_ema)
            else:
                local_best, local_best_ep, local_best_metrics = self.summary_epoch(epoch,
                                            train_loss, train_matrix,
                                            valid_loss, val_matrix,
                                            timer, 
                                            local_best, local_best_ep,local_best_metrics)

        if (not self.cfg['distributed']) or (self.cfg['distributed'] and dist.get_rank() == 0):
            self.reporter.close()
        return local_best_ep_ema,local_best_metrics_ema

if __name__ == "__main__":
    str2bool = lambda x: x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument('--config', type=str, default='./configs/default.yaml')
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=2, help="Number of timesteps for SRT-GT")
    parser.add_argument("--lambda_reg", type=float, default=0.0005, help="Regularization weight for temporal smoothness")
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as file:
            cfg = yaml.safe_load(file)
        for key, value in vars(args).items():
            if value is not None:
                cfg[key] = value
        cfg['log_path'] = os.path.join(cfg['log_path'], os.path.basename(args.config)[:-5])
        metadata = (cfg['data']['meta']['node'],
                    list(map(tuple, cfg['data']['meta']['edge'])))
        set_random_seed(cfg["seed"] if cfg["seed"] > 0 else 1, deterministic=False)
        if cfg['distributed']:
            rank, word_size = setup_distributed()
            if not os.path.exists(cfg["log_path"]) and rank == 0:
                os.makedirs(cfg["log_path"])
            if rank == 0:
                # curr_timestr = setup_default_logging(cfg["log_path"], False)
                curr_timestr = setup_default_logging_wt_dir(cfg["log_path"])
                cfg["log_path"] = os.path.join(cfg["log_path"], curr_timestr)
                os.makedirs(cfg["log_path"], exist_ok=True)
                csv_path = os.path.join(cfg["log_path"], "out_stat.csv")

                from shutil import copyfile
                output_yaml = os.path.join(cfg["log_path"], "config.yaml")
                copyfile(cfg['config'], output_yaml) 
            else:
                csv_path = None
            if rank == 0:
                logger.info("\n{}".format(pprint.pformat(cfg)))
            # make sure all folder are correctly created at rank == 0
            dist.barrier()
        else:
            if not os.path.exists(cfg["log_path"]):
                os.makedirs(cfg["log_path"])
            # curr_timestr = setup_default_logging(cfg["log_path"], False)
            curr_timestr = setup_default_logging_wt_dir(cfg["log_path"])
            cfg["log_path"] = os.path.join(cfg["log_path"], curr_timestr)
            os.makedirs(cfg["log_path"], exist_ok=True)
            csv_path = os.path.join(cfg["log_path"], "info_{}_stat.csv".format(curr_timestr))

            from shutil import copyfile
            output_yaml = os.path.join(cfg["log_path"], "config.yaml")
            copyfile(cfg['config'], output_yaml)

            logger.info("\n{}".format(pprint.pformat(cfg)))
        log_dir = os.path.join(args.out_dir, 'logs')
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        oven = SubclassOven(cfg, log_dir)
        local_best_ep_ema,local_best_metrics_ema = oven.train()
        local_best_metrics_ema.update({"epoch":local_best_ep_ema})
        final_infos = {
            "IEEE39":{
                "means": local_best_metrics_ema
            }
        }
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
            json.dump(final_infos, f)
    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
