import argparse
import logging
import os
import sys
import json
import re
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning import LightningModule

from visnet import datasets, models, priors
from visnet.data import DataModule
from visnet.models import output_modules
from visnet.utils import LoadFromCheckpoint, LoadFromFile, number, save_argparse

from typing import Optional, Tuple , List
from metrics import calculate_mae
from visnet.models.utils import (
    CosineCutoff,
    Distance, 
    EdgeEmbedding,
    NeighborEmbedding, 
    Sphere, 
    VecLayerNorm,
    act_class_mapping, 
    rbf_class_mapping,
    ExpNormalSmearing, 
    GaussianSmearing
)

"""
Models
"""
class ViSNetBlock(nn.Module):

    def __init__(
        self,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        vertex_type="Edge",
    ):
        super(ViSNetBlock, self).__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
    
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors, loop=True)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels).jittable()

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads, 
            hidden_channels=hidden_channels, 
            activation=activation, 
            attn_activation=attn_activation, 
            cutoff=cutoff, 
            vecnorm_type=vecnorm_type, 
            trainable_vecnorm=trainable_vecnorm
        )
        vis_mp_class = VIS_MP_MAP.get(vertex_type, ViS_MP)
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs).jittable()
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(vis_mp_class(last_layer=True, **vis_mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()
        
    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        
        z, pos, batch = data.z, data.pos, data.batch
        
        # Embedding Layers
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)
        
        # ViS-MP Layers
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec
        
        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec
    
class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        
        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
        self.reset_parameters()
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
    
class ViS_MP_Vertex_Edge(ViS_MP):
    
    def __init__(
        self, 
        num_heads, 
        hidden_channels, 
        activation, 
        attn_activation, 
        cutoff, 
        vecnorm_type, 
        trainable_vecnorm, 
        last_layer=False
    ):
        super().__init__(num_heads, hidden_channels, activation, attn_activation, cutoff, vecnorm_type, trainable_vecnorm, last_layer)
        
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        
        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1)

        return f1 * w_dot + f2 * t_dot

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None
    
class ViS_MP_Vertex_Node(ViS_MP):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super().__init__(num_heads, hidden_channels, activation, attn_activation, cutoff, vecnorm_type, trainable_vecnorm, last_layer)

        self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 4)
        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out, t_dot = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3, o4 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + t_dot * o3 + o4
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def message(self, q_i, k_j, v_j, vec_i, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j, t_dot

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec, t_dot = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        t_dot = scatter(t_dot, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec, t_dot
    
VIS_MP_MAP = {'Node': ViS_MP_Vertex_Node, 'Edge': ViS_MP_Vertex_Edge, 'None': ViS_MP}

def create_model(args, prior_model=None, mean=None, std=None):
    visnet_args = dict(
        lmax=args["lmax"],
        vecnorm_type=args["vecnorm_type"],
        trainable_vecnorm=args["trainable_vecnorm"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        hidden_channels=args["embedding_dimension"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        attn_activation=args["attn_activation"],
        max_z=args["max_z"],
        cutoff=args["cutoff"],
        max_num_neighbors=args["max_num_neighbors"],
        vertex_type=args["vertex_type"],
    )

    # representation network
    if args["model"] == "ViSNetBlock":
        representation_model = ViSNetBlock(**visnet_args)
    else:
        raise ValueError(f"Unknown model {args['model']}.")
    
    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant"
    output_model = getattr(output_modules, output_prefix + args["output_model"])(args["embedding_dimension"], args["activation"])

    model = ViSNet(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
    )
    return model


def load_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            rank_zero_warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    
    return model.to(device)


class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                "Prior model was given but the output model does "
                "not allow prior models. Dropping the prior model."
            )

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        
        if self.derivative:
            data.pos.requires_grad_(True)

        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, data.z)

        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)
        
        out = out + self.mean

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy
        return out, None
    
class LNNP(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        self._reset_losses_dict()
        self._reset_ema_dict()
        self._reset_inference_results()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        loss_fn = mse_loss if self.hparams.loss_type == 'MSE' else l1_loss
        
        return self.step(batch, loss_fn, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            pred, deriv = self(batch)
        if stage == "test":
            self.inference_results['y_pred'].append(pred.squeeze(-1).detach().cpu())
            self.inference_results['y_true'].append(batch.y.squeeze(-1).detach().cpu())
            if self.hparams.derivative:
                self.inference_results['dy_pred'].append(deriv.squeeze(-1).detach().cpu())
                self.inference_results['dy_true'].append(batch.dy.squeeze(-1).detach().cpu())

        loss_y, loss_dy = 0, 0
        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0

            loss_dy = loss_fn(deriv, batch.dy)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.loss_scale_dy * loss_dy
                    + (1 - self.hparams.loss_scale_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            loss_y = loss_fn(pred, batch.y)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.loss_scale_y * loss_y
                    + (1 - self.hparams.loss_scale_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()
            
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight
        
        self.losses[stage].append(loss.detach())
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            delta = 0 if self.hparams.reload == 1 else 1
            should_reset = (
                (self.current_epoch + delta + 1) % self.hparams.test_interval == 0
                or ((self.current_epoch + delta) % self.hparams.test_interval == 0 and self.current_epoch != 0)
            )
            if should_reset:
                self.trainer.reset_val_dataloader()
                self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop._reset_dl_batch_idx(len(self.trainer.val_dataloaders))

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(self.losses["train_dy"]).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

            if len(self.losses["test_y"]) > 0 and len(self.losses["test_dy"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
                result_dict["test_loss_dy"] = torch.stack(self.losses["test_dy"]).mean()

            self.log_dict(result_dict, sync_dist=True)
            
        self._reset_losses_dict()
        self._reset_inference_results()

    def test_epoch_end(self, outputs) -> None:
        for key in self.inference_results.keys():
            if len(self.inference_results[key]) > 0:
                self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)

    def _reset_losses_dict(self):
        self.losses = {
            "train": [], "val": [], "test": [],
            "train_y": [], "val_y": [], "test_y": [],
            "train_dy": [], "val_dy": [], "test_dy": [],
        }

    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [], 'y_true': [], 'dy_pred': [], 'dy_true': []}
        
    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    
    # training settings
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--loss-type', type=str, default='MSE', choices=['MSE', 'MAE'], help='Loss type')
    parser.add_argument('--loss-scale-y', type=float, default=1.0, help="Scale the loss y of the target")
    parser.add_argument('--loss-scale-dy', type=float, default=1.0, help="Scale the loss dy of the target")
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')
    
    # dataset specific
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument')
    parser.add_argument('--dataset-root', default=None, type=str, help='Data storage directory')
    parser.add_argument('--derivative', default=False, action=argparse.BooleanOptionalAction, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--split-mode', default=None, type=str, help='Split mode for Molecule3D dataset')
    
    # dataloader specific
    parser.add_argument('--reload', type=int, default=0, help='Reload dataloaders every n epoch')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=950, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=50, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=None, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    
    # model architecture specific
    parser.add_argument('--model', type=str, default='ViSNetBlock', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')
    parser.add_argument('--prior-args', type=dict, default=None, help='Additional arguments for the prior model')
    
    # architectural specific
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', action=argparse.BooleanOptionalAction, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff in model')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    parser.add_argument('--lmax', type=int, default=2, help='Max order of spherical harmonics')
    parser.add_argument('--vecnorm-type', type=str, default='max_min', help='Type of vector normalization')
    parser.add_argument('--trainable-vecnorm', action=argparse.BooleanOptionalAction, default=False, help='If vector normalization should be trainable')
    parser.add_argument('--vertex-type', type=str, default='Edge', choices=['None', 'Edge', 'Node'], help='If add vertex angle and Where to add vertex angles')

    # other specific
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', type=str, default="aspirin_log", help='Log directory')
    parser.add_argument('--task', type=str, default='train', choices=['train', 'inference'], help='Train or inference') 
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend')
    parser.add_argument('--redirect', action=argparse.BooleanOptionalAction, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--accelerator', default='gpu', help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=2, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument("--out_dir", type=str, default="run_0")
    
    args = parser.parse_args()

    if args.redirect:
        os.makedirs(args.log_dir, exist_ok=True)
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(logging.StreamHandler(sys.stdout))

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])
    
    return args

def main(args):
    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = DataModule(args)
    data.prepare_dataset()

    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    dir_name = f"output_ngpus_{len(cuda_visible_devices)}_bs_{args.batch_size}_lr_{args.lr}_seed_{args.seed}" + \
               f"_reload_{args.reload}_lmax_{args.lmax}_vnorm_{args.vecnorm_type}" + \
               f"_vertex_{args.vertex_type}_L{args.num_layers}_D{args.embedding_dimension}_H{args.num_heads}" + \
               f"_cutoff_{args.cutoff}_E{args.energy_weight}_F{args.force_weight}_loss_{args.loss_type}"
    
    if args.load_model is None:
        args.log_dir = os.path.join(args.out_dir, args.log_dir , dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(os.path.join(args.log_dir, "last.ckpt")):
                args.load_model = os.path.join(args.log_dir, "last.ckpt")
            csv_path = os.path.join(args.log_dir, "metrics.csv")
            while os.path.exists(csv_path):
                csv_path = csv_path + '.bak'
            if os.path.exists(os.path.join(args.log_dir, "metrics.csv")):
                os.rename(os.path.join(args.log_dir, "metrics.csv"), csv_path)

    prior = None
    if args.prior_model:
        assert hasattr(priors, args.prior_model), (
            f"Unknown prior model {args['prior_model']}. "
            f"Available models are {', '.join(priors.__all__)}"
        )
        # initialize the prior model
        prior = getattr(priors, args.prior_model)(dataset=data.dataset)
        args.prior_args = prior.get_init_args()

    # initialize lightning module
    model = LNNP(args, prior_model=prior, mean=data.mean, std=data.std)

    if args.task == "train":
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.log_dir,
            monitor="val_loss",
            save_top_k=2,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_loss:.4f}-{test_loss:.4f}",
        )
        
        early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience)
        
        tb_logger = TensorBoardLogger(os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/"), name="", version="", default_hp_metric=False)
        csv_logger = CSVLogger(args.log_dir, name="", version="")
        ddp_plugin = DDPStrategy(find_unused_parameters=False)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            gpus=args.ngpus,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            default_root_dir=args.log_dir,
            auto_lr_find=False,
            callbacks=[early_stopping, checkpoint_callback],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=ddp_plugin,
            enable_progress_bar=True,
        )

        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)

    test_trainer = pl.Trainer(
        logger=False,
        max_epochs=-1,
        num_nodes=1,
        gpus=1,
        default_root_dir=args.log_dir,
        enable_progress_bar=True,
        inference_mode=False,
    )
        
    if args.task == 'train':
        test_trainer.test(model=model, ckpt_path=trainer.checkpoint_callback.best_model_path, datamodule=data)
    elif args.task == 'inference':
        test_trainer.test(model=model, datamodule=data)
        torch.save(model.inference_results, os.path.join(args.log_dir, "inference_results.pt"))
    
    emae = calculate_mae(model.inference_results['y_true'].numpy(), model.inference_results['y_pred'].numpy())
    Scalar_MAE = "{:.6f}".format(emae)
    print('Scalar MAE: {:.6f}'.format(emae))

    final_infos = {
        "AutoMolecule3D":{
            "means":{
                "Scalar MAE": Scalar_MAE
            }
        }
    }

    if args.derivative:
        fmae = calculate_mae(model.inference_results['dy_true'].numpy(), model.inference_results['dy_pred'].numpy())
        Forces_MAE = "{:.6f}".format(fmae)
        print('Forces MAE: {:.6f}'.format(fmae))
        final_infos["AutoMolecule3D"]["means"]["Forces MAE"] = Forces_MAE

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except Exception as e:
        print("Origin error in main process:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
