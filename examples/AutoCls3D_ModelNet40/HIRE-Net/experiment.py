import os
from tqdm import tqdm
import pickle
import argparse
import pathlib
import json
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from metrics import ConfusionMatrix
import data_transforms
import argparse
import random
import traceback

"""
Model
"""
class STN3d(nn.Module):
    def __init__(self, in_channels):
        super(STN3d, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9)
        )
        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.linear_layers(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k * k)
        )
        self.k = k
        self.iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).reshape(1, self.k * self.k)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.linear_layers(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class EnhancedSTN(nn.Module):
    """
    Enhanced Spatial Transformer Network with improved rotation equivariance.
    """
    def __init__(self, in_channels):
        super(EnhancedSTN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9)
        )
        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)
        
        # Orthogonality regularization weight
        self.ortho_weight = 0.01

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv_layers(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.linear_layers(x)
        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        
        # Apply soft orthogonality constraint to ensure rotation matrix properties
        # This helps maintain rotation equivariance
        ortho_loss = torch.mean(torch.norm(
            torch.bmm(x, x.transpose(2, 1)) - torch.eye(3, device=x.device).unsqueeze(0), dim=(1, 2)
        ))
        
        return x, self.ortho_weight * ortho_loss

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, in_channels=3, num_alignments=2):
        super(PointNetEncoder, self).__init__()

        self.stn = EnhancedSTN(in_channels)
        

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        

        self.ortho_loss = 0

    def forward(self, x):
        B, D, N = x.size()
        
        trans, ortho_loss = self.stn(x)
        self.ortho_loss = ortho_loss
        
        x_aligned = x.transpose(2, 1)
        if D > 3:
            feature = x_aligned[:, :, 3:]
            coords = x_aligned[:, :, :3]
            coords = torch.bmm(coords, trans)
            x_aligned = torch.cat([coords, feature], dim=2)
        else:
            x_aligned = torch.bmm(x_aligned, trans)
        x_aligned = x_aligned.transpose(2, 1)
        

        x = self.conv_layer1(x_aligned)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        graph = construct_graph(x, args.k)
        context_features = compute_context_aware_features(x, graph)
        x = x + context_features
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat



def construct_graph(points, k):
    """
    Construct a dynamic graph where nodes represent points and edges capture semantic similarities.
    """
    # Compute pairwise distances
    dist = torch.cdist(points, points)
    # Get the top k neighbors
    _, indices = torch.topk(dist, k, largest=False, dim=1)
    return indices

def compute_attention_weights(points, graph, epsilon=0.01):
    """
    Compute attention weights with energy-based normalization for numerical stability.
    Improved implementation with better numerical stability and efficiency.
    
    Args:
        points: Input feature points [B, N, C]
        graph: Neighborhood indices [B, N, K]
        epsilon: Regularization parameter for bounded energy constraints
        
    Returns:
        Attention weights that satisfy bounded energy constraints
    """
    num_points = points.shape[0]
    k = graph.shape[1]
    attention_weights = torch.zeros(num_points, k, device=points.device)
    
    for i in range(num_points):
        neighbors = graph[i]
        
        center_feat = points[i].unsqueeze(0)  # [1, C]
        neighbor_feats = points[neighbors]     # [k, C]
        
        center_norm = torch.norm(center_feat, dim=1, keepdim=True)
        neighbor_norms = torch.norm(neighbor_feats, dim=1, keepdim=True)
        
        center_norm = torch.clamp(center_norm, min=1e-8)
        neighbor_norms = torch.clamp(neighbor_norms, min=1e-8)
        
        center_feat_norm = center_feat / center_norm
        neighbor_feats_norm = neighbor_feats / neighbor_norms
        
        similarity = torch.sum(center_feat_norm * neighbor_feats_norm, dim=1)
        
        weights = torch.exp(similarity)
        
        norm_const = torch.sum(weights) + 1e-8
        weights = weights / norm_const
        
        sq_sum = torch.sum(weights * weights)
        if sq_sum > epsilon:
            scale_factor = torch.sqrt(epsilon / sq_sum)
            weights = weights * scale_factor
            
        attention_weights[i, :len(neighbors)] = weights
        
    return attention_weights

def compute_context_aware_features(points, graph):
    """
    Compute context-aware feature adjustments using the constructed graph.
    Enhanced with edge-aware attention pooling (EEGA) and improved stability.
    """
    # Calculate weighted edge features
    context_features = torch.zeros_like(points)
    
    # Compute attention weights with energy constraints
    attention_weights = compute_attention_weights(points, graph, epsilon=args.epsilon)
    
    # Calculate weighted edge features
    for i in range(points.size(0)):
        neighbors = graph[i]
        weights = attention_weights[i, :len(neighbors)].unsqueeze(1)
        
        # Calculate weighted edge features (φ_local(p_j) - φ_local(p_i))
        # Using hybrid method: consider both differences and original features
        edge_features = points[neighbors] - points[i].unsqueeze(0)
        neighbor_features = points[neighbors]
        
        # Weight edge features and neighbor features
        weighted_edges = edge_features * weights * 0.5
        weighted_neighbors = neighbor_features * weights * 0.5
        
        # Aggregate features: combine edge differences and neighbor information
        context_features[i] = torch.sum(weighted_edges, dim=0) + torch.sum(weighted_neighbors, dim=0)
    
    return context_features

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=40, scale=0.001, num_alignments=2):
        super().__init__()
        self.mat_diff_loss_scale = scale
        self.in_channels = in_channels
        self.backbone = PointNetEncoder(
            global_feat=True, 
            feature_transform=True, 
            in_channels=in_channels,
            num_alignments=num_alignments
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, gts):

        global_features, trans, trans_feat = self.backbone(x)

        x = self.cls_head(global_features)
        x = F.log_softmax(x, dim=1)
        
        loss = F.nll_loss(x, gts)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        ortho_loss = self.backbone.ortho_loss
        
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale + ortho_loss
        
        return total_loss, x


"""
dataset and normalization
"""
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset(Dataset):
    def __init__(self, data_root, num_category, num_points, split='train'):
        self.root = data_root
        self.npoints = num_points
        self.uniform = True
        self.use_normals = True
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.data_path = os.path.join(data_root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.data_path = os.path.join(data_root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        print('Load processed data from %s...' % self.data_path)
        with open(self.data_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]        
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        return point_set, label[0]


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):

    seed_everything(args.seed)

    final_infos = {}
    all_results = {}

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    datasets, dataloaders = {}, {}
    for split in ['train', 'test']:
        datasets[split] = ModelNetDataset(args.data_root, args.num_category, args.num_points, split)
        dataloaders[split] = DataLoader(datasets[split], batch_size=args.batch_size, shuffle=(split == 'train'),
                                                      drop_last=(split == 'train'), num_workers=8)
    
    model = Model(in_channels=args.in_channels, num_alignments=args.num_alignments).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7
    )
    train_losses = []
    print("Training model...")
    model.train()
    global_step = 0
    cur_epoch = 0
    best_oa = 0
    best_acc = 0

    start_time = time.time()
    for epoch in tqdm(range(args.max_epoch), desc='training'):
        model.train()
        cm = ConfusionMatrix(num_classes=len(datasets['train'].classes))
        for points, target in tqdm(dataloaders['train'], desc=f'epoch {cur_epoch}/{args.max_epoch}'):
            # data transforms
            points = points.data.numpy()
            points = data_transforms.random_point_dropout(points)
            points[:, :, 0:3] = data_transforms.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = data_transforms.shift_point_cloud(points[:, :, 0:3])
            points = torch.from_numpy(points).transpose(2, 1).contiguous()
            
            points, target = points.cuda(), target.long().cuda()
        
            loss, logits = model(points, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
            optimizer.step()
            model.zero_grad()
            
            
            logs = {"loss": loss.detach().item()}
            train_losses.append(loss.detach().item())
            cm.update(logits.argmax(dim=1), target)
        
        scheduler.step()
        end_time = time.time()
        training_time = end_time - start_time
        macc, overallacc, accs = cm.all_acc()
        print(f"iter: {global_step}/{args.max_epoch*len(dataloaders['train'])}, \
              train_macc: {macc}, train_oa: {overallacc}")
        
        if (cur_epoch % args.val_per_epoch == 0 and cur_epoch != 0) or cur_epoch == (args.max_epoch - 1):
            model.eval()
            cm = ConfusionMatrix(num_classes=datasets['test'].num_category)
            pbar = tqdm(enumerate(dataloaders['test']), total=dataloaders['test'].__len__())
            # with torch.no_grad():
            for idx, (points, target) in pbar:
                points, target = points.cuda(), target.long().cuda()
                points = points.transpose(2, 1).contiguous()
                loss, logits = model(points, target)
                cm.update(logits.argmax(dim=1), target)
                
            tp, count = cm.tp, cm.count
            macc, overallacc, accs = cm.cal_acc(tp, count)
            print(f"iter: {global_step}/{args.max_epoch*len(dataloaders['train'])}, \
            val_macc: {macc}, val_oa: {overallacc}")
                
            if overallacc > best_oa:
                best_oa = overallacc
                best_acc = macc
                best_epoch = cur_epoch
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'best.pth'))
        cur_epoch += 1

        print(f"finish epoch {cur_epoch} training")

    final_infos = {
        "modelnet" + str(args.num_category):{
            "means":{
                "best_oa": best_oa,
                "best_acc": best_acc,
                "epoch": best_epoch
            }
        }
    }
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_category", type=int, choices=[10, 40], default=40)
    parser.add_argument("--data_root", type=str, default='./datasets/modelnet40')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--val_per_epoch", type=int, default=5)
    parser.add_argument("--k", type=int, default=16, help="Number of neighbors for graph construction")
    parser.add_argument("--num_alignments", type=int, default=2, help="Number of rotational alignments for RE-MA")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Regularization parameter for attention weights")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise
