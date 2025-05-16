import os
import json
import time
import argparse
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
import traceback
import numpy as np
from harcnet import AdaptiveAugmentation, TemporalConsistencyRegularization

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MILESTONES = [60, 120, 160]


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(x))
        skip_x = x if isinstance(self.shortcut, nn.Identity) else out

        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(skip_x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout_rate):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6

        n_stages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_wide_stage(WideBasicBlock, n_stages[0], n_stages[1], n, dropout_rate, stride=1)
        self.stage2 = self._make_wide_stage(WideBasicBlock, n_stages[1], n_stages[2], n, dropout_rate, stride=2)
        self.stage3 = self._make_wide_stage(WideBasicBlock, n_stages[2], n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(n_stages[3], num_classes)

        self._init_params()

    @staticmethod
    def _make_wide_stage(block, in_planes, out_planes, num_blocks, dropout_rate, stride):
        stride_list = [stride] + [1] * (int(num_blocks) - 1)
        in_planes_list = [in_planes] + [out_planes] * (int(num_blocks) - 1)
        blocks = []

        for _in_planes, _stride in zip(in_planes_list, stride_list):
            blocks.append(block(_in_planes, out_planes, dropout_rate, _stride))

        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def wide_resnet_28_10_old():
    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=100,
        dropout_rate=0.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="run_5")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--data_root", type=str, default='./datasets/imagenet')
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--val_per_epoch", type=int, default=5)
    # HARCNet parameters
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for variance in adaptive augmentation")
    parser.add_argument("--beta", type=float, default=0.6, help="Weight for entropy in adaptive augmentation")
    parser.add_argument("--gamma", type=float, default=2.2, help="Scaling factor for MixUp interpolation")
    parser.add_argument("--memory_size", type=int, default=5, help="Number of past predictions to store")
    parser.add_argument("--decay_rate", type=float, default=2.0, help="Decay rate for temporal consistency")
    parser.add_argument("--consistency_weight", type=float, default=0.05, help="Weight for consistency loss")
    parser.add_argument("--auxiliary_weight", type=float, default=0.05, help="Weight for auxiliary loss")
    parser.add_argument("--use_adaptive_aug", type=bool, default=True, help="Use adaptive augmentation")
    parser.add_argument("--use_temporal_consistency", type=bool, default=True, help="Use temporal consistency")
    config = parser.parse_args()


    try: 
        final_infos = {}
        all_results = {}

        pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

        model = wide_resnet_28_10_old().cuda()
        
        # Initialize HARCNet components
        adaptive_aug = AdaptiveAugmentation(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma
        )
        
        temporal_consistency = TemporalConsistencyRegularization(
            memory_size=config.memory_size,
            decay_rate=config.decay_rate,
            consistency_weight=config.consistency_weight
        )
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        
        train_dataset = datasets.CIFAR100(root=config.data_root, train=True,
                                        download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=config.data_root, train=False,
                                        download=True, transform=transform_test)
        
        # Create a dataset wrapper that provides sample indices
        class IndexedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                
            def __getitem__(self, index):
                data, target = self.dataset[index]
                return data, target, index
                
            def __len__(self):
                return len(self.dataset)
        
        indexed_train_dataset = IndexedDataset(train_dataset)
        
        train_loader = DataLoader(indexed_train_dataset, shuffle=True, num_workers=config.num_workers, batch_size=config.batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, num_workers=config.num_workers, batch_size=config.batch_size)

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * config.max_epoch)

        best_acc = 0.0
        start_time = time.time()
        for cur_epoch in tqdm(range(1, config.max_epoch + 1)):
            model.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_consistency_loss = 0.0
            
            for batch_idx, (images, labels, indices) in enumerate(tqdm(train_loader)):
                images, labels, indices = images.cuda(), labels.cuda(), indices.cuda()
                
                # Apply adaptive augmentation if enabled
                if config.use_adaptive_aug:
                    # First forward pass to get predictions for adaptive augmentation
                    with torch.no_grad():
                        initial_outputs = model(images)
                        initial_probs = F.softmax(initial_outputs, dim=1)
                    
                    # Apply MixUp with adaptive coefficient
                    if np.random.rand() < 0.5:  # Apply MixUp with 50% probability
                        mixed_images, labels_a, labels_b, lam = adaptive_aug.apply_mixup(images, labels, num_classes=100)
                        images = mixed_images
                        
                        # Forward pass with mixed images
                        outputs = model(images)
                        
                        # MixUp loss
                        cls_loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        # Forward pass without MixUp
                        outputs = model(images)
                        cls_loss = criterion(outputs, labels)
                else:
                    # Standard forward pass without adaptive augmentation
                    outputs = model(images)
                    cls_loss = criterion(outputs, labels)
                
                # Compute consistency loss if enabled
                consistency_loss = torch.tensor(0.0).cuda()
                if config.use_temporal_consistency:
                    # Get softmax probabilities
                    probs = F.softmax(outputs, dim=1)
                    
                    # Update prediction history
                    temporal_consistency.update_history(indices, probs)
                    
                    # Compute consistency loss
                    consistency_loss = temporal_consistency.compute_consistency_loss(probs, indices)
                
                # Total loss
                loss = cls_loss + config.consistency_weight * consistency_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Track losses
                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_consistency_loss += consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else 0
            
            # Calculate average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_cls_loss = epoch_cls_loss / len(train_loader)
            avg_consistency_loss = epoch_consistency_loss / len(train_loader)
            
            print(f'Epoch {cur_epoch} - Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Consistency Loss: {avg_consistency_loss:.4f}')
            print(f'Finished epoch {cur_epoch} training.')

            if (cur_epoch % config.val_per_epoch == 0 and cur_epoch != 0) or cur_epoch == (config.max_epoch - 1):
                model.eval()
                correct = 0.0
                for images, labels in tqdm(test_loader):
                    images, labels = images.cuda(), labels.cuda()
                    with torch.no_grad():
                        outputs = model(images)

                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum()
                cur_acc = correct.float() / len(test_loader.dataset)
                print(f"Epoch: {cur_epoch}, Accuracy: {correct.float() / len(test_loader.dataset)}")

                if cur_acc > best_acc:
                    best_acc = cur_acc
                    best_epoch = cur_epoch
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'best.pth'))

        final_infos = {
            "cifar100": {
                "means": {
                    "best_acc": best_acc.item(),
                    "epoch": best_epoch
                },
                "config": {
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "gamma": config.gamma,
                    "memory_size": config.memory_size,
                    "decay_rate": config.decay_rate,
                    "consistency_weight": config.consistency_weight,
                    "auxiliary_weight": config.auxiliary_weight,
                    "use_adaptive_aug": config.use_adaptive_aug,
                    "use_temporal_consistency": config.use_temporal_consistency
                }
            }
        }

        with open(os.path.join(config.out_dir, "final_info.json"), "w") as f:
            json.dump(final_infos, f)

    except Exception as e:
        print("Original error in subprocess:", flush=True)
        traceback.print_exc(file=open(os.path.join(config.out_dir, "traceback.log"), "w"))
        raise
