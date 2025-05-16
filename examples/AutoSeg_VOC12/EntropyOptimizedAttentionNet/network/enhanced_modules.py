import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NormalizedMultiScaleAttention(nn.Module):
    """
    Normalized Multi-Scale Attention (Normalized-MSA) module
    Enhances multi-scale feature representation by balancing computational efficiency with representation strength.
    """
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(NormalizedMultiScaleAttention, self).__init__()
        self.scales = scales
        self.in_channels = in_channels
        
        # Spatial attention convolutions for each scale
        self.spatial_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            ) for _ in range(len(scales))
        ])
        
        # Add edge-aware convolution to better preserve boundary information
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Scale weights for combining features
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        multi_scale_features = []
        
        # Extract edge information
        edge_features = self.edge_conv(x)
        
        for i, scale in enumerate(self.scales):
            # Generate multi-scale feature using pooling
            if scale == 1:
                x_s = x
            else:
                # Downsample using average pooling
                x_s = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            
            # Compute spatial attention
            spatial_attn = self.spatial_convs[i](x_s)
            
            # Compute channel attention with normalization factor
            # Reshape for matrix multiplication
            x_flat = x_s.view(batch_size, channels, -1)  # B x C x HW
            x_t = x_flat.transpose(1, 2)  # B x HW x C
            
            # Normalized channel attention
            norm_factor = math.sqrt(x_flat.size(2))  # sqrt(HW) for normalization
            channel_attn = torch.bmm(x_flat, x_t) / norm_factor  # B x C x C
            channel_attn = F.softmax(channel_attn, dim=2)  # Softmax along the last dimension
            
            # Apply attention
            attended = torch.bmm(channel_attn, x_flat)  # B x C x HW
            attended = attended.view(batch_size, channels, *x_s.size()[2:])  # B x C x H' x W'
            
            # Apply spatial attention
            attended = attended * spatial_attn
            
            # Upsample back to original size if needed
            if scale != 1:
                attended = F.interpolate(attended, size=(height, width), mode='bilinear', align_corners=False)
            
            multi_scale_features.append(attended)
        
        # Combine multi-scale features with learnable weights
        weighted_features = []
        for i, feature in enumerate(multi_scale_features):
            weighted_features.append(feature * self.scale_weights[i])
        
        # Sum weighted features
        output = torch.stack(weighted_features, dim=0).sum(dim=0)
        
        # Add edge features with a small weight to preserve boundary information
        output = output + 0.1 * edge_features
        
        return output

class EntropyOptimizedGating(nn.Module):
    """
    Entropy-Optimized Gating (EOG) module
    Feature redundancy is adaptively suppressed using a normalized entropy function.
    """
    def __init__(self, channels, beta=0.3, epsilon=1e-5):  # Reduced beta threshold to be less aggressive
        super(EntropyOptimizedGating, self).__init__()
        self.channels = channels
        self.beta = nn.Parameter(torch.tensor([beta]))  # Learnable threshold
        self.epsilon = epsilon
        # Add a small residual connection to preserve some original features
        self.residual_weight = nn.Parameter(torch.tensor([0.2]))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate normalized entropy for each channel
        entropies = []
        gates = []
        
        for c in range(channels):
            # Extract channel
            channel = x[:, c, :, :]  # B x H x W
            
            # Calculate normalized probability distribution
            abs_channel = torch.abs(channel)
            sum_abs = torch.sum(abs_channel, dim=(1, 2), keepdim=True) + self.epsilon
            norm_prob = abs_channel / sum_abs  # B x H x W
            
            # Calculate entropy
            # Add epsilon to avoid log(0)
            log_prob = torch.log(norm_prob + self.epsilon)
            entropy = -torch.sum(norm_prob * log_prob, dim=(1, 2))  # B
            
            # Normalize entropy to [0, 1] range
            max_entropy = math.log(height * width)  # Maximum possible entropy
            norm_entropy = entropy / max_entropy  # B
            
            # Apply gating based on entropy threshold
            gate = (norm_entropy > self.beta).float()  # B
            
            entropies.append(norm_entropy)
            gates.append(gate)
        
        # Stack entropies and gates
        entropies = torch.stack(entropies, dim=1)  # B x C
        gates = torch.stack(gates, dim=1)  # B x C
        
        # Apply gates to channels
        gates = gates.view(batch_size, channels, 1, 1)  # B x C x 1 x 1
        gated_output = x * gates
        
        # Add residual connection to preserve some original features
        output = gated_output + self.residual_weight * x
        
        return output

class EOANetModule(nn.Module):
    """
    Entropy-Optimized Attention Network (EOANet) module
    Combines Normalized Multi-Scale Attention with Entropy-Optimized Gating
    """
    def __init__(self, in_channels, scales=[1, 2, 4], beta=0.5):
        super(EOANetModule, self).__init__()
        self.msa = NormalizedMultiScaleAttention(in_channels, scales)
        self.eog = EntropyOptimizedGating(in_channels, beta)
        
    def forward(self, x):
        # Apply normalized multi-scale attention
        x_msa = self.msa(x)
        
        # Apply entropy-optimized gating
        x_eog = self.eog(x_msa)
        
        return x_eog