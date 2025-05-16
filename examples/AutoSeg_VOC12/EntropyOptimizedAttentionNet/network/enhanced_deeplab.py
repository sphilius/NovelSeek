import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
from ._deeplab import ASPPConv, ASPPPooling, ASPP, AtrousSeparableConvolution
from .enhanced_modules import EOANetModule


class EnhancedDeepLabV3(_SimpleSegmentationModel):
    """
    Implements Enhanced DeepLabV3 model with Normalized Multi-Scale Attention and Entropy-Optimized Gating.
    """
    pass


class EnhancedDeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], 
                 use_eoaNet=True, msa_scales=[1, 2, 4], eog_beta=0.5):
        super(EnhancedDeepLabHeadV3Plus, self).__init__()
        self.use_eoaNet = use_eoaNet
        
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        # Add EOANet module after ASPP if enabled
        if self.use_eoaNet:
            self.eoaNet = EOANetModule(256, scales=msa_scales, beta=eog_beta)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        
        # Apply EOANet if enabled
        if self.use_eoaNet:
            output_feature = self.eoaNet(output_feature)
            
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EnhancedDeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36], 
                 use_eoaNet=True, msa_scales=[1, 2, 4], eog_beta=0.5):
        super(EnhancedDeepLabHead, self).__init__()
        self.use_eoaNet = use_eoaNet

        self.aspp = ASPP(in_channels, aspp_dilate)
        
        # Add EOANet module after ASPP if enabled
        if self.use_eoaNet:
            self.eoaNet = EOANetModule(256, scales=msa_scales, beta=eog_beta)
            
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        output = self.aspp(feature['out'])
        
        # Apply EOANet if enabled
        if self.use_eoaNet:
            output = self.eoaNet(output)
            
        return self.classifier(output)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module