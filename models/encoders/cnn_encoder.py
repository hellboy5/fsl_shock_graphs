import torch
import torch.nn as nn
from models.backbones.resnet12 import resnet12


class VisionEncoder(nn.Module):
    def __init__(self, proj_feat_dim=640, drop_rate=0.1, **kwargs):
        """
        Args:
            proj_feat_dim: Accepted for interface compatibility with multimodal_network
            drop_rate: Dropout rate for the ResNet backbone
        """
        super().__init__()
        
        # Instantiate the FSL ResNet-12
        self.cnn = resnet12(drop_rate=drop_rate)
        
        # Pool the spatial feature map to a single vector per image (native 640D)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # 1. Extract Dense Spatial Features [Batch, 640, 5, 5]
        spatial_features = self.cnn(x)
        
        # 2. Global Average Pooling -> [Batch, 640]
        pooled_features = self.avgpool(spatial_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        return pooled_features
