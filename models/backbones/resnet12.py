# models/backbones/resnet12.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock(nn.Module):
    """
    Vectorized, GPU-optimized DropBlock (Ghiasi et al., NeurIPS 2018).
    
    Replaces slow CPU-synchronized tensor indexing with native 2D MaxPool dilation.
    Mathematically identical to standard DropBlock while remaining fully robust
    against zero-size tensor indexing crashes on small batch sizes.
    """
    def __init__(self, block_size: int = 5):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
        if not self.training or gamma <= 0.0:
            return x

        batch_size, channels, height, width = x.shape

        # 1. Sample Bernoulli drop seeds directly on the target GPU
        mask = (torch.rand(batch_size, 1, height, width, device=x.device) < gamma).float()

        # 2. Expand seed points into block_size x block_size square drop zones
        padding = self.block_size // 2
        block_mask = 1.0 - F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=padding
        )

        # Boundary alignment for even/odd spatial dimensions
        if block_mask.shape[-2:] != x.shape[-2:]:
            block_mask = block_mask[:, :, :height, :width]

        # 3. Normalize activations to preserve expected magnitude
        count_total = block_mask.numel()
        count_ones = block_mask.sum().clamp(min=1.0)
        normalize_factor = count_total / count_ones

        return x * block_mask * normalize_factor


class BasicBlock(nn.Module):
    """
    Standard 3-convolution residual block for ResNet-12.
    """
    def __init__(self, in_planes, planes, keep_prob=1.0, block_size=5):
        super(BasicBlock, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

        # 3-conv sequence per block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsample shortcut connection
        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.dropblock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply residual addition
        out = out + residual
        out = self.relu(out)
        out = self.maxpool(out)

        # DropBlock applied after pooling
        if self.keep_prob < 1.0:
            gamma = (1.0 - self.keep_prob)
            out = self.dropblock(out, gamma=gamma)

        return out


class ResNet12(nn.Module):
    """
    Standard ResNet-12 backbone for Few-Shot Learning (TADAM, MetaOptNet, RFS, DeepEMD, FRN).
    
    Architecture:
      - 4 Residual Blocks with channel progression: [64, 160, 320, 640]
      - DropBlock enabled on Block 3 and Block 4 with keep_prob=0.9
      - Input resolution: (B, 3, 84, 84) -> Spatial Output: (B, 640, 5, 5)
    """
    def __init__(self, keep_prob=0.9, block_size=5):
        super(ResNet12, self).__init__()
        self.in_planes = 3

        # 4 Residual Blocks
        self.layer1 = BasicBlock(self.in_planes, 64, keep_prob=1.0, block_size=block_size)
        self.layer2 = BasicBlock(64, 160, keep_prob=1.0, block_size=block_size)
        self.layer3 = BasicBlock(160, 320, keep_prob=keep_prob, block_size=block_size)
        self.layer4 = BasicBlock(320, 640, keep_prob=keep_prob, block_size=block_size)

        # Weight initialization following standard PyTorch practices
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Extracts spatial feature map representation.
        Input:  (B, 3, 84, 84)
        Output: (B, 640, 5, 5)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet12(keep_prob=0.9, block_size=5, **kwargs):
    """
    Constructs a standard ResNet-12 backbone.
    """
    return ResNet12(keep_prob=keep_prob, block_size=block_size)
