# models/encoders/cnn_encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from models.backbones.resnet12 import resnet12


STAGE_CHANNELS = {1: 64, 2: 160, 3: 320, 4: 640}


class VisionEncoder(nn.Module):
    def __init__(
        self,
        proj_feat_dim=640,
        drop_rate=0.1,
        hc_layers=(1, 2, 3),
        hc_radius_mode='disk_5pt',
        hc_disk_scale=0.7,
        hc_stage_l2norm=True,
        **kwargs,
    ):
        """
        Args:
            proj_feat_dim: Accepted for interface compatibility with multimodal_network
            drop_rate: Dropout rate for the ResNet-12 backbone
            hc_layers: Int (e.g. 3 -> [1,2,3]) or list of stage indices in {1,2,3,4}
            hc_radius_mode: 'none' | 'concat_r' | 'disk_5pt' | 'disk_9pt' | 'center_ring_concat'
            hc_disk_scale: Multiplier on node radius r when sampling disk offsets
            hc_stage_l2norm: Whether to L2-normalize each stage slice before concatenation
        """
        super().__init__()
        keep_prob = 1.0 - float(drop_rate)
        self.cnn = resnet12(keep_prob=keep_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Parse hc_layers from Hydra (supports int like 3 or list like [1, 2, 3])
        if isinstance(hc_layers, int):
            self.hc_layers = list(range(1, hc_layers + 1))
        else:
            self.hc_layers = [int(i) for i in hc_layers]

        self.hc_radius_mode = str(hc_radius_mode).lower()
        self.hc_disk_scale = float(hc_disk_scale)
        self.hc_stage_l2norm = bool(hc_stage_l2norm)

        # Compute exact output hypercolumn dimension per node for downstream GNN/MLP
        base_hc_dim = sum(STAGE_CHANNELS[l] for l in self.hc_layers)
        if self.hc_radius_mode == 'concat_r':
            self.hc_out_dim = base_hc_dim + 1
        elif self.hc_radius_mode == 'center_ring_concat':
            self.hc_out_dim = base_hc_dim * 2
        elif self.hc_radius_mode in ('none', 'disk_5pt', 'disk_9pt'):
            self.hc_out_dim = base_hc_dim
        else:
            raise ValueError(f"Unknown hc_radius_mode: {self.hc_radius_mode}")

    def forward(self, x):
        # 1. Extract Dense Spatial Features [Batch, 640, 5, 5]
        spatial_features = self.cnn(x)

        # 2. Global Average Pooling -> [Batch, 640]
        pooled_features = self.avgpool(spatial_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        return pooled_features

    def _build_sampling_points(self, pos_xyr):
        """
        Constructs sampling coordinates [Total_Nodes, K_pts, 2] in [-1, 1] grid space
        from pos_xyr [Total_Nodes, 3] = (grid_x, grid_y, grid_r).
        """
        gx, gy, gr = pos_xyr[:, 0], pos_xyr[:, 1], pos_xyr[:, 2]
        delta = self.hc_disk_scale * gr

        if self.hc_radius_mode in ('none', 'concat_r'):
            xs = gx.unsqueeze(-1)
            ys = gy.unsqueeze(-1)

        elif self.hc_radius_mode in ('disk_5pt', 'center_ring_concat'):
            # Center + 4 cardinal points on the inscribed shock disk
            xs = torch.stack([gx, gx + delta, gx - delta, gx, gx], dim=-1)
            ys = torch.stack([gy, gy, gy, gy + delta, gy - delta], dim=-1)

        elif self.hc_radius_mode == 'disk_9pt':
            # Center + 4 cardinal + 4 diagonal points on the inscribed shock disk
            diag = delta * (1.0 / math.sqrt(2.0))
            xs = torch.stack(
                [gx, gx + delta, gx - delta, gx, gx, gx + diag, gx + diag, gx - diag, gx - diag],
                dim=-1,
            )
            ys = torch.stack(
                [gy, gy, gy, gy + delta, gy - delta, gy + diag, gy - diag, gy + diag, gy - diag],
                dim=-1,
            )

        coords = torch.stack([xs, ys], dim=-1).clamp(-1.0, 1.0)  # [Total_Nodes, K_pts, 2]
        return coords

    def extract_hypercolumns(self, images, pos_xyr, batch_idx):
        """
        Extracts multi-scale node hypercolumns for a batch of PyG shock graphs.

        Args:
            images:    [B, 3, H, W] normalized image tensor
            pos_xyr:   [Total_Nodes, 3] tensor of (grid_x, grid_y, grid_r) in [-1, 1]
            batch_idx: [Total_Nodes] PyG batch index mapping each node to image b in [0, B-1]

        Returns:
            x_app:     [Total_Nodes, self.hc_out_dim] node hypercolumn tensor
        """
        stage_maps = self.cnn.forward_stages(images)
        coords = self._build_sampling_points(pos_xyr)  # [Total_Nodes, K_pts, 2]

        # Pad variable node counts per graph to [B, N_max, K_pts, 2] for batched grid_sample
        coords_dense, mask = to_dense_batch(coords, batch_idx, batch_size=images.size(0))

        stage_feats = []
        for l in self.hc_layers:
            fmap = stage_maps[l]  # [B, C_l, H_l, W_l]
            # sampled: [B, C_l, N_max, K_pts]
            sampled = F.grid_sample(
                fmap,
                coords_dense,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )

            if self.hc_radius_mode == 'center_ring_concat':
                # Separate center [idx 0] and mean of 4 ring points [idx 1..4]
                center_feat = sampled[:, :, :, 0].permute(0, 2, 1)[mask]           # [Total_Nodes, C_l]
                ring_feat = sampled[:, :, :, 1:].mean(dim=-1).permute(0, 2, 1)[mask]  # [Total_Nodes, C_l]
                if self.hc_stage_l2norm:
                    center_feat = F.normalize(center_feat, p=2, dim=-1)
                    ring_feat = F.normalize(ring_feat, p=2, dim=-1)
                node_stage = torch.cat([center_feat, ring_feat], dim=-1)           # [Total_Nodes, 2 * C_l]
            else:
                # Average across the K_pts disk samples -> [Total_Nodes, C_l]
                node_stage = sampled.mean(dim=-1).permute(0, 2, 1)[mask]
                if self.hc_stage_l2norm:
                    node_stage = F.normalize(node_stage, p=2, dim=-1)

            stage_feats.append(node_stage)

        hc = torch.cat(stage_feats, dim=-1)  # [Total_Nodes, base_hc_dim (or 2*base_hc_dim)]

        if self.hc_radius_mode == 'concat_r':
            log_r = torch.log(pos_xyr[:, 2].clamp(min=1e-4)).unsqueeze(-1)
            hc = torch.cat([hc, log_r], dim=-1)

        return hc
