# data/transforms.py
import torch
from torch_geometric.transforms import BaseTransform

class NormalizeGraphContinuous(BaseTransform):
    def __init__(self, node_mean, node_std, edge_mean, edge_std):
        self.node_mean = torch.tensor(node_mean, dtype=torch.float32)
        # Add epsilon to prevent division by zero
        self.node_std = torch.tensor(node_std, dtype=torch.float32) + 1e-6
        
        self.edge_mean = torch.tensor(edge_mean, dtype=torch.float32)
        self.edge_std = torch.tensor(edge_std, dtype=torch.float32) + 1e-6

    def __call__(self, data):
        # 1. Normalize Node Features (Indices 0, 1, 2 only)
        if data.x is not None and data.x.shape[0] > 0:
            data.x[:, 0:3] = (data.x[:, 0:3] - self.node_mean) / self.node_std
            
        # 2. Normalize Edge Features (All 14 indices)
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
            
        return data
