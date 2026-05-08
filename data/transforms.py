# data/transforms.py
import torch
from torchvision import transforms
from torch_geometric.transforms import BaseTransform

class NormalizeGraphContinuous(BaseTransform):
    def __init__(self, node_mean, node_std, edge_mean, edge_std):
        self.node_mean = torch.tensor(node_mean, dtype=torch.float32)
        self.node_std = torch.tensor(node_std, dtype=torch.float32) + 1e-6
        self.edge_mean = torch.tensor(edge_mean, dtype=torch.float32)
        self.edge_std = torch.tensor(edge_std, dtype=torch.float32) + 1e-6

    def __call__(self, data):
        if data.x is not None and data.x.shape[0] > 0:
            data.x[:, 0:3] = (data.x[:, 0:3] - self.node_mean) / self.node_std
            
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
            
        return data

def get_vision_transform(cfg):
    """
    Reads vision normalization stats directly from the nested dataset config.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.vision.mean,
            std=cfg.dataset.vision.std   
        )
    ])

def get_graph_transform(cfg):
    """
    Reads graph normalization stats directly from the nested dataset config.
    """
    if hasattr(cfg.dataset, 'graph') and cfg.dataset.graph is not None:
        return NormalizeGraphContinuous(
            cfg.dataset.graph.node_mean,
            cfg.dataset.graph.node_std, 
            cfg.dataset.graph.edge_mean,
            cfg.dataset.graph.edge_std
        )
    return None
