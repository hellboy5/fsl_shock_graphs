# data/transforms.py
import math
import torch
from torchvision import transforms
from torch_geometric.transforms import BaseTransform


class NormalizeShockGraph(BaseTransform):
    """
    On-the-fly normalization transform for Shock Graphs.
    Applies image-relative centering, log transformations, and Z-score standardization.
    """
    def __init__(self, image_size, node_mean, node_std, edge_mean, edge_std):
        super().__init__()
        self.image_size = float(image_size)
        self.diag = math.sqrt(2.0) * self.image_size
        self.area = self.image_size * self.image_size

        # Continuous node features: [x, y, t] (safely cast from OmegaConf lists)
        self.node_mean = torch.tensor(list(node_mean), dtype=torch.float32)
        self.node_std = torch.tensor(list(node_std), dtype=torch.float32) + 1e-6

        # Edge features: 14 dimensions
        self.edge_mean = torch.tensor(list(edge_mean), dtype=torch.float32)
        self.edge_std = torch.tensor(list(edge_std), dtype=torch.float32) + 1e-6

    def __call__(self, data):
        # -------------------------------------------------------------------
        # 1. NODE NORMALIZATION (11 Dimensions)
        # -------------------------------------------------------------------
        if data.x is not None and data.x.shape[0] > 0:
            x = data.x.clone()
            device = x.device

            # (a) Coordinates: Center and map to [-0.5, 0.5]
            x[:, 0] = (x[:, 0] - (self.image_size / 2.0)) / self.image_size
            x[:, 1] = (x[:, 1] - (self.image_size / 2.0)) / self.image_size

            # (b) Local thickness t: scale by diagonal, log transform, standardize
            log_t = torch.log(torch.clamp(x[:, 2] / self.diag, min=0.0) + 1e-5)
            x[:, 2] = (log_t - self.node_mean[2].to(device)) / self.node_std[2].to(device)

            # (c) Categorical features (indices 3 to 10) remain binary {0.0, 1.0}
            data.x = x

        # -------------------------------------------------------------------
        # 2. EDGE NORMALIZATION (14 Dimensions)
        # -------------------------------------------------------------------
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            e = data.edge_attr.clone()
            device = e.device

            # (a) Lengths and Thicknesses (indices 0, 3, 6, 10, 11): scale by D, log
            len_thick_idx = [0, 3, 6, 10, 11]
            e[:, len_thick_idx] = torch.log(
                torch.clamp(e[:, len_thick_idx] / self.diag, min=0.0) + 1e-5
            )

            # (b) Bounded Polygon Area (index 9): scale by Area, log
            e[:, 9] = torch.log(torch.clamp(e[:, 9] / self.area, min=0.0) + 1e-5)

            # (c) Curvatures, Angles, Flare (indices 1, 2, 4, 5, 7, 8, 13): log1p
            curve_angle_idx = [1, 2, 4, 5, 7, 8, 13]
            e[:, curve_angle_idx] = torch.log1p(torch.clamp(e[:, curve_angle_idx], min=0.0))

            # (d) Standardize using broadcast vectors
            mean_vec = self.edge_mean.to(device)
            std_vec = self.edge_std.to(device)
            data.edge_attr = (e - mean_vec) / std_vec

        return data


def get_vision_transform(cfg):
    """Reads vision normalization stats directly from the nested dataset config."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.dataset.vision.mean,
            std=cfg.dataset.vision.std
        )
    ])


def get_graph_transform(cfg):
    """Instantiates the shock graph transform using the dataset config."""
    if hasattr(cfg.dataset, "graph") and cfg.dataset.graph is not None:
        return NormalizeShockGraph(
            image_size=getattr(cfg.dataset.graph, "image_size", cfg.dataset.vision.image_size),
            node_mean=cfg.dataset.graph.node_mean,
            node_std=cfg.dataset.graph.node_std,
            edge_mean=cfg.dataset.graph.edge_mean,
            edge_std=cfg.dataset.graph.edge_std,
        )
    return None
