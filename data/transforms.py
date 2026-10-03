# data/transforms.py
import math
import torch
from torchvision import transforms
from torch_geometric.transforms import BaseTransform


class NormalizeShockGraph(BaseTransform):
    """
    Dual-Mode Normalization Transform for Shock Graphs.
    
    Mode 1 (use_graph_local_norm=False) [DEFAULT]:
      - Scales spatial dimensions by global image size and diagonal.
      - Log transforms skewed values.
      - Standardizes using pre-calculated dataset-wide mean and std.
      
    Mode 2 (use_graph_local_norm=True):
      - Scales every continuous feature by its local maximum within that specific graph.
      - Applies log / log1p transforms to preserve fine-grained proportions.
      - Bypasses global dataset-wide mean/std to preserve true scale invariance.
    """
    def __init__(self, image_size, node_mean, node_std, edge_mean, edge_std, use_graph_local_norm=False):
        super().__init__()
        self.image_size = float(image_size)
        self.diag = math.sqrt(2.0) * self.image_size
        self.area = self.image_size * self.image_size
        self.use_graph_local_norm = use_graph_local_norm

        # Continuous node features: [x, y, t]
        self.node_mean = torch.tensor(list(node_mean), dtype=torch.float32)
        self.node_std = torch.tensor(list(node_std), dtype=torch.float32) + 1e-6

        # Edge features: 14 dimensions
        self.edge_mean = torch.tensor(list(edge_mean), dtype=torch.float32)
        self.edge_std = torch.tensor(list(edge_std), dtype=torch.float32) + 1e-6

    def __call__(self, data):
        eps = 1e-6

        # ===================================================================
        # 1. NODE NORMALIZATION (11 Dimensions)
        # ===================================================================
        if data.x is not None and data.x.shape[0] > 0:
            x = data.x.clone()
            device = x.device

            if self.use_graph_local_norm:
                # --- MODE 2: Per-Graph Normalization ---
                # (a) Coordinates: Center and normalize by graph's own bounding radius
                center_x = x[:, 0].mean()
                center_y = x[:, 1].mean()
                x[:, 0] = x[:, 0] - center_x
                x[:, 1] = x[:, 1] - center_y
                max_coord = torch.max(x[:, 0:2].abs()).clamp(min=eps)
                x[:, 0] = x[:, 0] / max_coord
                x[:, 1] = x[:, 1] / max_coord

                # (b) Local thickness t (index 2): Log ratio relative to graph's max thickness
                local_t_max = x[:, 2].abs().max().clamp(min=eps)
                x[:, 2] = torch.log(torch.clamp(x[:, 2] / local_t_max, min=0.0) + 1e-5)

            else:
                # --- MODE 1: Dataset-Wide Normalization (Default / Untouched) ---
                # (a) Coordinates: Center and map to [-0.5, 0.5]
                x[:, 0] = (x[:, 0] - (self.image_size / 2.0)) / self.image_size
                x[:, 1] = (x[:, 1] - (self.image_size / 2.0)) / self.image_size

                # (b) Local thickness t: scale by diagonal, log transform, standardize
                log_t = torch.log(torch.clamp(x[:, 2] / self.diag, min=0.0) + 1e-5)
                x[:, 2] = (log_t - self.node_mean[2].to(device)) / self.node_std[2].to(device)

            data.x = x

        # ===================================================================
        # 2. EDGE NORMALIZATION (14 Dimensions)
        # ===================================================================
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            e = data.edge_attr.clone()
            device = e.device

            if self.use_graph_local_norm:
                # --- MODE 2: Per-Graph Normalization ---
                # (a) Lengths and Thicknesses (indices 0, 3, 6, 10, 11): Log ratio to local max
                len_thick_idx = [0, 3, 6, 10, 11]
                for idx in len_thick_idx:
                    local_max = e[:, idx].abs().max().clamp(min=eps)
                    e[:, idx] = torch.log(torch.clamp(e[:, idx] / local_max, min=0.0) + 1e-5)

                # (b) Bounded Polygon Area (index 9): Log ratio to local max area
                local_area_max = e[:, 9].abs().max().clamp(min=eps)
                e[:, 9] = torch.log(torch.clamp(e[:, 9] / local_area_max, min=0.0) + 1e-5)

                # (c) Curvatures (indices 1, 4, 7): Signed local ratio + signed log1p
                curv_idx = [1, 4, 7]
                for idx in curv_idx:
                    local_curv_max = e[:, idx].abs().max().clamp(min=eps)
                    scaled_curv = e[:, idx] / local_curv_max
                    e[:, idx] = scaled_curv.sign() * torch.log1p(scaled_curv.abs())

                # (d) Angles & Flare (indices 2, 5, 8, 13): Local ratio + log1p
                angle_idx = [2, 5, 8, 13]
                for idx in angle_idx:
                    local_angle_max = e[:, idx].abs().max().clamp(min=eps)
                    e[:, idx] = torch.log1p(torch.clamp(e[:, idx] / local_angle_max, min=0.0))

                # (e) Taper rate (index 12): Local ratio [-1, 1]
                local_taper_max = e[:, 12].abs().max().clamp(min=eps)
                e[:, 12] = e[:, 12] / local_taper_max

                data.edge_attr = e

            else:
                # --- MODE 1: Dataset-Wide Normalization (Default / Untouched) ---
                # (a) Lengths and Thicknesses: scale by D, log
                len_thick_idx = [0, 3, 6, 10, 11]
                e[:, len_thick_idx] = torch.log(
                    torch.clamp(e[:, len_thick_idx] / self.diag, min=0.0) + 1e-5
                )

                # (b) Bounded Polygon Area: scale by Area, log
                e[:, 9] = torch.log(torch.clamp(e[:, 9] / self.area, min=0.0) + 1e-5)

                # (c) Curvatures: Absolute magnitude + log1p
                curv_idx = [1, 4, 7]
                e[:, curv_idx] = torch.log1p(torch.abs(e[:, curv_idx]))

                # (d) Angles & Flare: Non-negative + log1p
                angle_idx = [2, 5, 8, 13]
                e[:, angle_idx] = torch.log1p(torch.clamp(e[:, angle_idx], min=0.0))

                # (e) Standardize using pre-calculated dataset vectors
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
    """
    Instantiates the shock graph transform, automatically selecting 
    coarse or uncoarse stats and checking the use_graph_local_norm flag.
    """
    if hasattr(cfg.dataset, "graph") and cfg.dataset.graph is not None:
        g_cfg = cfg.dataset.graph
        use_coarse = getattr(g_cfg, "use_coarse", False)
        # Read the local normalization flag from Hydra (defaults to False)
        use_graph_local_norm = getattr(g_cfg, "use_graph_local_norm", False)

        if hasattr(g_cfg, "coarse") and hasattr(g_cfg, "uncoarse"):
            stats = g_cfg.coarse if use_coarse else g_cfg.uncoarse
        else:
            stats = g_cfg

        return NormalizeShockGraph(
            image_size=getattr(g_cfg, "image_size", cfg.dataset.vision.image_size),
            node_mean=stats.node_mean,
            node_std=stats.node_std,
            edge_mean=stats.edge_mean,
            edge_std=stats.edge_std,
            use_graph_local_norm=use_graph_local_norm
        )
    return None
