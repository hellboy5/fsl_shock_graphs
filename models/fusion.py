# models/fusion.py
import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """
    Decoupled Multimodal Fusion Module.
    
    Axis 1 (Architecture): 'dual_gate' (Contextual Gating), 'concat', 'add'
    Axis 2 (Optimization): 'modality_dropout' (0.0 to 1.0) to prevent modality dominance
    """
    def __init__(
        self, 
        proj_feat_dim: int = 128, 
        fusion_type: str = 'dual_gate', 
        dropout: float = 0.1,
        modality_dropout: float = 0.0
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.proj_feat_dim = proj_feat_dim
        self.modality_dropout = modality_dropout

        # --- Axis 1: Architecture Definition ---
        if self.fusion_type == 'concat':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(proj_feat_dim * 2, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_feat_dim, proj_feat_dim)
            )

        elif self.fusion_type in ['gate', 'dual_gate']:
            # Independent confidence gating networks (Arevalo et al.; Chen et al., NeurIPS 2025)
            self.gate_v = nn.Sequential(
                nn.Linear(proj_feat_dim, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim),
                nn.Sigmoid()
            )
            self.gate_g = nn.Sequential(
                nn.Linear(proj_feat_dim, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim),
                nn.Sigmoid()
            )
            # Cross-modal interaction MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(proj_feat_dim * 2, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_feat_dim, proj_feat_dim)
            )

        elif self.fusion_type == 'add':
            pass

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}. Choose from ['dual_gate', 'concat', 'add'].")

    def forward(self, vision_features: torch.Tensor, graph_features: torch.Tensor) -> torch.Tensor:
        # --- Axis 2: Upstream Modality Dropout (Training Only) ---
        if self.training and self.modality_dropout > 0.0:
            rand_val = torch.rand(1).item()
            # Drop vision to force the GNN to carry the episode
            if rand_val < self.modality_dropout:
                vision_features = torch.zeros_like(vision_features)
            # Drop graph with smaller probability (0.05) to balance co-learning
            elif rand_val < self.modality_dropout + 0.05:
                graph_features = torch.zeros_like(graph_features)

        # --- Axis 1: Fusion Operator Execution ---
        if self.fusion_type == 'concat':
            fused = torch.cat([vision_features, graph_features], dim=-1)
            return self.fusion_mlp(fused)

        elif self.fusion_type in ['gate', 'dual_gate']:
            g_v = self.gate_v(vision_features)
            g_g = self.gate_g(graph_features)

            v_weighted = g_v * vision_features
            g_weighted = g_g * graph_features

            joint = torch.cat([v_weighted, g_weighted], dim=-1)
            fused_interaction = self.fusion_mlp(joint)

            # Residual skip connections preserve unimodal features
            return fused_interaction + v_weighted + g_weighted

        elif self.fusion_type == 'add':
            return vision_features + graph_features
