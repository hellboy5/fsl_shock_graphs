import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, proj_feat_dim=128, fusion_type='concat'):
        """
        Args:
            proj_feat_dim: Dimension of input modalities
            fusion_type: 'concat', 'add', or 'gate'
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if self.fusion_type == 'concat':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(proj_feat_dim * 2, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(proj_feat_dim, proj_feat_dim)
            )
        elif self.fusion_type == 'gate':
            self.gate_layer = nn.Linear(proj_feat_dim * 2, proj_feat_dim)

    def forward(self, vision_features, graph_features):
        if self.fusion_type == 'concat':
            fused = torch.cat([vision_features, graph_features], dim=-1)
            return self.fusion_mlp(fused)
            
        elif self.fusion_type == 'add':
            return vision_features + graph_features
            
        elif self.fusion_type == 'gate':
            concat_feat = torch.cat([vision_features, graph_features], dim=-1)
            gate = torch.sigmoid(self.gate_layer(concat_feat))
            return gate * vision_features + (1 - gate) * graph_features
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
