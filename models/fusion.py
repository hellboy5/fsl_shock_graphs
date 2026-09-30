# models/fusion.py
import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
  """Decoupled Multimodal Fusion Module.

  Supported Types:
    - 'add': Element-wise sum (zero parameters).
    - 'concat': Concatenation followed by 2-layer MLP.
    - 'dual_gate': Independent gating with interaction MLP and shortcuts.
    - 'res_gate' / 'residual_gate': Residual connection preserving visual
    anchor,
      injecting graph features as a gated alignment vector.
  """

  def __init__(
      self,
      proj_feat_dim: int = 640,
      fusion_type: str = 'dual_gate',
      dropout: float = 0.1,
      modality_dropout: float = 0.0,
  ):
    super().__init__()
    self.fusion_type = fusion_type.lower()
    self.proj_feat_dim = proj_feat_dim
    self.modality_dropout = modality_dropout

    if self.fusion_type == 'add':
      pass

    elif self.fusion_type == 'concat':
      self.fusion_mlp = nn.Sequential(
          nn.Linear(proj_feat_dim * 2, proj_feat_dim),
          nn.BatchNorm1d(proj_feat_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(proj_feat_dim, proj_feat_dim),
      )

    elif self.fusion_type in ['gate', 'dual_gate']:
      self.gate_v = nn.Sequential(
          nn.Linear(proj_feat_dim, proj_feat_dim),
          nn.BatchNorm1d(proj_feat_dim),
          nn.Sigmoid(),
      )
      self.gate_g = nn.Sequential(
          nn.Linear(proj_feat_dim, proj_feat_dim),
          nn.BatchNorm1d(proj_feat_dim),
          nn.Sigmoid(),
      )
      self.fusion_mlp = nn.Sequential(
          nn.Linear(proj_feat_dim * 2, proj_feat_dim),
          nn.BatchNorm1d(proj_feat_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(proj_feat_dim, proj_feat_dim),
      )

    elif self.fusion_type in ['res_gate', 'residual_gate']:
      # Aligns graph vector to visual space without bias distortion
      self.align = nn.Linear(proj_feat_dim, proj_feat_dim, bias=False)
      # Cross-modal gate deciding channel-wise shape injection
      self.gate = nn.Sequential(
          nn.Linear(proj_feat_dim * 2, proj_feat_dim),
          nn.Sigmoid(),
      )
      # Zero-initialize gate bias so training begins as pure vision anchor
      nn.init.constant_(self.gate[0].bias, -3.0)

    else:
      raise ValueError(
          f'Unknown fusion type: {self.fusion_type}. Choose from'
          " ['add', 'concat', 'dual_gate', 'res_gate']."
      )

  def forward(
      self, vision_features: torch.Tensor, graph_features: torch.Tensor
  ) -> torch.Tensor:
    if self.training and self.modality_dropout > 0.0:
      rand_val = torch.rand(1).item()
      if rand_val < self.modality_dropout:
        vision_features = torch.zeros_like(vision_features)
      elif rand_val < self.modality_dropout + 0.05:
        graph_features = torch.zeros_like(graph_features)

    if self.fusion_type == 'add':
      return vision_features + graph_features

    elif self.fusion_type == 'concat':
      fused = torch.cat([vision_features, graph_features], dim=-1)
      return self.fusion_mlp(fused)

    elif self.fusion_type in ['gate', 'dual_gate']:
      g_v = self.gate_v(vision_features)
      g_g = self.gate_g(graph_features)
      v_weighted = g_v * vision_features
      g_weighted = g_g * graph_features
      joint = torch.cat([v_weighted, g_weighted], dim=-1)
      fused_interaction = self.fusion_mlp(joint)
      return fused_interaction + v_weighted + g_weighted

    elif self.fusion_type in ['res_gate', 'residual_gate']:
      joint = torch.cat([vision_features, graph_features], dim=-1)
      g = self.gate(joint)
      aligned_g = torch.tanh(self.align(graph_features))
      return vision_features + g * aligned_g
