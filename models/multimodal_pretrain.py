# models/multimodal_pretrain.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.cnn_encoder import VisionEncoder
from models.encoders.gnn_encoder import GraphEncoder
from models.layers.attention_fusion import build_attention_fusion


class NormalizedLinear(nn.Module):
  """Enforces cosine classification on the unit hypersphere."""

  def __init__(self, in_features, out_features, scale=16.0):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.scale = scale
    self.weight = nn.Parameter(torch.empty(out_features, in_features))
    nn.init.orthogonal_(self.weight)

  def forward(self, x):
    w_norm = F.normalize(self.weight, p=2, dim=-1)
    cos_sim = F.linear(x, w_norm)
    return self.scale * cos_sim


class AsymmetricResidualProjection(nn.Module):
  """Vision Anchor + Linear Graph Projection."""

  def __init__(self, dim=640, dropout=0.1):
    super().__init__()
    self.proj_g = nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
    )
    self.ln = nn.LayerNorm(dim)

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    base_g = z_g if z_g.dim() == 2 else z_g.mean(dim=1)
    return F.normalize(self.ln(base_v + self.proj_g(base_g)), p=2, dim=-1)


class GatedResidualFusion(nn.Module):
  """Vision Anchor + Channel-Wise Sigmoid Gated Graph Context."""

  def __init__(self, dim=640, dropout=0.1):
    super().__init__()
    self.proj_g = nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
    )
    self.gate = nn.Sequential(
        nn.Linear(dim * 2, dim // 2),
        nn.ReLU(),
        nn.Linear(dim // 2, dim),
        nn.Sigmoid(),
    )
    self.ln = nn.LayerNorm(dim)

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    base_g = z_g if z_g.dim() == 2 else z_g.mean(dim=1)
    g = self.gate(torch.cat([base_v, base_g], dim=-1))
    delta = self.proj_g(base_g)
    return F.normalize(self.ln(base_v + g * delta), p=2, dim=-1)


class ConcatLinearProjection(nn.Module):
  """Concatenation + 2-layer MLP."""

  def __init__(self, dim=640, dropout=0.1):
    super().__init__()
    self.proj = nn.Sequential(
        nn.Linear(dim * 2, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
    )
    self.ln = nn.LayerNorm(dim)

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    base_g = z_g if z_g.dim() == 2 else z_g.mean(dim=1)
    return F.normalize(
        self.ln(self.proj(torch.cat([base_v, base_g], dim=-1))), p=2, dim=-1
    )


class MultimodalPretrainModel(nn.Module):
  """Joint Multimodal Network with Auxiliary Heads and Modality Dropout."""

  def __init__(self, cfg, num_classes=64):
    super().__init__()
    self.cfg = cfg
    self.num_classes = num_classes
    self.fusion_type = getattr(cfg.model, "fusion_type", "asymmetric").lower()
    self.p_drop_v = float(getattr(cfg.model, "modality_dropout_v", 0.15))
    self.p_drop_g = float(getattr(cfg.model, "modality_dropout_g", 0.15))
    scale = float(getattr(cfg.model, "scale", 16.0))
    dropout = getattr(cfg.model, "dropout", 0.1)

    # 1. Encoders (Both Trainable)
    self.vision_encoder = VisionEncoder(drop_rate=dropout)

    g_hidden_dim = getattr(cfg.model, "graph_hidden_dim", 128)
    g_proj_dim = getattr(cfg.model, "graph_proj_dim", 640)
    self.graph_encoder = GraphEncoder(
        node_feat_dim=cfg.model.node_feat_dim,
        edge_feat_dim=cfg.model.edge_feat_dim,
        hidden_dim=g_hidden_dim,
        proj_feat_dim=g_proj_dim,
        gnn_type=cfg.model.gnn_type,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        norm_type=getattr(cfg.model, "norm_type", "graph"),
        pooling_method=getattr(cfg.model, "pooling_method", "global_attention"),
        train_eps=getattr(cfg.model, "train_eps", True),
        use_input_mlp=getattr(cfg.model, "use_input_mlp", True),
        use_jk=getattr(cfg.model, "use_jk", True),
    )

    # 2. Comprehensive Fusion Dispatcher (Includes Cross-Attention)
    if self.fusion_type in ["asymmetric", "asym"]:
      self.fusion = AsymmetricResidualProjection(dim=640, dropout=dropout)
    elif self.fusion_type in ["gated", "dual_gate"]:
      self.fusion = GatedResidualFusion(dim=640, dropout=dropout)
    elif self.fusion_type in ["concat", "linear"]:
      self.fusion = ConcatLinearProjection(dim=640, dropout=dropout)
    elif self.fusion_type in ["bottleneck", "mbtb"]:
      self.fusion = build_attention_fusion(
          "bottleneck", dim=640, dropout=dropout
      )
    elif self.fusion_type in [
        "asymmetric_attn",
        "asym_attn",
        "cross_attention",
        "cross_attn",
        "dense",
        "low_rank",
        "rank",
    ]:
      # Fully wires Cross-Attention into joint pre-training
      self.fusion = build_attention_fusion(
          self.fusion_type, dim=640, dropout=dropout
      )
    elif self.fusion_type == "add":
      self.fusion = None
    else:
      raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    # 3. Auxiliary & Fused Normalized Heads
    self.classifier_fused = NormalizedLinear(640, num_classes, scale=scale)
    self.classifier_v = NormalizedLinear(640, num_classes, scale=scale)
    self.classifier_g = NormalizedLinear(640, num_classes, scale=scale)

  def extract_fused_features(self, img_batch, graph_batch):
    z_v = self.vision_encoder(img_batch)
    z_g = self.graph_encoder(graph_batch)
    if self.fusion is None:
      return F.normalize(z_v + z_g, p=2, dim=-1)
    return self.fusion(z_v, z_g)

  def forward(self, img_batch, graph_batch):
    z_v = self.vision_encoder(img_batch)
    z_g = self.graph_encoder(graph_batch)

    # Unbiased representations for auxiliary heads
    z_v_norm = F.normalize(z_v, p=2, dim=-1)
    z_g_norm = F.normalize(z_g, p=2, dim=-1)

    # Modality Dropout (Forces both encoders to remain active)
    if self.training:
      r = torch.rand(1).item()
      if r < self.p_drop_v:
        z_v_in, z_g_in = torch.zeros_like(z_v), z_g
      elif r < (self.p_drop_v + self.p_drop_g):
        z_v_in, z_g_in = z_v, torch.zeros_like(z_g)
      else:
        z_v_in, z_g_in = z_v, z_g
    else:
      z_v_in, z_g_in = z_v, z_g

    # Compute Fused Representation
    if self.fusion is None:
      z_unified = F.normalize(z_v_in + z_g_in, p=2, dim=-1)
    else:
      z_unified = self.fusion(z_v_in, z_g_in)

    logits_fused = self.classifier_fused(z_unified)
    logits_v = self.classifier_v(z_v_norm)
    logits_g = self.classifier_g(z_g_norm)

    return logits_fused, logits_v, logits_g, z_v_norm, z_g_norm
