# models/multimodal_network.py
from models.encoders.cnn_encoder import VisionEncoder
from models.encoders.gnn_encoder import GraphEncoder
from models.fusion import MultimodalFusion
from models.heads import FewShotClassifier
import torch
import torch.nn as nn


class MultimodalFewShotNetwork(nn.Module):
  """Top-level network for Multimodal Few-Shot Learning.

  Routes data through unimodal encoders, fuses them if required,
  and passes the resulting features to the Prototypical Network head.
  """

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.modality = cfg.model.modality

    # --- 1. Vision Pathway (Native 640D) ---
    if self.modality in ['vision', 'multimodal']:
      self.vision_encoder = VisionEncoder(
          drop_rate=getattr(cfg.model, 'dropout', 0.1)
      )

    # --- 2. Graph Pathway ---
    if self.modality in ['graph', 'multimodal']:
      # Internal message-passing width (locked to champion 128D)
      g_hidden_dim = getattr(cfg.model, 'graph_hidden_dim', 128)

      # Output projection dimension: graph_proj_dim if set, otherwise matches g_hidden_dim
      g_proj_dim = (
          getattr(cfg.model, 'graph_proj_dim', None) or g_hidden_dim
      )

      self.graph_encoder = GraphEncoder(
          node_feat_dim=cfg.model.node_feat_dim,
          edge_feat_dim=cfg.model.edge_feat_dim,
          hidden_dim=g_hidden_dim,  # Convolutions run at graph_hidden_dim (128D)
          proj_feat_dim=g_proj_dim,  # Output projected to g_proj_dim (128D or 640D)
          gnn_type=cfg.model.gnn_type,
          num_layers=cfg.model.num_layers,
          dropout=cfg.model.dropout,
          norm_type=getattr(cfg.model, 'norm_type', 'graph'),
          use_dual_pool=getattr(cfg.model, 'use_dual_pool', False),
          train_eps=getattr(cfg.model, 'train_eps', False),
          use_input_mlp=getattr(cfg.model, 'use_input_mlp', False),
          use_jk=getattr(cfg.model, 'use_jk', False),
      )

    # --- 3. Fusion Block ---
    if self.modality == 'multimodal':
      self.fusion = MultimodalFusion(
          proj_feat_dim=640,
          fusion_type=cfg.model.fusion_type,
          modality_dropout=getattr(cfg.model, 'modality_dropout', 0.0),
      )

    # --- 4. Few-Shot Metric Head ---
    self.classifier = FewShotClassifier(
        method=cfg.model.fsl_method,
        distance=cfg.model.distance_metric,
        scale=getattr(cfg.model, 'scale', 10.0),
        learnable_scale=getattr(cfg.model, 'learnable_scale', False),
    )

  def forward(self, vision_batch, graph_batch, n_way, k_shot):
    """Forward pass for an entire FSL episode."""
    # --- A. Feature Extraction & Modality Routing ---
    if self.modality == 'vision':
      features = self.vision_encoder(vision_batch)

    elif self.modality == 'graph':
      features = self.graph_encoder(graph_batch)

    elif self.modality == 'multimodal':
      v_feat = self.vision_encoder(vision_batch)
      g_feat = self.graph_encoder(graph_batch)
      features = self.fusion(v_feat, g_feat)

    else:
      raise ValueError(f'Unknown modality configured: {self.modality}')

    # --- B. Episodic Splitting ---
    k_total = n_way * k_shot
    support_features = features[:k_total]
    query_features = features[k_total:]

    # --- C. Prototypical Classification ---
    logits = self.classifier(support_features, query_features, n_way, k_shot)
    return logits
