# models/multimodal_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.encoders.cnn_encoder import VisionEncoder
from models.encoders.gnn_encoder import GraphEncoder
from models.heads import FewShotClassifier


class MultimodalFewShotNetwork(nn.Module):
  """Top-level Few-Shot Network for Shock Graphs & Visual Hypercolumns.

  Supports:
    - Stage 2 Graph Matching Network (fsl_method='gmn')
    - Stage 1 / Unimodal ProtoNet evaluation (fsl_method='protonet')
  """

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.modality = cfg.model.modality
    self.fsl_method = cfg.model.fsl_method

    # --- 1. Vision Pathway (Used for vision baseline or frozen RFS/DeepEMD hypercolumns) ---
    self.freeze_vision = bool(getattr(cfg.model, 'freeze_vision', False))
    if self.modality in ['vision', 'multimodal']:
      self.vision_encoder = VisionEncoder(
          drop_rate=getattr(cfg.model, 'dropout', 0.1),
          hc_layers=getattr(cfg.model, 'hc_layers', [1, 2, 3]),
          hc_radius_mode=getattr(cfg.model, 'hc_radius_mode', 'disk_5pt'),
          hc_disk_scale=float(getattr(cfg.model, 'hc_disk_scale', 0.7)),
          hc_stage_l2norm=bool(getattr(cfg.model, 'hc_stage_l2norm', True)),
      )
      v_ckpt = getattr(cfg.model, 'vision_checkpoint', None)
      if v_ckpt:
        self.load_checkpoint(
            self.vision_encoder.cnn, v_ckpt, 'vision_encoder.cnn'
        )
      if self.freeze_vision:
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
          p.requires_grad = False
        print('--> [MultimodalNetwork] Vision encoder frozen.')

    # --- 2. Graph / GMN Pathway ---
    if self.modality in ['graph', 'multimodal']:
      g_hidden_dim = getattr(cfg.model, 'graph_hidden_dim', 128)
      g_proj_dim = getattr(cfg.model, 'graph_proj_dim', None) or g_hidden_dim
      app_feat_dim = (
          self.vision_encoder.hc_out_dim if self.modality == 'multimodal' else 0
      )

      self.graph_encoder = GraphEncoder(
          node_feat_dim=cfg.model.node_feat_dim,
          edge_feat_dim=cfg.model.edge_feat_dim,
          hidden_dim=g_hidden_dim,
          proj_feat_dim=g_proj_dim,
          gnn_type=getattr(cfg.model, 'gnn_type', 'GINE'),
          num_layers=cfg.model.num_layers,
          dropout=cfg.model.dropout,
          norm_type=getattr(cfg.model, 'norm_type', 'graph'),
          pooling_method=getattr(cfg.model, 'pooling_method', 'mean'),
          train_eps=getattr(cfg.model, 'train_eps', True),
          use_input_mlp=getattr(cfg.model, 'use_input_mlp', True),
          use_jk=getattr(cfg.model, 'use_jk', True),
          use_reconstruction_dist=getattr(
              cfg.model, 'use_reconstruction_dist', True
          ),
          app_feat_dim=app_feat_dim,
      )
      g_ckpt = getattr(cfg.model, 'graph_checkpoint', None)
      if g_ckpt:
        self.load_checkpoint(self.graph_encoder, g_ckpt, 'graph_encoder')
      if getattr(cfg.model, 'freeze_graph', False):
        for p in self.graph_encoder.parameters():
          p.requires_grad = False
        print('--> [MultimodalNetwork] Graph encoder frozen.')

    # --- 3. Few-Shot Metric Head ---
    self.scale = float(getattr(cfg.model, 'scale', 5.0))

    if self.fsl_method == 'gmn':
      if self.modality not in ['graph', 'multimodal']:
        raise ValueError(
            "fsl_method='gmn' requires graph inputs (modality='graph' or 'multimodal')."
        )
      classifier_method = 'protonet'
    else:
      classifier_method = self.fsl_method

    self.classifier = FewShotClassifier(
        method=classifier_method,
        distance=getattr(cfg.model, 'distance_metric', 'cosine'),
        scale=self.scale,
        learnable_scale=getattr(cfg.model, 'learnable_scale', False),
    )

  def train(self, mode: bool = True):
    """Keeps frozen vision encoder BatchNorm and DropBlock in eval mode during training."""
    super().train(mode)
    if self.modality in ['vision', 'multimodal'] and self.freeze_vision:
      self.vision_encoder.eval()
    return self

  def load_checkpoint(self, module, path, target_name):
    print(
        f'--> [MultimodalNetwork] Loading {target_name} weights from: {path}'
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
      for key in ('model_state_dict', 'params', 'model', 'state_dict', 'encoder'):
        if key in ckpt and isinstance(ckpt[key], dict):
          state = ckpt[key]
          break

    prefixes = (
        f'{target_name}.',
        'vision_encoder.cnn.',
        'vision_encoder.',
        'graph_encoder.',
        'feature_extractor.',
        'model.',
        'module.',
        'encoder.',
        'features.',
        'cnn.',
        'backbone.',
    )

    raw_clean = {}
    for k, v in state.items():
      if any(
          skip in k
          for skip in ('classifier', 'fc', 'linear', 'num_batches_tracked')
      ):
        continue
      clean_k = k
      changed = True
      while changed:
        changed = False
        for prefix in prefixes:
          if clean_k.startswith(prefix):
            clean_k = clean_k[len(prefix) :]
            changed = True
      raw_clean[clean_k] = v

    target_dict = {
        k: v
        for k, v in module.state_dict().items()
        if 'num_batches_tracked' not in k
    }

    matched = {}
    # Pass 1: Direct key-name match
    for clean_k, v in raw_clean.items():
      if clean_k in target_dict and v.shape == target_dict[clean_k].shape:
        matched[clean_k] = v
      elif (
          f'cnn.{clean_k}' in target_dict
          and v.shape == target_dict[f'cnn.{clean_k}'].shape
      ):
        matched[f'cnn.{clean_k}'] = v

    # Pass 2: Ordered tensor shape match fallback
    if len(matched) < len(target_dict):
      raw_vals = list(raw_clean.values())
      tgt_keys = list(target_dict.keys())
      if len(raw_vals) == len(tgt_keys) and all(
          raw_vals[i].shape == target_dict[tgt_keys[i]].shape
          for i in range(len(tgt_keys))
      ):
        matched = dict(zip(tgt_keys, raw_vals))

    module.load_state_dict(matched, strict=False)
    print(
        f'--> [MultimodalNetwork] Loaded {len(matched)} / {len(target_dict)}'
        f' layers for {target_name}.'
    )

  def attach_node_hypercolumns(self, vision_batch, graph_batch):
    """Samples (x, y, r) hypercolumns from vision_batch onto graph_batch.x_app."""
    with torch.set_grad_enabled(not self.freeze_vision):
      x_app = self.vision_encoder.extract_hypercolumns(
          vision_batch, graph_batch.pos_xyr, graph_batch.batch
      )
    graph_batch.x_app = x_app
    if hasattr(graph_batch, '_slice_dict') and 'x' in graph_batch._slice_dict:
      graph_batch._slice_dict['x_app'] = graph_batch._slice_dict['x'].clone()
      graph_batch._inc_dict['x_app'] = torch.zeros_like(
          graph_batch._inc_dict['x']
      )
    return graph_batch

  def extract_fused_features(self, vision_batch, graph_batch):
    """Extracts feature embeddings for unimodal/multimodal ProtoNet evaluation."""
    if self.modality == 'vision':
      return self.vision_encoder(vision_batch)
    if self.modality == 'multimodal':
      graph_batch = self.attach_node_hypercolumns(vision_batch, graph_batch)
    return self.graph_encoder(graph_batch)

  def forward(self, vision_batch, graph_batch, n_way, k_shot):
    # --- GMN Branch: Pairwise Cross-Graph Matching & Set Reconstruction ---
    if self.fsl_method == 'gmn':
      if self.modality == 'multimodal':
        graph_batch = self.attach_node_hypercolumns(vision_batch, graph_batch)

      num_support = n_way * k_shot
      graph_list = graph_batch.to_data_list()

      device = graph_batch.x.device
      batch_s = Batch.from_data_list(graph_list[:num_support]).to(device)
      batch_q = Batch.from_data_list(graph_list[num_support:]).to(device)

      dist_matrix, _, _ = self.graph_encoder.forward_gmn(batch_q, batch_s)

      if k_shot > 1:
        dist_matrix = dist_matrix.view(
            dist_matrix.size(0), n_way, k_shot
        ).mean(dim=-1)

      return -self.scale * dist_matrix

    # --- Standard ProtoNet / Matching Siamese Branch ---
    features = self.extract_fused_features(vision_batch, graph_batch)
    k_total = n_way * k_shot
    support_features = features[:k_total]
    query_features = features[k_total:]
    return self.classifier(support_features, query_features, n_way, k_shot)
