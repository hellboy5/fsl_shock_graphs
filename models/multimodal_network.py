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
      g_hidden_dim = getattr(cfg.model, 'graph_hidden_dim', 128)
      g_proj_dim = getattr(cfg.model, 'graph_proj_dim', None) or g_hidden_dim

      self.graph_encoder = GraphEncoder(
          node_feat_dim=cfg.model.node_feat_dim,
          edge_feat_dim=cfg.model.edge_feat_dim,
          hidden_dim=g_hidden_dim,
          proj_feat_dim=g_proj_dim,
          gnn_type=cfg.model.gnn_type,
          num_layers=cfg.model.num_layers,
          dropout=cfg.model.dropout,
          norm_type=getattr(cfg.model, 'norm_type', 'graph'),
          pooling_method=getattr(cfg.model, 'pooling_method', 'mean'),
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

    # --- 5. Pretrained Checkpoint Loading & Freezing Hook ---
    v_ckpt = getattr(cfg.model, 'vision_checkpoint', None)
    if self.modality == 'multimodal' and v_ckpt:
      self._load_submodule(self.vision_encoder, v_ckpt, 'vision_encoder')
      if getattr(cfg.model, 'freeze_vision', False):
        for p in self.vision_encoder.parameters():
          p.requires_grad = False
        print('--> [MultimodalNetwork] Vision encoder frozen.')

    g_ckpt = getattr(cfg.model, 'graph_checkpoint', None)
    if self.modality == 'multimodal' and g_ckpt:
      self._load_submodule(self.graph_encoder, g_ckpt, 'graph_encoder')
      if getattr(cfg.model, 'freeze_graph', False):
        for p in self.graph_encoder.parameters():
          p.requires_grad = False
        print('--> [MultimodalNetwork] Graph encoder frozen.')

  def _load_submodule(self, module, path, target_name):
    print(
        f'--> [MultimodalNetwork] Loading {target_name} weights from: {path}'
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    target_dict = module.state_dict()
    matched = {}

    for k, v in state.items():
      # Strip prefix if loading from another multimodal checkpoint
      clean_k = k
      for prefix in [f'{target_name}.', 'model.', 'module.']:
        if clean_k.startswith(prefix):
          clean_k = clean_k[len(prefix) :]
      if clean_k in target_dict and v.shape == target_dict[clean_k].shape:
        matched[clean_k] = v

    target_dict.update(matched)
    module.load_state_dict(target_dict)
    print(
        f'--> [MultimodalNetwork] Loaded {len(matched)} / {len(target_dict)}'
        f' layers for {target_name}.'
    )

  def forward(self, vision_batch, graph_batch, n_way, k_shot):
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

    k_total = n_way * k_shot
    support_features = features[:k_total]
    query_features = features[k_total:]
    return self.classifier(support_features, query_features, n_way, k_shot)
