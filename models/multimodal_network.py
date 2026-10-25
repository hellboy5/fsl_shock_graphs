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

    # --- 1. Vision Pathway (Used for vision baseline or frozen RFS hypercolumns) ---
    if self.modality in ['vision', 'multimodal']:
      self.vision_encoder = VisionEncoder(
          drop_rate=getattr(cfg.model, 'dropout', 0.1)
      )
      v_ckpt = getattr(cfg.model, 'vision_checkpoint', None)
      if v_ckpt:
        self._load_submodule(self.vision_encoder, v_ckpt, 'vision_encoder')
      if getattr(cfg.model, 'freeze_vision', False):
        for p in self.vision_encoder.parameters():
          p.requires_grad = False
        print('--> [MultimodalNetwork] Vision encoder frozen.')

    # --- 2. Graph / GMN Pathway ---
    if self.modality in ['graph', 'multimodal']:
      g_hidden_dim = getattr(cfg.model, 'graph_hidden_dim', 128)
      g_proj_dim = getattr(cfg.model, 'graph_proj_dim', None) or g_hidden_dim

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
      )
      g_ckpt = getattr(cfg.model, 'graph_checkpoint', None)
      if g_ckpt:
        self._load_submodule(self.graph_encoder, g_ckpt, 'graph_encoder')
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

  def _load_submodule(self, module, path, target_name):
    print(
        f'--> [MultimodalNetwork] Loading {target_name} weights from: {path}'
    )
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    target_dict = module.state_dict()
    matched = {}

    for k, v in state.items():
      clean_k = k
      for prefix in [f'{target_name}.', 'model.', 'module.', 'encoder.']:
        if clean_k.startswith(prefix):
          clean_k = clean_k[len(prefix) :]
      if clean_k in target_dict and v.shape == target_dict[clean_k].shape:
        matched[clean_k] = v

    target_dict.update(matched)
    module.load_state_dict(target_dict, strict=False)
    print(
        f'--> [MultimodalNetwork] Loaded {len(matched)} / {len(target_dict)}'
        f' layers for {target_name}.'
    )

  def extract_fused_features(self, vision_batch, graph_batch):
    """Extracts 640D feature embeddings for unimodal ProtoNet evaluation."""
    if self.modality == 'vision':
      return self.vision_encoder(vision_batch)
    return self.graph_encoder(graph_batch)

  def forward(self, vision_batch, graph_batch, n_way, k_shot):
    # --- GMN Branch: Pairwise Cross-Graph Matching & Set Reconstruction ---
    if self.fsl_method == 'gmn':
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
