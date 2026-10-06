# models/multimodal_network.py
from models.encoders.cnn_encoder import VisionEncoder
from models.encoders.gnn_encoder import GraphEncoder
from models.heads import FewShotClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing the attention fusion suite; provide self-contained fallback if needed
try:
  from models.layers.attention_fusion import build_attention_fusion
except ImportError:

  def _ensure_token_seq(x):
    if x.dim() == 2:
      return x.unsqueeze(1)
    elif x.dim() == 4:
      B, D, H, W = x.shape
      return x.view(B, D, H * W).transpose(1, 2)
    return x

  class MultimodalBottleneckAttention(nn.Module):

    def __init__(self, dim=640, num_bottlenecks=4, num_heads=8, dropout=0.1):
      super().__init__()
      self.dim = dim
      self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim))
      nn.init.trunc_normal_(self.bottlenecks, std=0.02)
      self.v_to_b = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.g_to_b = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.b_to_v = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.b_to_g = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.ln_b = nn.LayerNorm(dim)
      self.ln_v = nn.LayerNorm(dim)
      self.ln_g = nn.LayerNorm(dim)
      self.ln_out = nn.LayerNorm(dim)
      self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_v, z_g):
      B = z_v.size(0)
      t_v = _ensure_token_seq(z_v)
      t_g = _ensure_token_seq(z_g)
      b = self.bottlenecks.to(z_v.device).expand(B, -1, -1)
      b_from_v, _ = self.v_to_b(query=b, key=t_v, value=t_v)
      b_from_g, _ = self.g_to_b(query=b, key=t_g, value=t_g)
      b_up = self.ln_b(b + b_from_v + b_from_g)
      t_v_up, _ = self.b_to_v(query=t_v, key=b_up, value=b_up)
      t_g_up, _ = self.b_to_g(query=t_g, key=b_up, value=b_up)
      v_p = self.ln_v(t_v + t_v_up).mean(dim=1)
      g_p = self.ln_g(t_g + t_g_up).mean(dim=1)
      base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
      z_out = self.ln_out(base_v + self.alpha * (v_p + g_p))
      return F.normalize(z_out, p=2, dim=-1)

  class AsymmetricResidualCrossAttention(nn.Module):

    def __init__(self, dim=640, num_heads=8, dropout=0.1):
      super().__init__()
      self.cross_attn = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.gate_mlp = nn.Sequential(
          nn.Linear(dim * 2, dim // 2),
          nn.ReLU(),
          nn.Linear(dim // 2, dim),
          nn.Sigmoid(),
      )
      self.ln_delta = nn.LayerNorm(dim)
      self.ln_out = nn.LayerNorm(dim)
      self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, z_v, z_g):
      base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
      q_v = base_v.unsqueeze(1)
      t_g = _ensure_token_seq(z_g)
      attn_out, _ = self.cross_attn(query=q_v, key=t_g, value=t_g)
      delta_g = self.ln_delta(attn_out.squeeze(1))
      gate = self.gate_mlp(torch.cat([base_v, delta_g], dim=-1))
      z_out = self.ln_out(base_v + self.alpha * (gate * delta_g))
      return F.normalize(z_out, p=2, dim=-1)

  class LowRankAttention(nn.Module):

    def __init__(self, dim=640, rank=32, num_heads=4, dropout=0.1):
      super().__init__()
      self.down_v = nn.Linear(dim, rank, bias=False)
      self.down_g = nn.Linear(dim, rank, bias=False)
      self.subspace_attn = nn.MultiheadAttention(
          embed_dim=rank,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.up_proj = nn.Sequential(
          nn.Linear(rank, dim),
          nn.GELU(),
          nn.LayerNorm(dim),
      )
      self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
      self.ln_out = nn.LayerNorm(dim)

    def forward(self, z_v, z_g):
      base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
      t_v = _ensure_token_seq(z_v)
      t_g = _ensure_token_seq(z_g)
      r_v = self.down_v(t_v)
      r_g = self.down_g(t_g)
      r_fused, _ = self.subspace_attn(query=r_v, key=r_g, value=r_g)
      delta_z = self.up_proj(r_fused.mean(dim=1))
      g = self.gate(torch.cat([base_v, delta_z], dim=-1))
      z_out = self.ln_out(base_v + g * delta_z)
      return F.normalize(z_out, p=2, dim=-1)

  class DenseCrossAttention(nn.Module):

    def __init__(self, dim=640, num_heads=8, dropout=0.1):
      super().__init__()
      self.v_to_g = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.g_to_v = nn.MultiheadAttention(
          embed_dim=dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True,
      )
      self.ln_v = nn.LayerNorm(dim)
      self.ln_g = nn.LayerNorm(dim)
      self.proj = nn.Sequential(
          nn.Linear(dim * 2, dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(dim, dim),
      )
      self.ln_out = nn.LayerNorm(dim)

    def forward(self, z_v, z_g):
      t_v = _ensure_token_seq(z_v)
      t_g = _ensure_token_seq(z_g)
      a_v, _ = self.v_to_g(query=t_v, key=t_g, value=t_g)
      a_g, _ = self.g_to_v(query=t_g, key=t_v, value=t_v)
      o_v = self.ln_v(t_v + a_v).mean(dim=1)
      o_g = self.ln_g(t_g + a_g).mean(dim=1)
      fused = self.proj(torch.cat([o_v, o_g], dim=-1))
      base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
      return F.normalize(self.ln_out(base_v + fused), p=2, dim=-1)

  def build_attention_fusion(fusion_type: str, dim: int = 640, **kwargs):
    ft = fusion_type.lower()
    if ft in ['bottleneck', 'mbtb']:
      return MultimodalBottleneckAttention(dim=dim, **kwargs)
    elif ft in ['asymmetric', 'asym']:
      return AsymmetricResidualCrossAttention(dim=dim, **kwargs)
    elif ft in ['low_rank', 'rank']:
      return LowRankAttention(dim=dim, **kwargs)
    elif ft in ['cross_attention', 'cross_attn', 'dense']:
      return DenseCrossAttention(dim=dim, **kwargs)
    else:
      raise ValueError(f'Unknown attention fusion type: {fusion_type}')


# Standard Non-Attention Fusion Layer
class ResidualGateFusion(nn.Module):

  def __init__(self, dim=640):
    super().__init__()
    self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
    self.ln = nn.LayerNorm(dim)

  def forward(self, v_feat, g_feat):
    g = self.gate(torch.cat([v_feat, g_feat], dim=-1))
    fused = v_feat + g * g_feat
    return F.normalize(self.ln(fused), p=2, dim=-1)


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
          pooling_method=getattr(cfg.model, 'pooling_method', 'global_attention'),
          train_eps=getattr(cfg.model, 'train_eps', True),
          use_input_mlp=getattr(cfg.model, 'use_input_mlp', True),
          use_jk=getattr(cfg.model, 'use_jk', True),
      )

    # --- 3. Fusion Block ---
    if self.modality == 'multimodal':
      self.fusion_type = getattr(cfg.model, 'fusion_type', 'dual_gate')

      attention_types = [
          'bottleneck',
          'mbtb',
          'asymmetric',
          'asym',
          'low_rank',
          'rank',
          'cross_attention',
          'cross_attn',
          'dense',
      ]
      if self.fusion_type in attention_types:
        self.fusion = build_attention_fusion(
            fusion_type=self.fusion_type,
            dim=640,
            dropout=getattr(cfg.model, 'dropout', 0.1),
        )
      elif self.fusion_type == 'dual_gate':
        self.fusion = ResidualGateFusion(dim=640)
      elif self.fusion_type == 'concat':
        self.fusion = nn.Linear(640 * 2, 640)
      elif self.fusion_type == 'add':
        self.fusion = None
      else:
        raise ValueError(f'Unknown fusion_type configured: {self.fusion_type}')

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

      if self.fusion_type in [
          'bottleneck',
          'mbtb',
          'asymmetric',
          'asym',
          'low_rank',
          'rank',
          'cross_attention',
          'cross_attn',
          'dense',
      ]:
        features = self.fusion(v_feat, g_feat)
      elif self.fusion_type == 'dual_gate':
        features = self.fusion(v_feat, g_feat)
      elif self.fusion_type == 'concat':
        features = F.normalize(
            self.fusion(torch.cat([v_feat, g_feat], dim=-1)), p=2, dim=-1
        )
      elif self.fusion_type == 'add':
        features = F.normalize(v_feat + g_feat, p=2, dim=-1)
    else:
      raise ValueError(f'Unknown modality configured: {self.modality}')

    k_total = n_way * k_shot
    support_features = features[:k_total]
    query_features = features[k_total:]
    return self.classifier(support_features, query_features, n_way, k_shot)
