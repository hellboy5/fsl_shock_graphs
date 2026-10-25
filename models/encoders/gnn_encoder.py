# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, GraphNorm, global_mean_pool
from torch_geometric.utils import degree, to_dense_batch


def compute_pairwise_seed_residuals(
    h1_seeds,
    h2_seeds,
    batch1_seeds,
    batch2_seeds,
    N_query,
    N_support,
    scale,
):
  """Computes isolated pairwise cross-attention residuals strictly within each (q, s) pair.

  Uses L2-normalized keypoint similarity for stability and contracts candidate
  weights before feature retrieval to avoid 4D [N_q, N_s, K, D] memory overhead.
  """
  H_q, M_q = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  # L2-normalize for bounded, stable cosine attention scores across all layers
  H_q_norm = F.normalize(H_q, p=2, dim=-1)
  H_s_norm = F.normalize(H_s, p=2, dim=-1)

  # Pairwise cosine similarity strictly between Query q and Support s: [N_q, N_s, K_q, K_s]
  cos_sim = torch.einsum("qid, sjd -> qsij", H_q_norm, H_s_norm)
  scores = cos_sim * 5.0

  # 1. Query -> Support attention (softmax strictly over Support graph s's keypoints)
  scores_q2s = scores.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9)
  a12 = torch.nan_to_num(F.softmax(scores_q2s, dim=-1), nan=0.0)

  # Weight support candidates by mean keypoint cosine match similarity: [N_q, N_s]
  q_counts = M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)
  pair_sim_q = (
      ((cos_sim * a12).sum(dim=-1) * M_q.unsqueeze(1)).sum(dim=-1) / q_counts
  )
  w_q2s = F.softmax(pair_sim_q * 5.0, dim=-1)  # [N_q, N_s]

  # Contract (w_q2s * a12) with H_s directly -> [N_q, K_q, D]
  a12_weighted = w_q2s.unsqueeze(-1).unsqueeze(-1) * a12
  retrieved2 = torch.einsum("qsij, sjd -> qid", a12_weighted, H_s)
  mu1_dense = H_q - retrieved2
  mu1_seeds = mu1_dense[M_q]

  # 2. Support -> Query attention (softmax strictly over Query graph q's keypoints)
  scores_s2q = scores.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9)
  a21 = torch.nan_to_num(F.softmax(scores_s2q, dim=-2), nan=0.0)

  s_counts = M_s.sum(dim=-1, keepdim=True).t().clamp(min=1.0)  # Shape: [1, N_s]
  pair_sim_s = (
      ((cos_sim * a21).sum(dim=-2) * M_s.unsqueeze(0)).sum(dim=-1) / s_counts
  )
  w_s2q = F.softmax(pair_sim_s * 5.0, dim=0)  # [N_q, N_s]

  a21_weighted = w_s2q.unsqueeze(-1).unsqueeze(-1) * a21
  retrieved1 = torch.einsum("qsij, qid -> sjd", a21_weighted, H_q)
  mu2_dense = H_s - retrieved1
  mu2_seeds = mu2_dense[M_s]

  return mu1_seeds, mu2_seeds


def compute_cross_transformer_reconstruction_distance(
    h1_final,
    h2_final,
    mask1,
    mask2,
    batch1,
    batch2,
    N_query,
    N_support,
    projector=None,
    scale=None,
):
  """Computes unpooled Bidirectional CrossTransformers Set-to-Set Keypoint Distance.

  1. Projects keypoints through Stage 1's `self.projector` (Linear + BatchNorm1d)
     to center the positive ReLU activations and prevent constant-vector mode collapse.
  2. Evaluates expected squared Euclidean distance on the unit sphere (2 - 2 * cos_sim)
     without Jensen norm shrinkage or 4D feature reconstruction tensors.
  """
  h1_seeds = h1_final[mask1]
  h2_seeds = h2_final[mask2]
  batch1_seeds = batch1[mask1]
  batch2_seeds = batch2[mask2]

  # Map seed nodes through Stage 1's Linear + BatchNorm1d projector to center channels & prevent collapse
  if projector is not None and h1_seeds.size(-1) == projector[0].in_features:
    all_seeds = torch.cat([h1_seeds, h2_seeds], dim=0)
    proj_seeds = projector(all_seeds)
    h1_seeds = proj_seeds[: h1_seeds.size(0)]
    h2_seeds = proj_seeds[h1_seeds.size(0) :]

  H_q, M_q = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  # Unit-normalize centered 640D keypoint descriptors: ||h_i|| = 1, ||h_j|| = 1
  H_q = F.normalize(H_q, p=2, dim=-1)
  H_s = F.normalize(H_s, p=2, dim=-1)

  # Pairwise cosine similarity in [-1, 1] and squared Euclidean distance in [0, 4]
  cos_sim = torch.einsum("qid, sjd -> qsij", H_q, H_s)
  pair_dist_sq = 2.0 - 2.0 * cos_sim  # ||h_i^q - h_j^s||^2 on the unit sphere
  scores = cos_sim * 10.0

  # Direction 1: Query keypoints matched against Support s keypoints
  scores_q2s = scores.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9)
  A_q2s = torch.nan_to_num(F.softmax(scores_q2s, dim=-1), nan=0.0)

  err_q = (A_q2s * pair_dist_sq).sum(dim=-1)  # [N_query, N_support, K_max_q]
  err_q = err_q.masked_fill(~M_q.unsqueeze(1), 0.0)
  K_q_counts = M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)
  dist_q2s = err_q.sum(dim=-1) / K_q_counts  # [N_query, N_support]

  # Direction 2: Support keypoints matched against Query q keypoints (Symmetric term)
  scores_s2q = scores.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9)
  A_s2q = torch.nan_to_num(F.softmax(scores_s2q, dim=-2), nan=0.0)

  err_s = (A_s2q * pair_dist_sq).sum(dim=-2)  # [N_query, N_support, K_max_s]
  err_s = err_s.masked_fill(~M_s.unsqueeze(0), 0.0)
  K_s_counts = M_s.sum(dim=-1, keepdim=True).t().clamp(min=1.0)
  dist_s2q = err_s.sum(dim=-1) / K_s_counts  # [N_query, N_support]

  return 0.5 * (dist_q2s + dist_s2q)


class GraphEncoder(nn.Module):
  """GINE & Graph Matching Network (GMN) Backbone for Shock Graphs.

  Supports Stage 1 unimodal representation learning (`forward`) and
  Stage 2 pairwise cross-graph matching with set reconstruction (`forward_gmn`).
  """

  def __init__(
      self,
      node_feat_dim=11,
      edge_feat_dim=14,
      hidden_dim=128,
      proj_feat_dim=640,
      gnn_type="GINE",
      num_layers=4,
      dropout=0.1,
      norm_type="graph",
      pooling_method="mean",
      train_eps=True,
      use_input_mlp=True,
      use_jk=True,
      use_reconstruction_dist=True,
      **kwargs,
  ):
    super(GraphEncoder, self).__init__()
    self.gnn_type = gnn_type
    self.use_jk = use_jk
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.pooling_method = str(pooling_method).lower()
    self.scale = 1.0 / (hidden_dim**0.5)
    self.use_reconstruction_dist = use_reconstruction_dist

    effective_dim = (hidden_dim * num_layers) if use_jk else hidden_dim

    # 1. Input Encoders (Node & Edge Projections)
    if use_input_mlp:
      self.node_encoder = nn.Sequential(
          nn.Linear(node_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
      self.edge_encoder = nn.Sequential(
          nn.Linear(edge_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
    else:
      self.node_encoder = nn.Sequential(
          nn.Linear(node_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
      self.edge_encoder = nn.Sequential(
          nn.Linear(edge_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )

    # 2. GINE Message Passing Convolution Layers
    self.layers = nn.ModuleList()
    self.norms = nn.ModuleList()

    for _ in range(num_layers):
      nn_mlp = nn.Sequential(
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
      )
      self.layers.append(
          GINEConv(nn_mlp, edge_dim=hidden_dim, train_eps=train_eps)
      )

      if norm_type == "graph":
        self.norms.append(GraphNorm(hidden_dim))
      elif norm_type == "layer":
        self.norms.append(nn.LayerNorm(hidden_dim))
      elif norm_type == "batch":
        self.norms.append(nn.BatchNorm1d(hidden_dim))

    # 3. Pooling / Readout Setup ('mean' or 'global_attention')
    if self.pooling_method == "global_attention":
      gate_nn = nn.Sequential(
          nn.Linear(effective_dim, effective_dim // 2),
          nn.ReLU(),
          nn.Linear(effective_dim // 2, 1),
      )
      self.pool_op = GlobalAttention(gate_nn=gate_nn)

    # 4. Output Projection Head to Metric Space (640D)
    self.projector = nn.Sequential(
        nn.Linear(effective_dim, proj_feat_dim),
        nn.BatchNorm1d(proj_feat_dim),
    )

    # 5. GMN Cross-Graph Matching Heads (Zero-init final layer so Step 0 == Pre-trained GINE)
    self.proj_match = nn.Linear(hidden_dim, hidden_dim)
    self.node_update = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    nn.init.zeros_(self.node_update[-1].weight)
    nn.init.zeros_(self.node_update[-1].bias)

  def _apply_pooling(self, x_final, batch):
    """Applies 'mean' or 'global_attention' pooling to produce a single graph vector."""
    if self.pooling_method == "global_attention":
      return self.pool_op(x_final, batch)
    return global_mean_pool(x_final, batch)

  def forward(self, data):
    """Standard forward pass on a single graph or batch (Stage 1 Pre-Training / ProtoNet)."""
    x, edge_index, edge_attr, batch = (
        data.x,
        data.edge_index,
        data.edge_attr,
        data.batch,
    )

    x = self.node_encoder(x)
    edge_attr = self.edge_encoder(edge_attr)

    layer_outputs = []
    for i, layer in enumerate(self.layers):
      x_in = x
      x = layer(x, edge_index, edge_attr=edge_attr)
      x = (
          self.norms[i](x, batch)
          if isinstance(self.norms[i], GraphNorm)
          else self.norms[i](x)
      )
      x = F.relu(x)
      x = x + x_in

      if self.use_jk:
        layer_outputs.append(x)

    x_final = torch.cat(layer_outputs, dim=-1) if self.use_jk else x
    pooled = self._apply_pooling(x_final, batch)
    return self.projector(pooled)

  def forward_gmn(self, g1, g2):
    """Stage 2 GMN forward pass: Joint cross-graph propagation & set reconstruction distance."""
    x1, edge_index1, edge_attr1 = g1.x, g1.edge_index, g1.edge_attr
    x2, edge_index2, edge_attr2 = g2.x, g2.edge_index, g2.edge_attr

    batch1 = (
        g1.batch
        if hasattr(g1, "batch") and g1.batch is not None
        else torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
    )
    batch2 = (
        g2.batch
        if hasattr(g2, "batch") and g2.batch is not None
        else torch.zeros(x2.size(0), dtype=torch.long, device=x2.device)
    )

    N_query = (
        g1.num_graphs
        if hasattr(g1, "num_graphs")
        else int(batch1.max().item() + 1)
    )
    N_support = (
        g2.num_graphs
        if hasattr(g2, "num_graphs")
        else int(batch2.max().item() + 1)
    )

    h1 = self.node_encoder(x1)
    h2 = self.node_encoder(x2)
    e1 = self.edge_encoder(edge_attr1)
    e2 = self.edge_encoder(edge_attr2)

    deg1 = degree(edge_index1[0], num_nodes=h1.size(0))
    deg2 = degree(edge_index2[0], num_nodes=h2.size(0))
    mask1 = (deg1 == 1) | (deg1 >= 3)
    mask2 = (deg2 == 1) | (deg2 >= 3)

    if not mask1.any():
      mask1 = torch.ones_like(mask1)
    if not mask2.any():
      mask2 = torch.ones_like(mask2)

    batch1_seeds = batch1[mask1]
    batch2_seeds = batch2[mask2]

    layer_outputs1 = []
    layer_outputs2 = []

    for i, layer in enumerate(self.layers):
      h1_in = h1
      h2_in = h2

      # A. Intra-graph GINE message passing
      h1 = layer(h1, edge_index1, edge_attr=e1)
      h2 = layer(h2, edge_index2, edge_attr=e2)

      h1 = (
          self.norms[i](h1, batch1)
          if isinstance(self.norms[i], GraphNorm)
          else self.norms[i](h1)
      )
      h2 = (
          self.norms[i](h2, batch2)
          if isinstance(self.norms[i], GraphNorm)
          else self.norms[i](h2)
      )

      h1 = F.relu(h1) + h1_in
      h2 = F.relu(h2) + h2_in

      # B. Isolated Pairwise Inter-graph Cross-Attention Matching (GMN)
      h1_seeds = h1[mask1]
      h2_seeds = h2[mask2]

      mu1_seeds, mu2_seeds = compute_pairwise_seed_residuals(
          h1_seeds,
          h2_seeds,
          batch1_seeds,
          batch2_seeds,
          N_query,
          N_support,
          self.scale,
      )

      # C. Additive residual update strictly on seed keypoints
      delta1 = torch.zeros_like(h1)
      delta2 = torch.zeros_like(h2)
      delta1[mask1] = self.node_update(
          torch.cat([h1_seeds, self.proj_match(mu1_seeds)], dim=-1)
      )
      delta2[mask2] = self.node_update(
          torch.cat([h2_seeds, self.proj_match(mu2_seeds)], dim=-1)
      )

      h1 = h1 + delta1
      h2 = h2 + delta2

      if self.use_jk:
        layer_outputs1.append(h1)
        layer_outputs2.append(h2)

    h1_final = torch.cat(layer_outputs1, dim=-1) if self.use_jk else h1
    h2_final = torch.cat(layer_outputs2, dim=-1) if self.use_jk else h2

    if self.use_reconstruction_dist:
      dist = compute_cross_transformer_reconstruction_distance(
          h1_final=h1_final,
          h2_final=h2_final,
          mask1=mask1,
          mask2=mask2,
          batch1=batch1,
          batch2=batch2,
          N_query=N_query,
          N_support=N_support,
          projector=self.projector,
          scale=self.scale,
      )
      return dist, None, None
    else:
      pooled1 = self._apply_pooling(h1_final, batch1)
      pooled2 = self._apply_pooling(h2_final, batch2)
      z1_norm = F.normalize(self.projector(pooled1), p=2, dim=-1)
      z2_norm = F.normalize(self.projector(pooled2), p=2, dim=-1)
      dist = torch.cdist(z1_norm, z2_norm, p=2) ** 2
      return dist, z1_norm, z2_norm
