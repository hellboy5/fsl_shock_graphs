# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, GraphNorm, global_mean_pool
from torch_geometric.utils import degree, scatter, softmax, to_dense_batch


def compute_pairwise_seed_residuals(
    h1_seeds,
    h2_seeds,
    batch1_seeds,
    batch2_seeds,
    N_query,
    N_support,
    scale,
):
  """Computes isolated pairwise cross-attention residuals strictly within each (q, s) pair."""
  H_q, M_q = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  H_q_norm = F.normalize(H_q, p=2, dim=-1)
  H_s_norm = F.normalize(H_s, p=2, dim=-1)

  # Pairwise cosine similarity strictly between Query q and Support s: [N_q, N_s, K_q, K_s]
  cos_sim = torch.einsum("qid, sjd -> qsij", H_q_norm, H_s_norm)

  scores = cos_sim * 5.0

  # 1. Query -> Support attention
  scores_q2s = scores.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9)
  a12 = torch.nan_to_num(F.softmax(scores_q2s, dim=-1), nan=0.0)

  q_counts = M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)
  pair_sim_q = (
      ((cos_sim * a12).sum(dim=-1) * M_q.unsqueeze(1)).sum(dim=-1) / q_counts
  )
  w_q2s = F.softmax(pair_sim_q * 5.0, dim=-1)

  a12_weighted = w_q2s.unsqueeze(-1).unsqueeze(-1) * a12
  retrieved2 = torch.einsum("qsij, sjd -> qid", a12_weighted, H_s)
  mu1_dense = H_q - retrieved2
  mu1_seeds = mu1_dense[M_q]

  # 2. Support -> Query attention
  scores_s2q = scores.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9)
  a21 = torch.nan_to_num(F.softmax(scores_s2q, dim=-2), nan=0.0)

  s_counts = M_s.sum(dim=-1, keepdim=True).t().clamp(min=1.0)
  pair_sim_s = (
      ((cos_sim * a21).sum(dim=-2) * M_s.unsqueeze(0)).sum(dim=-1) / s_counts
  )
  w_s2q = F.softmax(pair_sim_s * 5.0, dim=0)

  a21_weighted = w_s2q.unsqueeze(-1).unsqueeze(-1) * a21
  retrieved1 = torch.einsum("qsij, qid -> sjd", a21_weighted, H_q)
  mu2_dense = H_s - retrieved1
  mu2_seeds = mu2_dense[M_s]

  return mu1_seeds, mu2_seeds


def compute_cross_transformer_reconstruction_distance(
    h1_final,
    h2_final,
    batch1,
    batch2,
    N_query,
    N_support,
    projector=None,
    scale=None,
    x1_app=None,
    x2_app=None,
    app_recon_alpha=None,
    use_center_norm=False,
    return_gnn_dist=False,
):
  """Computes unpooled Bidirectional CrossTransformers Set-to-Set Keypoint Distance."""
  h1_seeds = h1_final
  h2_seeds = h2_final
  batch1_seeds = batch1
  batch2_seeds = batch2

  if projector is not None and h1_seeds.size(-1) == projector[0].in_features:
    all_seeds = torch.cat([h1_seeds, h2_seeds], dim=0)
    proj_seeds = projector(all_seeds)
    h1_seeds = proj_seeds[: h1_seeds.size(0)]
    h2_seeds = proj_seeds[h1_seeds.size(0) :]

  H_q, M_q = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  # Global GNN Graph Anchor
  h_q_global = F.normalize(
      (H_q * M_q.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
  )
  h_s_global = F.normalize((H_s * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1)
  gnn_global_cos = torch.matmul(h_q_global, h_s_global.t())
  gnn_global_dist = 2.0 - 2.0 * gnn_global_cos
  global_cos = gnn_global_cos

  H_q = F.normalize(H_q, p=2, dim=-1)
  H_s = F.normalize(H_s, p=2, dim=-1)

  use_late_app = (
      app_recon_alpha is not None
      and x1_app is not None
      and x2_app is not None
  )
  if use_late_app:
    X_q_raw, _ = to_dense_batch(
        x1_app, batch1_seeds, batch_size=N_query
    )
    X_s_raw, _ = to_dense_batch(
        x2_app, batch2_seeds, batch_size=N_support
    )

    with torch.no_grad():
      X_q_det = X_q_raw.detach()
      X_s_det = X_s_raw.detach()
      q_active_all = X_q_det.abs().sum(dim=(-2, -1)) > 0
      s_active = X_s_det.abs().sum(dim=(-2, -1)) > 0

      q_mean_det = F.normalize(
          (X_q_det * M_q.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
      )
      s_mean_det = F.normalize(
          (X_s_det * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
      )

      fg_wq = torch.einsum("qid, qd -> qi", X_q_det, q_mean_det).clamp(min=1e-3)
      fg_wq = fg_wq.masked_fill(~M_q, 0.0)
      fg_wq = fg_wq / fg_wq.sum(dim=-1, keepdim=True).clamp(min=1e-6)
      unif_wq = M_q.float() / M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)
      W_q = torch.where(
          q_active_all.unsqueeze(-1), 0.5 * unif_wq + 0.5 * fg_wq, unif_wq
      ).unsqueeze(1)

      fg_ws = torch.einsum("sjd, sd -> sj", X_s_det, s_mean_det).clamp(min=1e-3)
      fg_ws = fg_ws.masked_fill(~M_s, 0.0)
      fg_ws = fg_ws / fg_ws.sum(dim=-1, keepdim=True).clamp(min=1e-6)
      unif_ws = M_s.float() / M_s.sum(dim=-1, keepdim=True).clamp(min=1.0)
      W_s = torch.where(
          s_active.unsqueeze(-1), 0.5 * unif_ws + 0.5 * fg_ws, unif_ws
      ).unsqueeze(0)
      pair_2d = q_active_all.unsqueeze(1) & s_active.unsqueeze(0)
      pair_active = pair_2d.unsqueeze(-1).unsqueeze(-1)

    q_mean = F.normalize(
        (X_q_raw * M_q.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
    )
    s_mean = F.normalize(
        (X_s_raw * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
    )
    app_global_cos = torch.matmul(q_mean, s_mean.t())

    # Step 4: Optional DeepEMD Per-Node Channel Mean Centering
    if use_center_norm:
      X_q_c = X_q_raw - X_q_raw.mean(dim=-1, keepdim=True)
      X_s_c = X_s_raw - X_s_raw.mean(dim=-1, keepdim=True)
      X_q = F.normalize(X_q_c, p=2, dim=-1)
      X_s_app = F.normalize(X_s_c, p=2, dim=-1)
    else:
      X_q = F.normalize(X_q_raw, p=2, dim=-1)
      X_s_app = F.normalize(X_s_raw, p=2, dim=-1)

    w_recon = torch.sigmoid(app_recon_alpha + 1.1)
    blended_global_cos = (1.0 - w_recon) * gnn_global_cos + w_recon * app_global_cos
    global_cos = torch.where(pair_2d, blended_global_cos, gnn_global_cos)
  else:
    w_recon = None
    W_q = None
    W_s = None
    X_q = None
    X_s_app = None
    pair_active = None

  global_dist = 2.0 - 2.0 * global_cos

  cos_sim_gnn = torch.einsum("qid, sjd -> qsij", H_q, H_s)
  K_q_counts = M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)
  K_s_counts = M_s.sum(dim=-1, keepdim=True).t().clamp(min=1.0)

  # 1. Standalone GNN Branch Reconstruction Distance (for Dual-Loss Co-Training)
  scores_gnn = cos_sim_gnn * 10.0
  A_q2s_g = torch.nan_to_num(
      F.softmax(scores_gnn.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9), dim=-1),
      nan=0.0,
  )
  err_q_g = (2.0 - 2.0 * (A_q2s_g * cos_sim_gnn).sum(dim=-1)).masked_fill(
      ~M_q.unsqueeze(1), 0.0
  )
  dist_q2s_g = (
      (err_q_g * W_q).sum(dim=-1)
      if W_q is not None
      else err_q_g.sum(dim=-1) / K_q_counts
  )

  A_s2q_g = torch.nan_to_num(
      F.softmax(scores_gnn.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9), dim=-2),
      nan=0.0,
  )
  err_s_g = (2.0 - 2.0 * (A_s2q_g * cos_sim_gnn).sum(dim=-2)).masked_fill(
      ~M_s.unsqueeze(0), 0.0
  )
  dist_s2q_g = (
      (err_s_g * W_s).sum(dim=-1)
      if W_s is not None
      else err_s_g.sum(dim=-1) / K_s_counts
  )
  local_gnn_dist = 0.5 * (dist_q2s_g + dist_s2q_g)

  # 2. Blended Multimodal Set Reconstruction Distance
  cos_sim = cos_sim_gnn
  if use_late_app:
    app_cos_sim = torch.einsum("qid, sjd -> qsij", X_q, X_s_app)
    blended_cos = (1.0 - w_recon) * cos_sim_gnn + w_recon * app_cos_sim
    cos_sim = torch.where(pair_active, blended_cos, cos_sim_gnn)

  scores = cos_sim * 10.0

  A_q2s = torch.nan_to_num(
      F.softmax(scores.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9), dim=-1),
      nan=0.0,
  )
  err_q = (2.0 - 2.0 * (A_q2s * cos_sim).sum(dim=-1)).masked_fill(
      ~M_q.unsqueeze(1), 0.0
  )
  dist_q2s = (
      (err_q * W_q).sum(dim=-1)
      if W_q is not None
      else err_q.sum(dim=-1) / K_q_counts
  )

  A_s2q = torch.nan_to_num(
      F.softmax(scores.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9), dim=-2),
      nan=0.0,
  )
  err_s = (2.0 - 2.0 * (A_s2q * cos_sim).sum(dim=-2)).masked_fill(
      ~M_s.unsqueeze(0), 0.0
  )
  dist_s2q = (
      (err_s * W_s).sum(dim=-1)
      if W_s is not None
      else err_s.sum(dim=-1) / K_s_counts
  )
  local_recon_dist = 0.5 * (dist_q2s + dist_s2q)

  mm_dist = 0.85 * local_recon_dist + 0.15 * global_dist
  gnn_dist = 0.85 * local_gnn_dist + 0.15 * gnn_global_dist
  if return_gnn_dist:
    return mm_dist, gnn_dist
  return mm_dist


class GraphEncoder(nn.Module):
  """GINE & Graph Matching Network (GMN) Backbone with Modular & Direct Full-Vector Visual Fusion."""

  def __init__(
      self,
      node_feat_dim=11,
      edge_feat_dim=14,
      hidden_dim=128,
      proj_feat_dim=640,
      num_layers=4,
      gnn_dropout=0.1,
      norm_type="graph",
      pooling_method="mean",
      train_eps=True,
      use_input_mlp=True,
      use_jk=True,
      use_reconstruction_dist=True,
      use_full_app_prop=False,
      use_center_norm=False,
      app_feat_dim=0,
      app_mlp_dim=None,
      app_graph_dropout=0.35,
      use_app_node_input=True,
      use_app_edges=False,
      use_app_recon_fusion=False,
      **kwargs,
  ):
    super(GraphEncoder, self).__init__()
    self.use_jk = use_jk
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.proj_feat_dim = proj_feat_dim
    self.gnn_dropout = float(gnn_dropout)
    self.pooling_method = str(pooling_method).lower()
    self.scale = 1.0 / (hidden_dim**0.5)
    self.use_reconstruction_dist = use_reconstruction_dist
    self.use_center_norm = bool(use_center_norm)
    self.last_gnn_dist = None

    self.app_feat_dim = int(app_feat_dim)
    self.app_mlp_dim = (
        int(app_mlp_dim) if app_mlp_dim is not None else int(hidden_dim)
    )
    self.app_graph_dropout = float(app_graph_dropout)

    self.use_full_app_prop = bool(use_full_app_prop) and self.app_feat_dim > 0
    self.use_app_node_input = bool(use_app_node_input) and self.app_feat_dim > 0
    self.use_app_edges = bool(use_app_edges) and self.app_feat_dim > 0
    self.use_app_recon_fusion = (
        bool(use_app_recon_fusion) and self.app_feat_dim > 0
    )

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

    # 3. Pooling / Readout Setup
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

    # 5. GMN Cross-Graph Matching Heads
    self.proj_match = nn.Linear(hidden_dim, hidden_dim)
    self.node_update = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    nn.init.zeros_(self.node_update[-1].weight)
    nn.init.zeros_(self.node_update[-1].bias)

    # 6. Visual Appearance Fusion Heads (All prefixed with `app_` for LR routing)
    needs_app_stem = (
        self.use_app_node_input
        or self.use_app_edges
    )
    if needs_app_stem:
      self.app_input_mlp = nn.Sequential(
          nn.LayerNorm(self.app_feat_dim),
          nn.Linear(self.app_feat_dim, self.app_mlp_dim),
          nn.GELU(),
          nn.Dropout(self.gnn_dropout),
          nn.Linear(self.app_mlp_dim, hidden_dim),
          nn.LayerNorm(hidden_dim),
          nn.GELU(),
      )

    if self.use_app_node_input:
      self.app_node_gate = nn.Sequential(
          nn.Linear(hidden_dim, hidden_dim, bias=False),
          nn.LayerNorm(hidden_dim, elementwise_affine=False),
      )
      self.app_node_alpha = nn.Parameter(torch.zeros(1, hidden_dim))

    if self.use_app_edges:
      self.app_edge_mlp = nn.Sequential(
          nn.Linear(hidden_dim * 2, hidden_dim),
          nn.LayerNorm(hidden_dim),
          nn.GELU(),
          nn.Linear(hidden_dim, hidden_dim, bias=False),
          nn.LayerNorm(hidden_dim, elementwise_affine=False),
      )
      self.app_edge_alpha = nn.Parameter(torch.zeros(1, hidden_dim))

    if self.use_app_recon_fusion:
      self.app_recon_alpha = nn.Parameter(torch.tensor(0.0))

    # Step 3: Nonlinear relational edge gates & active beta=0.23 init for 1,120-D propagation
    if self.use_full_app_prop:
      self.app_prop_gates = nn.ModuleList(
          [
              nn.Sequential(
                  nn.Linear(hidden_dim * 3, hidden_dim // 2),
                  nn.LayerNorm(hidden_dim // 2),
                  nn.GELU(),
                  nn.Linear(hidden_dim // 2, 1),
              )
              for _ in range(num_layers)
          ]
      )
      # Initialize at 0.5 -> beta = 0.5 * tanh(0.5) = 0.23 (Active from Step 0!)
      self.app_prop_betas = nn.ParameterList(
          [nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)]
      )

  def _apply_pooling(self, x_final, batch):
    if self.pooling_method == "global_attention":
      return self.pool_op(x_final, batch)
    return global_mean_pool(x_final, batch)

  def _encode_appearance(self, data, batch=None):
    if (
        self.app_feat_dim <= 0
        or not hasattr(data, "x_app")
        or data.x_app is None
    ):
      return None, None

    x_app = data.x_app

    if self.training and self.app_graph_dropout > 0.0 and batch is not None:
      num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
      if num_graphs > 0:
        keep_prob = 1.0 - self.app_graph_dropout
        graph_keep = (
            torch.rand(num_graphs, 1, device=x_app.device) < keep_prob
        ).type_as(x_app)
        mask_g = graph_keep[batch]
        x_app = x_app * mask_g

    if hasattr(self, "app_input_mlp"):
      a = self.app_input_mlp(x_app)
    else:
      a = None

    return a, x_app

  def _compute_edge_appearance(self, a, edge_index):
    src, dst = edge_index[0], edge_index[1]
    a_mean = 0.5 * (a[src] + a[dst])
    a_diff = torch.abs(a[src] - a[dst])
    return torch.tanh(self.app_edge_alpha) * self.app_edge_mlp(
        torch.cat([a_mean, a_diff], dim=-1)
    )

  def _propagate_full_appearance(self, x_app, h, e, edge_index, layer_idx):
    """Step 3: Propagates uncompressed 1,120-D x_app 1 hop along shock graph edges."""
    if x_app is None or edge_index.numel() == 0:
      return x_app
    src, dst = edge_index[0], edge_index[1]
    num_nodes = x_app.size(0)

    with torch.no_grad():
      x_norm = F.normalize(x_app.detach(), p=2, dim=-1)
      vis_edge_cos = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)

    h_norm = F.normalize(h, p=2, dim=-1)
    struct_edge_cos = (h_norm[src] * h_norm[dst]).sum(dim=-1, keepdim=True)
    rel_feat = torch.cat(
        [(h[src] - h[dst]).abs(), h[src] * h[dst], e], dim=-1
    )

    edge_logits = (
        self.app_prop_gates[layer_idx](rel_feat)
        + 3.0 * struct_edge_cos
        + 3.0 * vis_edge_cos
    ).squeeze(-1)
    w_uv = softmax(edge_logits, index=dst, num_nodes=num_nodes).unsqueeze(-1)

    nbr_x = scatter(
        w_uv * x_app[src], dst, dim=0, dim_size=num_nodes, reduce="sum"
    )
    has_in = (
        (degree(dst, num_nodes=num_nodes) > 0).unsqueeze(-1).type_as(x_app)
    )
    beta = 0.5 * torch.tanh(self.app_prop_betas[layer_idx])
    return x_app + beta * has_in * (nbr_x - x_app)

  def forward(self, data):
    x, edge_index, edge_attr, batch = (
        data.x,
        data.edge_index,
        data.edge_attr,
        data.batch,
    )

    a, x_app = self._encode_appearance(data, batch=batch)

    x = self.node_encoder(x)
    edge_attr = self.edge_encoder(edge_attr)

    if self.use_app_node_input and a is not None:
      x = x + torch.tanh(self.app_node_alpha) * self.app_node_gate(a)
    if self.use_app_edges and a is not None:
      edge_attr = edge_attr + self._compute_edge_appearance(a, edge_index)

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
      
      if i < self.num_layers - 1 and self.gnn_dropout > 0.0:
        x = F.dropout(x, p=self.gnn_dropout, training=self.training)
      
      x = x + x_in
      if self.use_full_app_prop and x_app is not None:
        x_app = self._propagate_full_appearance(
            x_app, x, edge_attr, edge_index, i
        )

      if self.use_jk:
        layer_outputs.append(x)

    x_final = torch.cat(layer_outputs, dim=-1) if self.use_jk else x
    pooled = self._apply_pooling(x_final, batch)
    out = self.projector(pooled)

    if self.use_app_recon_fusion and x_app is not None:
      w = torch.sigmoid(self.app_recon_alpha + 1.1)
      out_norm = F.normalize(out, p=2, dim=-1)
      app_pooled = F.normalize(global_mean_pool(x_app, batch), p=2, dim=-1)
      out = torch.cat(
          [torch.sqrt(1.0 - w) * out_norm, torch.sqrt(w) * app_pooled], dim=-1
      )

    return out

  def forward_gmn(self, g1, g2):
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

    a1, x1_app = self._encode_appearance(
        g1, batch=batch1
    )
    a2, x2_app = self._encode_appearance(
        g2, batch=batch2
    )

    if self.use_app_node_input and a1 is not None and a2 is not None:
      gate_node = torch.tanh(self.app_node_alpha)
      h1 = h1 + gate_node * self.app_node_gate(a1)
      h2 = h2 + gate_node * self.app_node_gate(a2)

    if self.use_app_edges and a1 is not None and a2 is not None:
      e1 = e1 + self._compute_edge_appearance(a1, edge_index1)
      e2 = e2 + self._compute_edge_appearance(a2, edge_index2)

    layer_outputs1 = []
    layer_outputs2 = []

    for i, layer in enumerate(self.layers):
      h1_in = h1
      h2_in = h2

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

      h1 = F.relu(h1)
      h2 = F.relu(h2)
      if i < self.num_layers - 1 and self.gnn_dropout > 0.0:
        h1 = F.dropout(h1, p=self.gnn_dropout, training=self.training)
        h2 = F.dropout(h2, p=self.gnn_dropout, training=self.training)
      h1 = h1 + h1_in
      h2 = h2 + h2_in

      mu1_seeds, mu2_seeds = compute_pairwise_seed_residuals(
          h1,
          h2,
          batch1,
          batch2,
          N_query,
          N_support,
          self.scale,
      )

      h1 = h1 + self.node_update(
          torch.cat([h1, self.proj_match(mu1_seeds)], dim=-1)
      )
      h2 = h2 + self.node_update(
          torch.cat([h2, self.proj_match(mu2_seeds)], dim=-1)
      )

      # Step 3: Coupled Multi-Hop Propagation of Full 1,120-D Vectors (After GMN seed update!)
      if self.use_full_app_prop and x1_app is not None and x2_app is not None:
        x1_app = self._propagate_full_appearance(
            x1_app, h1, e1, edge_index1, i
        )
        x2_app = self._propagate_full_appearance(
            x2_app, h2, e2, edge_index2, i
        )

      if self.use_jk:
        layer_outputs1.append(h1)
        layer_outputs2.append(h2)

    h1_final = torch.cat(layer_outputs1, dim=-1) if self.use_jk else h1
    h2_final = torch.cat(layer_outputs2, dim=-1) if self.use_jk else h2

    if self.use_reconstruction_dist:
      # Step 4: Centered Set Reconstruction + Standalone GNN Distance for Co-Training
      dist, gnn_dist = compute_cross_transformer_reconstruction_distance(
          h1_final=h1_final,
          h2_final=h2_final,
          batch1=batch1,
          batch2=batch2,
          N_query=N_query,
          N_support=N_support,
          projector=self.projector,
          scale=self.scale,
          x1_app=x1_app if self.use_app_recon_fusion else None,
          x2_app=x2_app if self.use_app_recon_fusion else None,
          app_recon_alpha=(
              self.app_recon_alpha if self.use_app_recon_fusion else None
          ),
          use_center_norm=self.use_center_norm,
          return_gnn_dist=True,
      )
      self.last_gnn_dist = gnn_dist
      return dist, None, None
    else:
      pooled1 = self._apply_pooling(h1_final, batch1)
      pooled2 = self._apply_pooling(h2_final, batch2)
      z1_norm = F.normalize(self.projector(pooled1), p=2, dim=-1)
      z2_norm = F.normalize(self.projector(pooled2), p=2, dim=-1)
      self.last_gnn_dist = torch.cdist(z1_norm, z2_norm, p=2) ** 2
      if self.use_app_recon_fusion and x1_app is not None and x2_app is not None:
        w = torch.sigmoid(self.app_recon_alpha + 1.1)
        app1 = F.normalize(global_mean_pool(x1_app, batch1), p=2, dim=-1)
        app2 = F.normalize(global_mean_pool(x2_app, batch2), p=2, dim=-1)
        z1_norm = torch.cat(
            [torch.sqrt(1.0 - w) * z1_norm, torch.sqrt(w) * app1], dim=-1
        )
        z2_norm = torch.cat(
            [torch.sqrt(1.0 - w) * z2_norm, torch.sqrt(w) * app2], dim=-1
        )
      dist = torch.cdist(z1_norm, z2_norm, p=2) ** 2
      return dist, z1_norm, z2_norm
