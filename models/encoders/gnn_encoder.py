# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, GraphNorm, MessagePassing, global_mean_pool
from torch_geometric.utils import degree, scatter, softmax, to_dense_batch


class GatedGINEConv(MessagePassing):
  """Anisotropic Edge-Gated GINE layer preserving 1-WL sum aggregation."""

  def __init__(self, nn_mlp, hidden_dim, train_eps=True):
    super().__init__(aggr="add")
    self.nn = nn_mlp
    if train_eps:
      self.eps = nn.Parameter(torch.zeros(1))
    else:
      self.register_buffer("eps", torch.zeros(1))
    self.edge_gate = nn.Linear(hidden_dim * 3, hidden_dim)
    nn.init.zeros_(self.edge_gate.weight)
    nn.init.constant_(self.edge_gate.bias, 1.0)

  def forward(self, x, edge_index, edge_attr):
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
    out = out + (1.0 + self.eps) * x
    return self.nn(out)

  def message(self, x_i, x_j, edge_attr):
    gate = torch.sigmoid(
        self.edge_gate(torch.cat([edge_attr, x_j, x_i], dim=-1))
    )
    return gate * F.relu(x_j + edge_attr)


def _build_rev_edge_index(edge_index, num_nodes):
  """Finds reverse edge indices (v -> u) for each directed edge (u -> v) without ordering assumptions."""
  if edge_index.numel() == 0 or num_nodes <= 0:
    return None, None
  src, dst = edge_index[0].long(), edge_index[1].long()
  key_uv = src * num_nodes + dst
  key_vu = dst * num_nodes + src
  perm = torch.argsort(key_uv)
  sorted_keys = key_uv[perm]
  pos = torch.searchsorted(sorted_keys, key_vu).clamp(
      max=sorted_keys.numel() - 1
  )
  has_rev = sorted_keys[pos] == key_vu
  rev_idx = perm[pos]
  return rev_idx, has_rev


def compute_cross_transformer_reconstruction_distance(
    h1_final,
    h2_final,
    batch1,
    batch2,
    N_query,
    N_support,
    scale=None,
    x1_app=None,
    x2_app=None,
    app_recon_alpha=None,
    use_center_norm=False,
    app_running_mean=None,
    return_gnn_dist=False,
    use_global_dist=True,
):
  """Computes unpooled Bidirectional CrossTransformers Set-to-Set Keypoint Distance."""
  h1_seeds = h1_final
  h2_seeds = h2_final
  batch1_seeds = batch1
  batch2_seeds = batch2

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

    # Per-Channel [1, 1, D] Image-Level Running Mean Centering
    if use_center_norm and app_running_mean is not None:
      X_q_c = (X_q_raw - app_running_mean) * M_q.unsqueeze(-1)
      X_s_c = (X_s_raw - app_running_mean) * M_s.unsqueeze(-1)
      X_q = F.normalize(X_q_c, p=2, dim=-1)
      X_s_app = F.normalize(X_s_c, p=2, dim=-1)
      q_mean = F.normalize(X_q_c.sum(dim=1), p=2, dim=-1)
      s_mean = F.normalize(X_s_c.sum(dim=1), p=2, dim=-1)
    else:
      X_q = F.normalize(X_q_raw, p=2, dim=-1)
      X_s_app = F.normalize(X_s_raw, p=2, dim=-1)
      q_mean = F.normalize((X_q_raw * M_q.unsqueeze(-1)).sum(dim=1), p=2, dim=-1)
      s_mean = F.normalize((X_s_raw * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1)

    app_global_cos = torch.matmul(q_mean, s_mean.t())

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

  if use_global_dist:
    mm_dist = 0.85 * local_recon_dist + 0.15 * global_dist
    gnn_dist = 0.85 * local_gnn_dist + 0.15 * gnn_global_dist
  else:
    mm_dist = local_recon_dist
    gnn_dist = local_gnn_dist

  if return_gnn_dist:
    return mm_dist, gnn_dist
  return mm_dist


def compute_gluestick_edge_reconstruction_distance(
    e1_final,
    e2_final,
    edge_index1,
    edge_index2,
    batch1,
    batch2,
    N_query,
    N_support,
    x1_app=None,
    x2_app=None,
    use_center_norm=False,
    app_running_mean=None,
    edge_match_vis_logit=None,
    edge_match_vis_mode="both",
    edge_match_diff_gamma=0.25,
):
  """GlueStick-style orientation-invariant bidirectional branch reconstruction."""
  device = e1_final.device
  if edge_index1.numel() == 0 or edge_index2.numel() == 0:
    zero = torch.zeros(N_query, N_support, device=device, dtype=e1_final.dtype)
    return zero, zero

  # converter.py emits consecutive (2k: u->v, 2k+1: v->u) pairs per physical branch
  e1_fwd, e1_rev = F.normalize(e1_final[0::2], p=2, dim=-1), F.normalize(e1_final[1::2], p=2, dim=-1)
  e2_fwd, e2_rev = F.normalize(e2_final[0::2], p=2, dim=-1), F.normalize(e2_final[1::2], p=2, dim=-1)
  src1, dst1 = edge_index1[0, 0::2], edge_index1[1, 0::2]
  src2, dst2 = edge_index2[0, 0::2], edge_index2[1, 0::2]
  eb1 = batch1[src1]
  eb2 = batch2[src2]

  E1_f, M_eq = to_dense_batch(e1_fwd, eb1, batch_size=N_query)
  E1_r, _ = to_dense_batch(e1_rev, eb1, batch_size=N_query)
  E2_f, M_es = to_dense_batch(e2_fwd, eb2, batch_size=N_support)
  E2_r, _ = to_dense_batch(e2_rev, eb2, batch_size=N_support)

  s_geo_aligned = 0.5 * (
      torch.einsum("qid, sjd -> qsij", E1_f, E2_f)
      + torch.einsum("qid, sjd -> qsij", E1_r, E2_r)
  )
  s_geo_flipped = 0.5 * (
      torch.einsum("qid, sjd -> qsij", E1_f, E2_r)
      + torch.einsum("qid, sjd -> qsij", E1_r, E2_f)
  )
  s_edge_geo = torch.maximum(s_geo_aligned, s_geo_flipped)

  use_vis = (
      edge_match_vis_logit is not None
      and x1_app is not None
      and x2_app is not None
      and edge_match_vis_mode != "none"
  )
  if use_vis:
    with torch.no_grad():
      q_act = scatter(x1_app.detach().abs().sum(dim=-1), batch1, dim=0, dim_size=N_query, reduce="sum") > 0
      s_act = scatter(x2_app.detach().abs().sum(dim=-1), batch2, dim=0, dim_size=N_support, reduce="sum") > 0
      pair_active = (q_act.unsqueeze(1) & s_act.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)

    if use_center_norm and app_running_mean is not None:
      mu = app_running_mean.squeeze(0)
      x1_n = F.normalize(x1_app - mu, p=2, dim=-1)
      x2_n = F.normalize(x2_app - mu, p=2, dim=-1)
    else:
      x1_n = F.normalize(x1_app, p=2, dim=-1)
      x2_n = F.normalize(x2_app, p=2, dim=-1)

    V1_u, _ = to_dense_batch(x1_n[src1], eb1, batch_size=N_query)
    V1_v, _ = to_dense_batch(x1_n[dst1], eb1, batch_size=N_query)
    V2_u, _ = to_dense_batch(x2_n[src2], eb2, batch_size=N_support)
    V2_v, _ = to_dense_batch(x2_n[dst2], eb2, batch_size=N_support)

    s_end_aligned = 0.5 * (
        torch.einsum("qid, sjd -> qsij", V1_u, V2_u)
        + torch.einsum("qid, sjd -> qsij", V1_v, V2_v)
    )
    s_end_flipped = 0.5 * (
        torch.einsum("qid, sjd -> qsij", V1_u, V2_v)
        + torch.einsum("qid, sjd -> qsij", V1_v, V2_u)
    )
    if edge_match_vis_mode == "both":
      gamma = float(edge_match_diff_gamma)
      s_vis_aligned = s_end_aligned + gamma * (s_end_aligned - s_end_flipped)
      s_vis_flipped = s_end_flipped + gamma * (s_end_flipped - s_end_aligned)
    else:
      s_vis_aligned = s_end_aligned
      s_vis_flipped = s_end_flipped

    w_ev = torch.sigmoid(edge_match_vis_logit)
    s_aligned = torch.where(
        pair_active, w_ev * s_vis_aligned + (1.0 - w_ev) * s_geo_aligned, s_geo_aligned
    )
    s_flipped = torch.where(
        pair_active, w_ev * s_vis_flipped + (1.0 - w_ev) * s_geo_flipped, s_geo_flipped
    )
    s_edge = torch.maximum(s_aligned, s_flipped)
  else:
    s_edge = s_edge_geo

  K_eq = M_eq.sum(dim=-1, keepdim=True).clamp(min=1.0)
  K_es = M_es.sum(dim=-1, keepdim=True).t().clamp(min=1.0)

  def _bidir_recon(sim_mat):
    scores = sim_mat * 10.0
    A_q2s = torch.nan_to_num(
        F.softmax(scores.masked_fill(~M_es.unsqueeze(0).unsqueeze(2), -1e9), dim=-1),
        nan=0.0,
    )
    err_q = (2.0 - 2.0 * (A_q2s * sim_mat).sum(dim=-1)).masked_fill(~M_eq.unsqueeze(1), 0.0)
    d_q2s = err_q.sum(dim=-1) / K_eq

    A_s2q = torch.nan_to_num(
        F.softmax(scores.masked_fill(~M_eq.unsqueeze(1).unsqueeze(3), -1e9), dim=-2),
        nan=0.0,
    )
    err_s = (2.0 - 2.0 * (A_s2q * sim_mat).sum(dim=-2)).masked_fill(~M_es.unsqueeze(0), 0.0)
    d_s2q = err_s.sum(dim=-1) / K_es
    return 0.5 * (d_q2s + d_s2q)

  d_edge_geo = _bidir_recon(s_edge_geo)
  d_edge = _bidir_recon(s_edge) if use_vis else d_edge_geo
  return d_edge, d_edge_geo


class GraphEncoder(nn.Module):
  """GINE Backbone with Dynamic Edge Propagation & Centered Set-to-Set Visual Fusion."""

  VALID_EDGE_MODES = (
      "none",
      "co_update",
      "gated_static",
      "gated_co_update",
      "nb_co_update",
      "nb_gated_co_update",
      "nb_gated_dual_jk",
  )

  def __init__(
      self,
      node_feat_dim=11,
      edge_feat_dim=14,
      hidden_dim=128,
      edge_prop_mode="none",
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
      center_vis_edge=True,
      use_global_dist=True,
      use_edge_matching=False,
      edge_match_vis_mode="both",
      edge_match_diff_gamma=0.25,
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
    self.edge_prop_mode = str(edge_prop_mode).lower()
    if self.edge_prop_mode not in self.VALID_EDGE_MODES:
      raise ValueError(f"Unknown edge_prop_mode: {self.edge_prop_mode}")

    self.use_edge_co_update = self.edge_prop_mode in (
        "co_update",
        "gated_co_update",
        "nb_co_update",
        "nb_gated_co_update",
        "nb_gated_dual_jk",
    )
    self.use_edge_gate = self.edge_prop_mode in (
        "gated_static",
        "gated_co_update",
        "nb_gated_co_update",
        "nb_gated_dual_jk",
    )
    self.use_nb_relay = self.edge_prop_mode in (
        "nb_co_update",
        "nb_gated_co_update",
        "nb_gated_dual_jk",
    )
    self.use_edge_readout = self.edge_prop_mode == "nb_gated_dual_jk"

    self.gnn_dropout = float(gnn_dropout)
    self.pooling_method = str(pooling_method).lower()
    self.scale = 1.0 / (hidden_dim**0.5)
    self.use_reconstruction_dist = use_reconstruction_dist
    self.use_center_norm = bool(use_center_norm)
    self.center_vis_edge = bool(center_vis_edge)
    self.use_global_dist = bool(use_global_dist)
    self.use_edge_matching = bool(use_edge_matching)
    self.edge_match_vis_mode = str(edge_match_vis_mode).lower()
    self.edge_match_diff_gamma = float(edge_match_diff_gamma)
    self.last_gnn_dist = None

    print(
        f"--> [GraphEncoder] edge_prop_mode={self.edge_prop_mode} | "
        f"center_vis_edge={self.center_vis_edge} | "
        f"use_global_dist={self.use_global_dist} | "
        f"use_edge_matching={self.use_edge_matching}"
    )

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
    self.out_dim = effective_dim * 2 if self.use_edge_readout else effective_dim

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

    # 2. GINE / GatedGINE Message Passing & Dynamic Edge Co-Update Layers
    self.layers = nn.ModuleList()
    self.norms = nn.ModuleList()

    for _ in range(num_layers):
      nn_mlp = nn.Sequential(
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
      )
      if self.use_edge_gate:
        self.layers.append(
            GatedGINEConv(nn_mlp, hidden_dim=hidden_dim, train_eps=train_eps)
        )
      else:
        self.layers.append(
            GINEConv(nn_mlp, edge_dim=hidden_dim, train_eps=train_eps)
        )

      if norm_type == "graph":
        self.norms.append(GraphNorm(hidden_dim))
      elif norm_type == "layer":
        self.norms.append(nn.LayerNorm(hidden_dim))
      elif norm_type == "batch":
        self.norms.append(nn.BatchNorm1d(hidden_dim))

    if self.use_edge_co_update:
      num_edge_layers = (
          num_layers if self.use_edge_readout else max(num_layers - 1, 1)
      )
      edge_in_dim = hidden_dim * 4 if self.use_nb_relay else hidden_dim * 3
      self.edge_updaters = nn.ModuleList()
      self.edge_norms = nn.ModuleList()
      for _ in range(num_edge_layers):
        self.edge_updaters.append(
            nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        if norm_type == "graph":
          self.edge_norms.append(GraphNorm(hidden_dim))
        elif norm_type == "layer":
          self.edge_norms.append(nn.LayerNorm(hidden_dim))
        elif norm_type == "batch":
          self.edge_norms.append(nn.BatchNorm1d(hidden_dim))

    # 3. Pooling / Readout Setup
    if self.pooling_method == "global_attention":
      gate_nn = nn.Sequential(
          nn.Linear(self.out_dim, self.out_dim // 2),
          nn.ReLU(),
          nn.Linear(self.out_dim // 2, 1),
      )
      self.pool_op = GlobalAttention(gate_nn=gate_nn)

    # 4. Zero-Parameter GNN Output Mean Centering
    self.register_buffer("gnn_running_mean", torch.zeros(1, self.out_dim))
    self.register_buffer("gnn_mean_init", torch.tensor(False))

    if self.use_edge_matching:
      self.register_buffer("edge_running_mean", torch.zeros(1, hidden_dim))
      self.register_buffer("edge_mean_init", torch.tensor(False))
      self.edge_match_beta_logit = nn.Parameter(torch.tensor(-1.7346))
      if self.use_app_recon_fusion:
        self.edge_match_vis_logit = nn.Parameter(torch.tensor(0.0))

    # 5. Visual Appearance Fusion Heads (All prefixed with `app_` for LR routing)
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

    if self.app_feat_dim > 0:
      self.register_buffer("app_running_mean", torch.zeros(1, 1, self.app_feat_dim))
      self.register_buffer("app_mean_init", torch.tensor(False))

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

  def _center_gnn_nodes(self, h_nodes, batch):
    if self.training and batch is not None and h_nodes.numel() > 0:
      with torch.no_grad():
        g_means = global_mean_pool(h_nodes.detach(), batch)
        b_mean = g_means.mean(dim=0, keepdim=True)
        if not bool(self.gnn_mean_init.item()):
          self.gnn_running_mean.copy_(b_mean)
          self.gnn_mean_init.fill_(True)
        else:
          self.gnn_running_mean.lerp_(b_mean, 0.05)
    return h_nodes - self.gnn_running_mean

  def _center_gnn_edges(self, e_edges, edge_batch):
    if self.training and edge_batch is not None and e_edges.numel() > 0:
      with torch.no_grad():
        g_means = global_mean_pool(e_edges.detach(), edge_batch)
        b_mean = g_means.mean(dim=0, keepdim=True)
        if not bool(self.edge_mean_init.item()):
          self.edge_running_mean.copy_(b_mean)
          self.edge_mean_init.fill_(True)
        else:
          self.edge_running_mean.lerp_(b_mean, 0.05)
    return e_edges - self.edge_running_mean

  def _update_edge_state(
      self, e, x, edge_index, batch, layer_idx, rev_idx=None, has_rev=None
  ):
    if edge_index.numel() == 0:
      return e
    src, dst = edge_index[0], edge_index[1]
    h_sum = x[src] + x[dst]
    h_diff = x[dst] - x[src]
    if self.use_nb_relay and rev_idx is not None and has_rev is not None:
      num_nodes = x.size(0)
      in_sum = scatter(e, dst, dim=0, dim_size=num_nodes, reduce="sum")
      in_deg = degree(dst, num_nodes=num_nodes, dtype=e.dtype).unsqueeze(-1)
      rev_mask = has_rev.unsqueeze(-1).type_as(e)
      nb_sum = in_sum[src] - e[rev_idx] * rev_mask
      nb_deg = (in_deg[src] - rev_mask).clamp(min=1.0)
      nb_msg = nb_sum / nb_deg
      edge_in = torch.cat([e, h_sum, h_diff, nb_msg], dim=-1)
    else:
      edge_in = torch.cat([e, h_sum, h_diff], dim=-1)

    e_upd = self.edge_updaters[layer_idx](edge_in)
    edge_norm = self.edge_norms[layer_idx]
    if isinstance(edge_norm, GraphNorm):
      edge_batch = batch[src] if batch is not None else None
      e_upd = edge_norm(e_upd, edge_batch)
    else:
      e_upd = edge_norm(e_upd)
    e_upd = F.relu(e_upd)
    if layer_idx < self.num_layers - 1 and self.gnn_dropout > 0.0:
      e_upd = F.dropout(e_upd, p=self.gnn_dropout, training=self.training)
    return e + e_upd

  def _encode_appearance(self, data, batch=None):
    if (
        self.app_feat_dim <= 0
        or not hasattr(data, "x_app")
        or data.x_app is None
    ):
      return None, None

    x_app = data.x_app

    if (
        self.training
        and self.use_center_norm
        and hasattr(self, "app_running_mean")
        and batch is not None
        and x_app.numel() > 0
    ):
      with torch.no_grad():
        img_means = global_mean_pool(x_app.detach(), batch)
        batch_mean = img_means.mean(dim=0).view(1, 1, -1)
        if not bool(self.app_mean_init.item()):
          self.app_running_mean.copy_(batch_mean)
          self.app_mean_init.fill_(True)
        else:
          self.app_running_mean.lerp_(batch_mean, 0.05)

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
      x_det = x_app.detach()
      if (
          self.center_vis_edge
          and self.use_center_norm
          and hasattr(self, "app_running_mean")
      ):
        active_n = x_det.abs().sum(dim=-1, keepdim=True) > 0
        x_det = torch.where(
            active_n, x_det - self.app_running_mean.squeeze(0), x_det
        )
      x_norm = F.normalize(x_det, p=2, dim=-1)
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

    rev_idx, has_rev = (
        _build_rev_edge_index(edge_index, x.size(0))
        if self.use_nb_relay
        else (None, None)
    )

    layer_outputs = []
    edge_outputs = []
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

      if self.use_edge_co_update and (
          self.use_edge_readout or i < self.num_layers - 1
      ):
        edge_attr = self._update_edge_state(
            edge_attr, x, edge_index, batch, i, rev_idx=rev_idx, has_rev=has_rev
        )

      if self.use_full_app_prop and x_app is not None:
        x_app = self._propagate_full_appearance(
            x_app, x, edge_attr, edge_index, i
        )

      if self.use_jk:
        layer_outputs.append(x)
        if self.use_edge_readout:
          edge_outputs.append(edge_attr)

    x_final = torch.cat(layer_outputs, dim=-1) if self.use_jk else x
    if self.use_edge_readout:
      e_final = torch.cat(edge_outputs, dim=-1) if self.use_jk else edge_attr
      if edge_index.numel() > 0:
        e_node = scatter(
            e_final, edge_index[1], dim=0, dim_size=x_final.size(0), reduce="mean"
        )
      else:
        e_node = torch.zeros_like(x_final)
      x_final = torch.cat([x_final, e_node], dim=-1)

    x_final = self._center_gnn_nodes(x_final, batch)
    out = self._apply_pooling(x_final, batch)

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

    rev_idx1, has_rev1 = (
        _build_rev_edge_index(edge_index1, h1.size(0))
        if self.use_nb_relay
        else (None, None)
    )
    rev_idx2, has_rev2 = (
        _build_rev_edge_index(edge_index2, h2.size(0))
        if self.use_nb_relay
        else (None, None)
    )

    layer_outputs1 = []
    layer_outputs2 = []
    edge_outputs1 = []
    edge_outputs2 = []

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

      if self.use_edge_co_update and (
          self.use_edge_readout or i < self.num_layers - 1
      ):
        e1 = self._update_edge_state(
            e1, h1, edge_index1, batch1, i, rev_idx=rev_idx1, has_rev=has_rev1
        )
        e2 = self._update_edge_state(
            e2, h2, edge_index2, batch2, i, rev_idx=rev_idx2, has_rev=has_rev2
        )

      # Step 3: Coupled Multi-Hop Propagation of Full 1,120-D Vectors
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
        if self.use_edge_readout:
          edge_outputs1.append(e1)
          edge_outputs2.append(e2)

    h1_final = torch.cat(layer_outputs1, dim=-1) if self.use_jk else h1
    h2_final = torch.cat(layer_outputs2, dim=-1) if self.use_jk else h2
    if self.use_edge_readout:
      e1_final = torch.cat(edge_outputs1, dim=-1) if self.use_jk else e1
      e2_final = torch.cat(edge_outputs2, dim=-1) if self.use_jk else e2
      e1_node = (
          scatter(
              e1_final, edge_index1[1], dim=0, dim_size=h1_final.size(0), reduce="mean"
          )
          if edge_index1.numel() > 0
          else torch.zeros_like(h1_final)
      )
      e2_node = (
          scatter(
              e2_final, edge_index2[1], dim=0, dim_size=h2_final.size(0), reduce="mean"
          )
          if edge_index2.numel() > 0
          else torch.zeros_like(h2_final)
      )
      h1_final = torch.cat([h1_final, e1_node], dim=-1)
      h2_final = torch.cat([h2_final, e2_node], dim=-1)

    h1_final = self._center_gnn_nodes(h1_final, batch1)
    h2_final = self._center_gnn_nodes(h2_final, batch2)

    if self.use_reconstruction_dist:
      # Step 4: Centered Set Reconstruction + Standalone GNN Distance for Co-Training
      dist, gnn_dist = compute_cross_transformer_reconstruction_distance(
          h1_final=h1_final,
          h2_final=h2_final,
          batch1=batch1,
          batch2=batch2,
          N_query=N_query,
          N_support=N_support,
          scale=self.scale,
          x1_app=x1_app if self.use_app_recon_fusion else None,
          x2_app=x2_app if self.use_app_recon_fusion else None,
          app_recon_alpha=(
              self.app_recon_alpha if self.use_app_recon_fusion else None
          ),
          use_center_norm=self.use_center_norm,
          app_running_mean=getattr(self, "app_running_mean", None),
          return_gnn_dist=True,
          use_global_dist=self.use_global_dist,
      )
      if self.use_edge_matching:
        e1_c = (
            self._center_gnn_edges(e1, batch1[edge_index1[0]])
            if edge_index1.numel() > 0
            else e1
        )
        e2_c = (
            self._center_gnn_edges(e2, batch2[edge_index2[0]])
            if edge_index2.numel() > 0
            else e2
        )
        d_edge, d_edge_geo = compute_gluestick_edge_reconstruction_distance(
            e1_final=e1_c,
            e2_final=e2_c,
            edge_index1=edge_index1,
            edge_index2=edge_index2,
            batch1=batch1,
            batch2=batch2,
            N_query=N_query,
            N_support=N_support,
            x1_app=x1_app if self.use_app_recon_fusion else None,
            x2_app=x2_app if self.use_app_recon_fusion else None,
            use_center_norm=self.use_center_norm,
            app_running_mean=getattr(self, "app_running_mean", None),
            edge_match_vis_logit=getattr(self, "edge_match_vis_logit", None),
            edge_match_vis_mode=self.edge_match_vis_mode,
            edge_match_diff_gamma=self.edge_match_diff_gamma,
        )
        beta = torch.sigmoid(self.edge_match_beta_logit)
        dist = (1.0 - beta) * dist + beta * d_edge
        gnn_dist = (1.0 - beta) * gnn_dist + beta * d_edge_geo
      self.last_gnn_dist = gnn_dist
      return dist, None, None
    else:
      pooled1 = self._apply_pooling(h1_final, batch1)
      pooled2 = self._apply_pooling(h2_final, batch2)
      z1_norm = F.normalize(pooled1, p=2, dim=-1)
      z2_norm = F.normalize(pooled2, p=2, dim=-1)
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
