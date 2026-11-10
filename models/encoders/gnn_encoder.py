# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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
    a1_seeds=None,
    a2_seeds=None,
    app_match_logit=None,
):
  """Computes isolated pairwise cross-attention residuals strictly within each (q, s) pair."""
  H_q, M_q = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  H_q_norm = F.normalize(H_q, p=2, dim=-1)
  H_s_norm = F.normalize(H_s, p=2, dim=-1)

  # Pairwise cosine similarity strictly between Query q and Support s: [N_q, N_s, K_q, K_s]
  cos_sim = torch.einsum("qid, sjd -> qsij", H_q_norm, H_s_norm)

  # Option 4: Appearance-Guided Cross-Graph Seed Matching (Projection-Free)
  if (
      a1_seeds is not None
      and a2_seeds is not None
      and app_match_logit is not None
  ):
    with torch.no_grad():
      A_q, _ = to_dense_batch(a1_seeds, batch1_seeds, batch_size=N_query)
      A_s, _ = to_dense_batch(a2_seeds, batch2_seeds, batch_size=N_support)
      q_act = A_q.abs().sum(dim=(-2, -1)) > 0
      s_act = A_s.abs().sum(dim=(-2, -1)) > 0
      pair_act = (
          (q_act.unsqueeze(1) & s_act.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)
      )
      A_q_norm = F.normalize(A_q, p=2, dim=-1)
      A_s_norm = F.normalize(A_s, p=2, dim=-1)
      app_cos_sim = torch.einsum("qid, sjd -> qsij", A_q_norm, A_s_norm)
    w_app = 0.5 * torch.tanh(app_match_logit)
    blended_sim = (1.0 - w_app) * cos_sim + w_app * app_cos_sim
    cos_sim = torch.where(pair_act, blended_sim, cos_sim)

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


def _build_recon_mask(seed_mask, batch, num_graphs, max_nodes=256):
  """Returns all nodes up to `max_nodes` per graph (keeping 100% of seeds + uniform degree-2 stride on outliers)."""
  counts = torch.bincount(batch, minlength=num_graphs)
  if int(counts.max().item()) <= max_nodes:
    return torch.ones_like(seed_mask, dtype=torch.bool)

  recon_mask = seed_mask.clone()
  for g in range(num_graphs):
    c = int(counts[g].item())
    if c <= max_nodes:
      recon_mask[batch == g] = True
    else:
      g_idx = torch.where(batch == g)[0]
      g_seeds = seed_mask[g_idx]
      n_seeds = int(g_seeds.sum().item())
      if n_seeds >= max_nodes:
        seed_pos = g_idx[g_seeds]
        recon_mask[g_idx] = False
        step = torch.linspace(
            0, len(seed_pos) - 1, steps=max_nodes, device=seed_mask.device
        ).long()
        recon_mask[seed_pos[step]] = True
      else:
        rem = max_nodes - n_seeds
        non_seed_pos = g_idx[~g_seeds]
        step = torch.linspace(
            0, len(non_seed_pos) - 1, steps=rem, device=seed_mask.device
        ).long()
        recon_mask[non_seed_pos[step]] = True
  return recon_mask


def _recon_chunk_step(
    H_q,
    H_s,
    M_q,
    M_s,
    K_s_counts,
    X_q=None,
    X_s=None,
    X_q_std=None,
    X_s_std=None,
    pair_active=None,
    w_recon=None,
    W_q=None,
    W_s=None,
    use_sinkhorn=False,
    sinkhorn_iters=3,
):
  """Single query-chunk set reconstruction step returning stacked [mm_dist, gnn_dist]."""
  cos_sim_gnn = torch.einsum("qid, sjd -> qsij", H_q, H_s)
  K_q_counts = M_q.sum(dim=-1, keepdim=True).clamp(min=1.0)

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
  gnn_dist = 0.5 * (dist_q2s_g + dist_s2q_g)

  # 2. Blended Multimodal Set Reconstruction Distance
  cos_sim = cos_sim_gnn
  if (
      X_q is not None
      and X_s is not None
      and pair_active is not None
      and w_recon is not None
  ):
    app_cos_sim = torch.einsum("qid, sjd -> qsij", X_q, X_s)
    if X_q_std is not None and X_s_std is not None:
      std_cos_sim = torch.einsum("qid, sjd -> qsij", X_q_std, X_s_std)
      app_cos_sim = 0.80 * app_cos_sim + 0.20 * std_cos_sim
    blended_cos = (1.0 - w_recon) * cos_sim_gnn + w_recon * app_cos_sim
    cos_sim = torch.where(pair_active, blended_cos, cos_sim_gnn)

  scores = cos_sim * 10.0

  if use_sinkhorn:
    valid_mask = M_q.unsqueeze(1).unsqueeze(3) & M_s.unsqueeze(0).unsqueeze(2)
    log_P = scores.masked_fill(~valid_mask, -1e9)

    if W_q is not None:
      log_wq = W_q.clamp(min=1e-6).log().unsqueeze(-1)
    else:
      unif_q = M_q.float() / K_q_counts
      log_wq = unif_q.clamp(min=1e-6).log().unsqueeze(1).unsqueeze(-1)

    if W_s is not None:
      log_ws = W_s.clamp(min=1e-6).log().unsqueeze(-2)
    else:
      unif_s = M_s.float() / K_s_counts.t()
      log_ws = unif_s.clamp(min=1e-6).log().unsqueeze(0).unsqueeze(-2)

    log_P_row = log_P
    log_P_col = log_P
    for _ in range(int(sinkhorn_iters)):
      log_P_row = (
          log_P_col - torch.logsumexp(log_P_col, dim=-1, keepdim=True) + log_wq
      )
      log_P_row = log_P_row.masked_fill(~valid_mask, -1e9)
      log_P_col = (
          log_P_row - torch.logsumexp(log_P_row, dim=-2, keepdim=True) + log_ws
      )
      log_P_col = log_P_col.masked_fill(~valid_mask, -1e9)

    Pi_q2s = (
        torch.nan_to_num(torch.exp(log_P_row), nan=0.0) * valid_mask.float()
    )
    Pi_s2q = (
        torch.nan_to_num(torch.exp(log_P_col), nan=0.0) * valid_mask.float()
    )
    dist_q2s = 2.0 - 2.0 * (Pi_q2s * cos_sim).sum(dim=(-2, -1))
    dist_s2q = 2.0 - 2.0 * (Pi_s2q * cos_sim).sum(dim=(-2, -1))
    mm_dist = 0.5 * (dist_q2s + dist_s2q)
    return torch.stack([mm_dist, gnn_dist], dim=0)

  # Direction 1: Query -> Support
  A_q2s = torch.nan_to_num(
      F.softmax(scores.masked_fill(~M_s.unsqueeze(0).unsqueeze(2), -1e9), dim=-1),
      nan=0.0,
  )
  err_q = 2.0 - 2.0 * (A_q2s * cos_sim).sum(dim=-1)
  err_q = err_q.masked_fill(~M_q.unsqueeze(1), 0.0)
  if W_q is not None:
    dist_q2s = (err_q * W_q).sum(dim=-1)
  else:
    dist_q2s = err_q.sum(dim=-1) / K_q_counts

  # Direction 2: Support -> Query
  A_s2q = torch.nan_to_num(
      F.softmax(scores.masked_fill(~M_q.unsqueeze(1).unsqueeze(3), -1e9), dim=-2),
      nan=0.0,
  )
  err_s = 2.0 - 2.0 * (A_s2q * cos_sim).sum(dim=-2)
  err_s = err_s.masked_fill(~M_s.unsqueeze(0), 0.0)
  if W_s is not None:
    dist_s2q = (err_s * W_s).sum(dim=-1)
  else:
    dist_s2q = err_s.sum(dim=-1) / K_s_counts

  mm_dist = 0.5 * (dist_q2s + dist_s2q)
  return torch.stack([mm_dist, gnn_dist], dim=0)


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
    x1_app=None,
    x2_app=None,
    x1_app_std=None,
    x2_app_std=None,
    app_recon_alpha=None,
    use_sinkhorn=False,
    sinkhorn_iters=3,
    use_center_norm=False,
    use_app_only_recon=False,
    return_gnn_dist=False,
    query_chunk_size=15,
):
  """Computes unpooled Bidirectional CrossTransformers Set-to-Set Keypoint Distance."""
  h1_seeds = h1_final[mask1]
  h2_seeds = h2_final[mask2]
  batch1_seeds = batch1[mask1]
  batch2_seeds = batch2[mask2]

  if projector is not None and h1_seeds.size(-1) == projector[0].in_features:
    all_seeds = torch.cat([h1_seeds, h2_seeds], dim=0)
    proj_seeds = projector(all_seeds)
    h1_seeds = proj_seeds[: h1_seeds.size(0)]
    h2_seeds = proj_seeds[h1_seeds.size(0) :]

  H_q_all, M_q_all = to_dense_batch(h1_seeds, batch1_seeds, batch_size=N_query)
  H_s, M_s = to_dense_batch(h2_seeds, batch2_seeds, batch_size=N_support)

  # Global GNN Graph Anchor
  h_q_global = F.normalize(
      (H_q_all * M_q_all.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
  )
  h_s_global = F.normalize((H_s * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1)
  gnn_global_cos = torch.matmul(h_q_global, h_s_global.t())
  gnn_global_dist = 2.0 - 2.0 * gnn_global_cos
  global_cos = gnn_global_cos

  H_q_all = F.normalize(H_q_all, p=2, dim=-1)
  H_s = F.normalize(H_s, p=2, dim=-1)

  use_late_app = (
      app_recon_alpha is not None
      and x1_app is not None
      and x2_app is not None
  )
  if use_late_app:
    X_q_raw, _ = to_dense_batch(
        x1_app[mask1], batch1_seeds, batch_size=N_query
    )
    X_s_raw, _ = to_dense_batch(
        x2_app[mask2], batch2_seeds, batch_size=N_support
    )

    with torch.no_grad():
      X_q_det = X_q_raw.detach()
      X_s_det = X_s_raw.detach()
      q_active_all = X_q_det.abs().sum(dim=(-2, -1)) > 0
      s_active = X_s_det.abs().sum(dim=(-2, -1)) > 0

      q_mean_det = F.normalize(
          (X_q_det * M_q_all.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
      )
      s_mean_det = F.normalize(
          (X_s_det * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
      )

      fg_wq = torch.einsum("qid, qd -> qi", X_q_det, q_mean_det).clamp(min=1e-3)
      fg_wq = fg_wq.masked_fill(~M_q_all, 0.0)
      fg_wq = fg_wq / fg_wq.sum(dim=-1, keepdim=True).clamp(min=1e-6)
      unif_wq = M_q_all.float() / M_q_all.sum(dim=-1, keepdim=True).clamp(min=1.0)
      W_q_all = torch.where(
          q_active_all.unsqueeze(-1), 0.5 * unif_wq + 0.5 * fg_wq, unif_wq
      ).unsqueeze(1)

      fg_ws = torch.einsum("sjd, sd -> sj", X_s_det, s_mean_det).clamp(min=1e-3)
      fg_ws = fg_ws.masked_fill(~M_s, 0.0)
      fg_ws = fg_ws / fg_ws.sum(dim=-1, keepdim=True).clamp(min=1e-6)
      unif_ws = M_s.float() / M_s.sum(dim=-1, keepdim=True).clamp(min=1.0)
      W_s_all = torch.where(
          s_active.unsqueeze(-1), 0.5 * unif_ws + 0.5 * fg_ws, unif_ws
      ).unsqueeze(0)
      pair_2d = q_active_all.unsqueeze(1) & s_active.unsqueeze(0)

    q_mean = F.normalize(
        (X_q_raw * M_q_all.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
    )
    s_mean = F.normalize(
        (X_s_raw * M_s.unsqueeze(-1)).sum(dim=1), p=2, dim=-1
    )
    app_global_cos = torch.matmul(q_mean, s_mean.t())

    # Step 4: Optional DeepEMD Per-Node Channel Mean Centering
    if use_center_norm:
      X_q_c = X_q_raw - X_q_raw.mean(dim=-1, keepdim=True)
      X_s_c = X_s_raw - X_s_raw.mean(dim=-1, keepdim=True)
      X_q_all = F.normalize(X_q_c, p=2, dim=-1)
      X_s_app = F.normalize(X_s_c, p=2, dim=-1)
    else:
      X_q_all = F.normalize(X_q_raw, p=2, dim=-1)
      X_s_app = F.normalize(X_s_raw, p=2, dim=-1)

    if x1_app_std is not None and x2_app_std is not None:
      X_q_std_raw, _ = to_dense_batch(
          x1_app_std[mask1], batch1_seeds, batch_size=N_query
      )
      X_s_std_raw, _ = to_dense_batch(
          x2_app_std[mask2], batch2_seeds, batch_size=N_support
      )
      X_q_std_all = F.normalize(X_q_std_raw, p=2, dim=-1)
      X_s_std_app = F.normalize(X_s_std_raw, p=2, dim=-1)
    else:
      X_q_std_all = None
      X_s_std_app = None

    if use_app_only_recon:
      w_recon = torch.tensor(1.0, device=H_q_all.device, dtype=H_q_all.dtype)
    else:
      w_recon = torch.sigmoid(app_recon_alpha + 1.1)
    blended_global_cos = (1.0 - w_recon) * gnn_global_cos + w_recon * app_global_cos
    global_cos = torch.where(pair_2d, blended_global_cos, gnn_global_cos)
  else:
    w_recon = None
    W_s_all = None
    X_q_all = None
    X_s_app = None
    X_q_std_all = None
    X_s_std_app = None

  global_dist = 2.0 - 2.0 * global_cos

  K_s_counts = M_s.sum(dim=-1, keepdim=True).t().clamp(min=1.0)
  dist_chunks = []
  use_ckpt = torch.is_grad_enabled() and (
      H_q_all.requires_grad
      or H_s.requires_grad
      or (X_q_all is not None and X_q_all.requires_grad)
  )

  # Process queries in checkpointed chunks so 4D [chunk, N_s, L_q, L_s] tensors never OOM during backward()
  for q0 in range(0, N_query, query_chunk_size):
    q1 = min(q0 + query_chunk_size, N_query)
    H_q = H_q_all[q0:q1]
    M_q = M_q_all[q0:q1]

    if use_late_app:
      X_q_chunk = X_q_all[q0:q1]
      X_q_std_chunk = X_q_std_all[q0:q1] if X_q_std_all is not None else None
      pair_active = (
          (q_active_all[q0:q1].unsqueeze(1) & s_active.unsqueeze(0))
          .unsqueeze(-1)
          .unsqueeze(-1)
      )
      W_q = W_q_all[q0:q1]
    else:
      X_q_chunk = None
      X_q_std_chunk = None
      pair_active = None
      W_q = None

    if use_ckpt:
      chunk_dist = checkpoint(
          _recon_chunk_step,
          H_q,
          H_s,
          M_q,
          M_s,
          K_s_counts,
          X_q_chunk,
          X_s_app,
          X_q_std_chunk,
          X_s_std_app,
          pair_active,
          w_recon,
          W_q,
          W_s_all,
          use_sinkhorn,
          sinkhorn_iters,
          use_reentrant=False,
      )
    else:
      chunk_dist = _recon_chunk_step(
          H_q,
          H_s,
          M_q,
          M_s,
          K_s_counts,
          X_q_chunk,
          X_s_app,
          X_q_std_chunk,
          X_s_std_app,
          pair_active,
          w_recon,
          W_q,
          W_s_all,
          use_sinkhorn,
          sinkhorn_iters,
      )

    dist_chunks.append(chunk_dist)

  stacked = torch.cat(dist_chunks, dim=1)  # [2, N_q, N_s]
  local_recon_dist, local_gnn_dist = stacked[0], stacked[1]
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
      gnn_type="GINE",
      num_layers=4,
      dropout=0.1,
      norm_type="graph",
      pooling_method="mean",
      train_eps=True,
      use_input_mlp=True,
      use_jk=True,
      use_reconstruction_dist=True,
      use_sinkhorn=False,
      sinkhorn_iters=3,
      use_direct_app=False,
      use_full_app_prop=False,
      use_center_norm=False,
      use_app_only_recon=False,
      app_feat_dim=0,
      app_mlp_dim=None,
      hc_use_std=False,
      app_node_scope="all",
      recon_node_scope="all",
      app_dropout=0.2,
      use_app_node_input=True,
      use_app_edges=False,
      use_app_film=False,
      use_app_cross_match=False,
      use_app_recon_fusion=False,
      **kwargs,
  ):
    super(GraphEncoder, self).__init__()
    self.gnn_type = gnn_type
    self.use_jk = use_jk
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.proj_feat_dim = proj_feat_dim
    self.pooling_method = str(pooling_method).lower()
    self.scale = 1.0 / (hidden_dim**0.5)
    self.use_reconstruction_dist = use_reconstruction_dist
    self.use_sinkhorn = bool(use_sinkhorn)
    self.sinkhorn_iters = int(sinkhorn_iters)
    self.use_center_norm = bool(use_center_norm)
    self.use_app_only_recon = bool(use_app_only_recon)
    self.last_gnn_dist = None

    self.app_feat_dim = int(app_feat_dim)
    self.app_mlp_dim = (
        int(app_mlp_dim) if app_mlp_dim is not None else int(hidden_dim)
    )
    self.hc_use_std = bool(hc_use_std)
    self.app_node_scope = str(app_node_scope).lower()
    self.recon_node_scope = str(recon_node_scope).lower()
    self.app_dropout = float(app_dropout)

    self.use_direct_app = bool(use_direct_app) and self.app_feat_dim > 0
    self.use_full_app_prop = bool(use_full_app_prop) and self.app_feat_dim > 0
    self.use_app_node_input = bool(use_app_node_input) and self.app_feat_dim > 0
    self.use_app_edges = bool(use_app_edges) and self.app_feat_dim > 0
    self.use_app_film = bool(use_app_film) and self.app_feat_dim > 0
    self.use_app_cross_match = (
        bool(use_app_cross_match) and self.app_feat_dim > 0
    )
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
    if self.use_direct_app:
      # --- Direct Full-1,120-D Mode (No app_input_mlp compression!) ---
      # Step 1 (Optional via use_app_node_input): 9-D Node Salience -> Linear(9, 128)
      if self.use_app_node_input:
        self.app_node_salience_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        self.app_node_alpha = nn.Parameter(torch.zeros(1, hidden_dim))

      # Step 2 (via use_app_edges): 10-D Edge Contrast -> Linear(10, 128)
      if self.use_app_edges:
        self.app_edge_contrast_mlp = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        self.app_edge_alpha = nn.Parameter(torch.zeros(1, hidden_dim))
    else:
      # --- Legacy MLP Stem Mode (Compresses 1,120-D -> app_mlp_dim -> 128-D) ---
      needs_app_stem = (
          self.use_app_node_input
          or self.use_app_edges
          or self.use_app_film
      )
      if needs_app_stem:
        self.app_input_mlp = nn.Sequential(
            nn.LayerNorm(self.app_feat_dim),
            nn.Linear(self.app_feat_dim, self.app_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.app_mlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.hc_use_std:
          self.app_std_mlp = nn.Sequential(
              nn.LayerNorm(self.app_feat_dim),
              nn.Linear(self.app_feat_dim, 32),
              nn.GELU(),
              nn.Linear(32, hidden_dim, bias=False),
              nn.LayerNorm(hidden_dim, elementwise_affine=False),
          )
          self.app_std_alpha = nn.Parameter(torch.zeros(1, hidden_dim))

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

    if self.use_app_film:
      self.app_film_layers = nn.ModuleList(
          [nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(num_layers)]
      )
      for film_lin in self.app_film_layers:
        nn.init.zeros_(film_lin.weight)
        nn.init.zeros_(film_lin.bias)

    if self.use_app_cross_match:
      self.app_match_logit = nn.Parameter(torch.tensor(0.0))

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

  def _get_stage_slices(self, x_app):
    D = x_app.size(-1)
    if D == 1120:
      return [x_app[:, :160], x_app[:, 160:480], x_app[:, 480:]]
    elif D == 960:
      return [x_app[:, :320], x_app[:, 320:], x_app]
    return [x_app, x_app, x_app]

  def _compute_direct_node_salience(self, x_app, batch):
    """Step 1: Computes 9-D channel-invariant node salience directly from full 1,120-D x_app."""
    with torch.no_grad():
      x_det = x_app.detach()
      active = (x_det.abs().sum(dim=-1, keepdim=True) > 0).type_as(x_app)
      num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 1
      slices = self._get_stage_slices(x_det)
      feats = []
      for sl in slices:
        norms = sl.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        mean_n = scatter(
            norms, batch, dim=0, dim_size=num_graphs, reduce="mean"
        )[batch].clamp(min=1e-6)
        e_is = norms / mean_n

        g_mean = F.normalize(
            scatter(sl, batch, dim=0, dim_size=num_graphs, reduce="mean"),
            p=2,
            dim=-1,
        )
        c_is = (
            F.normalize(sl, p=2, dim=-1) * g_mean[batch]
        ).sum(dim=-1, keepdim=True)

        p_is = sl.max(dim=-1, keepdim=True)[0] / (
            sl.abs().mean(dim=-1, keepdim=True) + 1e-5
        )
        feats.extend([e_is, c_is, p_is])
      s_vec = torch.cat(feats, dim=-1) * active

    active_mask = (x_app.detach().abs().sum(dim=-1, keepdim=True) > 0).type_as(
        x_app
    )
    return (
        torch.tanh(self.app_node_alpha)
        * self.app_node_salience_mlp(s_vec)
        * active_mask
    )

  def _compute_direct_edge_contrast(self, x_app, edge_index, batch):
    """Step 2: Computes 10-D boundary contrast directly from full 1,120-D vectors across edges (u, v)."""
    src, dst = edge_index[0], edge_index[1]
    with torch.no_grad():
      x_det = x_app.detach()
      u_act = x_det[src].abs().sum(dim=-1, keepdim=True) > 0
      v_act = x_det[dst].abs().sum(dim=-1, keepdim=True) > 0
      edge_active = (u_act & v_act).type_as(x_app)

      slices = self._get_stage_slices(x_det)
      edge_feats = []
      for sl in slices:
        sl_n = F.normalize(sl, p=2, dim=-1)
        u_n, v_n = sl_n[src], sl_n[dst]
        cos_uv = (u_n * v_n).sum(dim=-1, keepdim=True)
        l2_uv = (u_n - v_n).norm(p=2, dim=-1, keepdim=True)
        edge_feats.extend([cos_uv, l2_uv])

      full_n = F.normalize(x_det, p=2, dim=-1)
      full_cos = (full_n[src] * full_n[dst]).sum(dim=-1, keepdim=True)

      x_c = F.normalize(
          x_det - x_det.mean(dim=-1, keepdim=True), p=2, dim=-1
      )
      full_corr = (x_c[src] * x_c[dst]).sum(dim=-1, keepdim=True)

      n_u = x_det[src].norm(p=2, dim=-1, keepdim=True)
      n_v = x_det[dst].norm(p=2, dim=-1, keepdim=True)
      energy_jump = (n_u - n_v).abs() / (n_u + n_v + 1e-5)

      num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 1
      sl4 = slices[-1]
      g_mean4 = F.normalize(
          scatter(sl4, batch, dim=0, dim_size=num_graphs, reduce="mean"),
          p=2,
          dim=-1,
      )
      c4 = (
          F.normalize(sl4, p=2, dim=-1) * g_mean4[batch]
      ).sum(dim=-1, keepdim=True)
      cent_jump = (c4[src] - c4[dst]).abs()

      edge_feats.extend([full_cos, full_corr, energy_jump, cent_jump])
      c_uv = torch.cat(edge_feats, dim=-1) * edge_active

    return (
        torch.tanh(self.app_edge_alpha)
        * self.app_edge_contrast_mlp(c_uv)
        * edge_active
    )

  def _encode_appearance(self, data, deg=None, batch=None):
    if (
        self.app_feat_dim <= 0
        or not hasattr(data, "x_app")
        or data.x_app is None
    ):
      return None, None, None

    x_app = data.x_app
    x_app_std = getattr(data, "x_app_std", None)

    if self.training and self.app_dropout > 0.0 and batch is not None:
      num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
      if num_graphs > 0:
        keep_prob = 1.0 - self.app_dropout
        graph_keep = (
            torch.rand(num_graphs, 1, device=x_app.device) < keep_prob
        ).type_as(x_app)
        mask_g = graph_keep[batch]
        x_app = x_app * mask_g
        if x_app_std is not None:
          x_app_std = x_app_std * mask_g

    if not self.use_direct_app and hasattr(self, "app_input_mlp"):
      a = self.app_input_mlp(x_app)
      if hasattr(self, "app_std_mlp") and x_app_std is not None:
        a = a + torch.tanh(self.app_std_alpha) * self.app_std_mlp(x_app_std)
    else:
      a = None

    if self.app_node_scope == "seeds_only":
      if deg is None:
        deg = degree(data.edge_index[0], num_nodes=data.x.size(0))
      seed_mask = ((deg == 1) | (deg >= 3)).unsqueeze(-1).type_as(x_app)
      x_app = x_app * seed_mask
      if x_app_std is not None:
        x_app_std = x_app_std * seed_mask
      if a is not None:
        a = a * seed_mask

    return a, x_app, x_app_std

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

    deg = degree(edge_index[0], num_nodes=x.size(0))
    a, x_app, _ = self._encode_appearance(data, deg=deg, batch=batch)

    x = self.node_encoder(x)
    edge_attr = self.edge_encoder(edge_attr)

    if self.use_direct_app:
      if self.use_app_node_input and x_app is not None:
        x = x + self._compute_direct_node_salience(x_app, batch)
      if self.use_app_edges and x_app is not None:
        edge_attr = edge_attr + self._compute_direct_edge_contrast(
            x_app, edge_index, batch
        )
    else:
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
      if self.use_app_film and a is not None:
        raw_gamma, raw_beta = self.app_film_layers[i](a).chunk(2, dim=-1)
        x = (1.0 + 0.5 * torch.tanh(raw_gamma)) * x + 0.5 * torch.tanh(raw_beta)
      x = F.relu(x)
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

    deg1 = degree(edge_index1[0], num_nodes=h1.size(0))
    deg2 = degree(edge_index2[0], num_nodes=h2.size(0))
    mask1 = (deg1 == 1) | (deg1 >= 3)
    mask2 = (deg2 == 1) | (deg2 >= 3)

    if not mask1.any():
      mask1 = torch.ones_like(mask1)
    if not mask2.any():
      mask2 = torch.ones_like(mask2)

    a1, x1_app, x1_app_std = self._encode_appearance(
        g1, deg=deg1, batch=batch1
    )
    a2, x2_app, x2_app_std = self._encode_appearance(
        g2, deg=deg2, batch=batch2
    )

    if self.use_direct_app:
      if self.use_app_node_input and x1_app is not None and x2_app is not None:
        h1 = h1 + self._compute_direct_node_salience(x1_app, batch1)
        h2 = h2 + self._compute_direct_node_salience(x2_app, batch2)
      if self.use_app_edges and x1_app is not None and x2_app is not None:
        e1 = e1 + self._compute_direct_edge_contrast(
            x1_app, edge_index1, batch1
        )
        e2 = e2 + self._compute_direct_edge_contrast(
            x2_app, edge_index2, batch2
        )
    else:
      if self.use_app_node_input and a1 is not None and a2 is not None:
        gate_node = torch.tanh(self.app_node_alpha)
        h1 = h1 + gate_node * self.app_node_gate(a1)
        h2 = h2 + gate_node * self.app_node_gate(a2)

      if self.use_app_edges and a1 is not None and a2 is not None:
        e1 = e1 + self._compute_edge_appearance(a1, edge_index1)
        e2 = e2 + self._compute_edge_appearance(a2, edge_index2)

    batch1_seeds = batch1[mask1]
    batch2_seeds = batch2[mask2]

    a1_seeds = (
        x1_app[mask1]
        if (self.use_app_cross_match and x1_app is not None)
        else None
    )
    a2_seeds = (
        x2_app[mask2]
        if (self.use_app_cross_match and x2_app is not None)
        else None
    )
    app_match_logit = (
        self.app_match_logit if self.use_app_cross_match else None
    )

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

      if self.use_app_film and a1 is not None and a2 is not None:
        raw_g1, raw_b1 = self.app_film_layers[i](a1).chunk(2, dim=-1)
        raw_g2, raw_b2 = self.app_film_layers[i](a2).chunk(2, dim=-1)
        h1 = (1.0 + 0.5 * torch.tanh(raw_g1)) * h1 + 0.5 * torch.tanh(raw_b1)
        h2 = (1.0 + 0.5 * torch.tanh(raw_g2)) * h2 + 0.5 * torch.tanh(raw_b2)

      h1 = F.relu(h1) + h1_in
      h2 = F.relu(h2) + h2_in

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
          a1_seeds=a1_seeds,
          a2_seeds=a2_seeds,
          app_match_logit=app_match_logit,
      )

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
      if self.recon_node_scope == "all":
        recon_mask1 = _build_recon_mask(mask1, batch1, N_query, max_nodes=256)
        recon_mask2 = _build_recon_mask(mask2, batch2, N_support, max_nodes=256)
      else:
        recon_mask1 = mask1
        recon_mask2 = mask2

      # Step 4: Centered Set Reconstruction + Standalone GNN Distance for Co-Training
      dist, gnn_dist = compute_cross_transformer_reconstruction_distance(
          h1_final=h1_final,
          h2_final=h2_final,
          mask1=recon_mask1,
          mask2=recon_mask2,
          batch1=batch1,
          batch2=batch2,
          N_query=N_query,
          N_support=N_support,
          projector=self.projector,
          scale=self.scale,
          x1_app=x1_app if self.use_app_recon_fusion else None,
          x2_app=x2_app if self.use_app_recon_fusion else None,
          x1_app_std=x1_app_std if self.use_app_recon_fusion else None,
          x2_app_std=x2_app_std if self.use_app_recon_fusion else None,
          app_recon_alpha=(
              self.app_recon_alpha if self.use_app_recon_fusion else None
          ),
          use_sinkhorn=self.use_sinkhorn,
          sinkhorn_iters=self.sinkhorn_iters,
          use_center_norm=self.use_center_norm,
          use_app_only_recon=self.use_app_only_recon,
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
