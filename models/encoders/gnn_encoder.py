# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    GlobalAttention,
    GPSConv,
    GraphNorm,
    PNAConv,
    ResGatedGraphConv,
    SAGPooling,
    TAGConv,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import (
    DeepSetsAggregation,
    GraphMultisetTransformer,
    MedianAggregation,
    MultiAggregation,
    SoftmaxAggregation,
)
from torch_geometric.utils import degree

try:
  from torch_scatter import scatter_add

  USE_TORCH_SCATTER = True
except ImportError:
  USE_TORCH_SCATTER = False


class TAGCN_EdgeAugmented(nn.Module):
  """Edge-Augmented TAGCN (Narayanan et al., ICCV 2021).

  Fuses node and edge representations before applying multi-hop TAGConv.
  """

  def __init__(
      self,
      node_feat_dim,
      edge_feat_dim,
      hidden_dim,
      num_layers,
      dropout,
      K_hops=2,
      use_input_mlp=False,
      use_jk=False,
  ):
    super(TAGCN_EdgeAugmented, self).__init__()
    self.num_layers = num_layers
    self.dropout = dropout
    self.use_jk = use_jk

    if use_input_mlp:
      self.node_embed = nn.Sequential(
          nn.Linear(node_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
      self.edge_embed = nn.Sequential(
          nn.Linear(edge_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
    else:
      self.node_embed = nn.Sequential(
          nn.Linear(node_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )
      self.edge_embed = nn.Sequential(
          nn.Linear(edge_feat_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU(),
      )

    self.fusion = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
    )
    self.convs = nn.ModuleList(
        [TAGConv(hidden_dim, hidden_dim, K=K_hops) for _ in range(num_layers)]
    )
    self.norms = nn.ModuleList(
        [GraphNorm(hidden_dim) for _ in range(num_layers)]
    )

  def forward(self, data):
    x, edge_index, edge_attr, batch = (
        data.x,
        data.edge_index,
        data.edge_attr,
        data.batch,
    )
    col = edge_index[1]

    x_proj = self.node_embed(x)
    edge_proj = self.edge_embed(edge_attr)

    if USE_TORCH_SCATTER:
      edge_context = scatter_add(edge_proj, col, dim=0, dim_size=x.size(0))
    else:
      edge_context = torch.zeros(
          x.size(0), edge_proj.size(1), device=x.device
      ).index_add_(0, col, edge_proj)

    x = self.fusion(torch.cat([x_proj, edge_context], dim=-1))

    layer_outputs = []
    for i in range(self.num_layers):
      x_in = x
      x = self.convs[i](x, edge_index)
      x = self.norms[i](x, batch)
      x = F.relu(x)
      x = x + x_in
      x = F.dropout(x, p=self.dropout, training=self.training)
      if self.use_jk:
        layer_outputs.append(x)

    return torch.cat(layer_outputs, dim=-1) if self.use_jk else x


class GMNLayer(nn.Module):
  """Single GNN layer supporting intra-graph message passing and GMN cross-graph matching."""

  def __init__(
      self,
      conv_type: str,
      in_dim: int,
      out_dim: int,
      edge_dim: int = None,
      heads: int = 4,
      dropout: float = 0.0,
      act: str = "relu",
      norm: str = "batch",
      use_residual: bool = True,
  ):
    super().__init__()
    self.conv_type = conv_type.lower()
    self.use_residual = use_residual and (in_dim == out_dim)
    self.dropout = nn.Dropout(dropout)
    self.scale = 1.0 / (out_dim**0.5)

    # 1. Activation function
    if act == "relu":
      self.act = nn.ReLU()
    elif act == "gelu":
      self.act = nn.GELU()
    elif act == "leaky_relu":
      self.act = nn.LeakyReLU(0.2)
    else:
      self.act = nn.Identity()

    # 2. Convolution operator
    if self.conv_type == "gcn":
      self.conv = GCNConv(in_dim, out_dim)
    elif self.conv_type == "gat":
      self.conv = GATConv(
          in_dim, out_dim // heads, heads=heads, dropout=dropout
      )
    elif self.conv_type == "gin":
      mlp = nn.Sequential(
          nn.Linear(in_dim, out_dim),
          self.act,
          nn.Linear(out_dim, out_dim),
      )
      self.conv = GINConv(mlp, train_eps=True)
    elif self.conv_type == "gine":
      mlp = nn.Sequential(
          nn.Linear(in_dim, out_dim),
          self.act,
          nn.Linear(out_dim, out_dim),
      )
      self.conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
    else:
      raise ValueError(f"Unsupported conv_type: {conv_type}")

    # 3. Normalization layer
    if norm == "batch":
      self.norm = nn.BatchNorm1d(out_dim)
    elif norm == "layer":
      self.norm = nn.LayerNorm(out_dim)
    else:
      self.norm = nn.Identity()

    # 4. GMN Cross-matching heads (Active only when cross_match=True)
    self.proj_match = nn.Linear(out_dim, out_dim)
    self.node_update = nn.Sequential(
        nn.Linear(out_dim * 2, out_dim),
        self.act,
        nn.Linear(out_dim, out_dim),
    )

  def forward(
      self,
      x: torch.Tensor,
      edge_index: torch.Tensor,
      edge_attr: torch.Tensor = None,
      cross_match: bool = False,
      mu: torch.Tensor = None,
  ) -> torch.Tensor:
    residual = x

    # Step A: Intra-graph message passing
    if self.conv_type == "gine":
      out = self.conv(x, edge_index, edge_attr=edge_attr)
    else:
      out = self.conv(x, edge_index)

    out = self.norm(out)
    out = self.act(out)
    out = self.dropout(out)

    if self.use_residual:
      out = out + residual

    # Step B: Joint node state update with cross-graph matching residuals
    if cross_match and (mu is not None):
      out = self.node_update(torch.cat([out, self.proj_match(mu)], dim=-1))

    return out


class GraphEncoder(nn.Module):
  """Modular GNN Backbone for Shock Graphs.

  Supports GINE, GPS, PNA, GATv2, ResGated, and TAGCN. Routes across 7 pooling
  and readout paradigms via `pooling_method`. Now supports integrated GMN
  cross-graph matching as well.
  """

  def __init__(
      self,
      node_feat_dim=11,
      edge_feat_dim=14,
      hidden_dim=128,
      proj_feat_dim=640,
      gnn_type="GINE",
      num_layers=3,
      dropout=0.1,
      norm_type="graph",
      pooling_method="mean",
      train_eps=True,
      use_input_mlp=True,
      use_jk=True,
      **kwargs,
  ):
    super(GraphEncoder, self).__init__()
    self.gnn_type = gnn_type
    self.use_jk = use_jk
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.pooling_method = str(pooling_method).lower()
    self.scale = 1.0 / (hidden_dim**0.5)

    # Dimension after multi-layer concatenation (JK-Net)
    effective_dim = (hidden_dim * num_layers) if use_jk else hidden_dim

    # Determine input projection dimension based on readout operator
    if self.pooling_method in ["dual_pool", "mean_max"]:
      in_dim = effective_dim * 2
    elif self.pooling_method == "multi_moment":
      in_dim = effective_dim * 4  # Concatenates [Mean || Std || Min || Max]
    else:
      in_dim = effective_dim

    # 1. TAGCN Path
    if gnn_type == "TAGCN":
      self.tagcn = TAGCN_EdgeAugmented(
          node_feat_dim=node_feat_dim,
          edge_feat_dim=edge_feat_dim,
          hidden_dim=hidden_dim,
          num_layers=num_layers,
          dropout=dropout,
          K_hops=2,
          use_input_mlp=use_input_mlp,
          use_jk=use_jk,
      )
      self._setup_pooling_modules(effective_dim)
      self.projector = nn.Sequential(
          nn.Linear(in_dim, proj_feat_dim),
          nn.BatchNorm1d(proj_feat_dim),
      )

      # Isolated GMN matching heads for TAGCN
      self.proj_match_tagcn = nn.Linear(effective_dim, effective_dim)
      self.node_update_tagcn = nn.Sequential(
          nn.Linear(effective_dim * 2, effective_dim),
          nn.GELU(),
          nn.Linear(effective_dim, effective_dim),
      )
      return

    # 2. Input Encoders (Node & Edge Projections)
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

    # 3. GNN Message Passing Convolution Layers
    self.layers = nn.ModuleList()
    self.norms = nn.ModuleList()

    # Prior degree histogram for PNA architecture
    self.register_buffer(
        "deg_histogram",
        torch.tensor(
            [0, 31275978, 254592, 26943186, 2124], dtype=torch.float
        ),
    )

    for _ in range(num_layers):
      if gnn_type == "GINE":
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers.append(
            GINEConv(nn_mlp, edge_dim=hidden_dim, train_eps=train_eps)
        )

      elif gnn_type == "GPS":
        local_conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=hidden_dim,
            train_eps=train_eps,
        )
        self.layers.append(
            GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=4,
                dropout=dropout,
                attn_type="multihead",
            )
        )

      elif gnn_type == "PNA":
        self.layers.append(
            PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=["mean", "min", "max", "std"],
                scalers=["identity", "amplification", "attenuation"],
                deg=self.deg_histogram,
                edge_dim=hidden_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )

      elif gnn_type == "GATv2":
        self.layers.append(
            GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=4,
                concat=False,
                edge_dim=hidden_dim,
            )
        )

      elif gnn_type == "ResGated":
        self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim))

      else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

      if norm_type == "graph":
        self.norms.append(GraphNorm(hidden_dim))
      elif norm_type == "layer":
        self.norms.append(nn.LayerNorm(hidden_dim))
      elif norm_type == "batch":
        self.norms.append(nn.BatchNorm1d(hidden_dim))

    # 4. Pooling / Readout Setup
    self._setup_pooling_modules(effective_dim)

    # 5. Output Projection Head to Metric Space (640D)
    self.projector = nn.Sequential(
        nn.Linear(in_dim, proj_feat_dim),
        nn.BatchNorm1d(proj_feat_dim),
    )

    # 6. GMN Cross-Graph Matching Heads (added for GMN support, keeps existing weight keys intact)
    self.proj_match = nn.Linear(hidden_dim, hidden_dim)
    self.node_update = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )

  def _setup_pooling_modules(self, effective_dim):
    """Initializes PyG aggregation and pooling modules without hardcoded node sizes."""
    if self.pooling_method == "median":
      self.pool_op = MedianAggregation()
    elif self.pooling_method == "multi_moment":
      self.pool_op = MultiAggregation(
          aggrs=["mean", "std", "min", "max"], mode="cat"
      )
    elif self.pooling_method == "global_attention":
      gate_nn = nn.Sequential(
          nn.Linear(effective_dim, effective_dim // 2),
          nn.ReLU(),
          nn.Linear(effective_dim // 2, 1),
      )
      self.pool_op = GlobalAttention(gate_nn=gate_nn)
    elif self.pooling_method == "softmax":
      self.pool_op = SoftmaxAggregation(learn=True)
    elif self.pooling_method == "deep_sets":
      phi = nn.Sequential(
          nn.Linear(effective_dim, effective_dim),
          nn.ReLU(),
          nn.Linear(effective_dim, effective_dim),
      )
      rho = nn.Sequential(
          nn.Linear(effective_dim, effective_dim), nn.ReLU()
      )
      self.pool_op = DeepSetsAggregation(local_nn=phi, global_nn=rho)
    elif self.pooling_method == "gmt":
      self.pool_op = GraphMultisetTransformer(
          channels=effective_dim, k=4, num_encoder_blocks=1, heads=2
      )
    elif self.pooling_method == "sag_pool":
      self.sag = SAGPooling(in_channels=effective_dim, ratio=0.7)

  def _apply_pooling(self, x_final, edge_index, edge_attr, batch):
    """Dispatches the configured pooling method to produce a single graph vector."""
    if self.pooling_method == "mean":
      return global_mean_pool(x_final, batch)
    elif self.pooling_method in ["dual_pool", "mean_max"]:
      p_mean = global_mean_pool(x_final, batch)
      p_max = global_max_pool(x_final, batch)
      return torch.cat([p_mean, p_max], dim=-1)
    elif self.pooling_method in [
        "median",
        "multi_moment",
        "softmax",
        "deep_sets",
        "gmt",
    ]:
      return self.pool_op(x_final, index=batch)
    elif self.pooling_method == "global_attention":
      return self.pool_op(x_final, batch)
    elif self.pooling_method == "sag_pool":
      x_pruned, _, _, batch_pruned, _ = self.sag(
          x_final, edge_index, edge_attr=edge_attr, batch=batch
      )
      return global_mean_pool(x_pruned, batch_pruned)
    else:
      return global_mean_pool(x_final, batch)

  def forward(self, data):
    """Standard forward pass on a single graph or batch (Stage 1 / Unimodal)."""
    if self.gnn_type == "TAGCN":
      x_nodes = self.tagcn(data)
      pooled = self._apply_pooling(
          x_nodes, data.edge_index, data.edge_attr, data.batch
      )
      return self.projector(pooled)

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
      if self.gnn_type in ["GINE", "GATv2", "PNA"]:
        x = layer(x, edge_index, edge_attr=edge_attr)
      elif self.gnn_type == "GPS":
        x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
      elif self.gnn_type == "ResGated":
        x = layer(x, edge_index)

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
    pooled = self._apply_pooling(x_final, edge_index, edge_attr, batch)
    return self.projector(pooled)

  def forward_gmn(self, g1, g2):
    """GMN forward pass: Joint cross-graph propagation over paired batches.

    Keeps GINE, GPS, PNA, GATv2, ResGated, and TAGCN fully supported.
    Uses batch_mask to restrict attention strictly within matching graph pairs.
    """
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

    # 1. Handle TAGCN branch
    if self.gnn_type == "TAGCN":
      x1 = self.tagcn(g1)
      x2 = self.tagcn(g2)

      deg1 = degree(edge_index1[0], num_nodes=x1.size(0))
      deg2 = degree(edge_index2[0], num_nodes=x2.size(0))
      mask1 = (deg1 == 1) | (deg1 >= 3)
      mask2 = (deg2 == 1) | (deg2 >= 3)

      # Safety Guard: If any graph has 0 seed nodes, default to all nodes
      if not mask1.any():
        mask1 = torch.ones_like(mask1)
      if not mask2.any():
        mask2 = torch.ones_like(mask2)

      x1_seeds = x1[mask1]
      x2_seeds = x2[mask2]
      batch1_seeds = batch1[mask1]
      batch2_seeds = batch2[mask2]

      scale = 1.0 / (x1.size(-1)**0.5)
      scores = torch.mm(x1_seeds, x2_seeds.t()) * scale

      # Restrict cross-attention to pairs from the same graph index in the batch
      batch_mask = batch1_seeds.unsqueeze(1) == batch2_seeds.unsqueeze(0)
      scores = scores.masked_fill(~batch_mask, -1e9)

      a12 = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
      scores_t = scores.t().masked_fill(~batch_mask.t(), -1e9)
      a21 = torch.nan_to_num(F.softmax(scores_t, dim=-1), nan=0.0)

      retrieved2 = torch.mm(a12, x2_seeds)
      retrieved1 = torch.mm(a21, x1_seeds)

      mu1 = torch.zeros_like(x1)
      mu2 = torch.zeros_like(x2)
      mu1[mask1] = x1_seeds - retrieved2
      mu2[mask2] = x2_seeds - retrieved1

      # Eliminate bias leakage from non-seed nodes
      proj_mu1 = self.proj_match_tagcn(mu1)
      proj_mu2 = self.proj_match_tagcn(mu2)
      proj_mu1[~mask1] = 0.0
      proj_mu2[~mask2] = 0.0

      x1 = self.node_update_tagcn(torch.cat([x1, proj_mu1], dim=-1))
      x2 = self.node_update_tagcn(torch.cat([x2, proj_mu2], dim=-1))

      pooled1 = self._apply_pooling(x1, edge_index1, edge_attr1, batch1)
      pooled2 = self._apply_pooling(x2, edge_index2, edge_attr2, batch2)

      z1 = self.projector(pooled1)
      z2 = self.projector(pooled2)

      z1_norm = F.normalize(z1, p=2, dim=-1)
      z2_norm = F.normalize(z2, p=2, dim=-1)
      dist = torch.sum((z1_norm - z2_norm) ** 2, dim=-1)
      return dist, z1_norm, z2_norm

    # 2. Standard GNN GMN path (GINE, GPS, PNA, GATv2, ResGated)
    h1 = self.node_encoder(x1)
    h2 = self.node_encoder(x2)

    e1 = (
        self.edge_encoder(edge_attr1)
        if (self.edge_encoder is not None and edge_attr1 is not None)
        else None
    )
    e2 = (
        self.edge_encoder(edge_attr2)
        if (self.edge_encoder is not None and edge_attr2 is not None)
        else None
    )

    deg1 = degree(edge_index1[0], num_nodes=h1.size(0))
    deg2 = degree(edge_index2[0], num_nodes=h2.size(0))
    mask1 = (deg1 == 1) | (deg1 >= 3)
    mask2 = (deg2 == 1) | (deg2 >= 3)

    # Safety Guard: If any graph has 0 seed nodes, default to all nodes
    if not mask1.any():
      mask1 = torch.ones_like(mask1)
    if not mask2.any():
      mask2 = torch.ones_like(mask2)

    batch1_seeds = batch1[mask1]
    batch2_seeds = batch2[mask2]
    batch_mask = batch1_seeds.unsqueeze(1) == batch2_seeds.unsqueeze(0)

    layer_outputs1 = []
    layer_outputs2 = []

    for i, layer in enumerate(self.layers):
      h1_in = h1
      h2_in = h2

      # A. Intra-graph message passing
      if self.gnn_type in ["GINE", "GATv2", "PNA"]:
        h1 = layer(h1, edge_index1, edge_attr=e1)
        h2 = layer(h2, edge_index2, edge_attr=e2)
      elif self.gnn_type == "GPS":
        h1 = layer(h1, edge_index1, batch=batch1, edge_attr=e1)
        h2 = layer(h2, edge_index2, batch=batch2, edge_attr=e2)
      elif self.gnn_type == "ResGated":
        h1 = layer(h1, edge_index1)
        h2 = layer(h2, edge_index2)

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

      # B. Inter-graph cross-attention matching (GMN)
      h1_seeds = h1[mask1]
      h2_seeds = h2[mask2]

      # Pairwise cross-attention scores between nodes
      scores = torch.mm(h1_seeds, h2_seeds.t()) * self.scale
      scores = scores.masked_fill(~batch_mask, -1e9)

      a12 = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
      scores_t = scores.t().masked_fill(~batch_mask.t(), -1e9)
      a21 = torch.nan_to_num(F.softmax(scores_t, dim=-1), nan=0.0)

      retrieved2 = torch.mm(a12, h2_seeds)
      retrieved1 = torch.mm(a21, h1_seeds)

      # Match vectors (residuals)
      mu1_seeds = h1_seeds - retrieved2
      mu2_seeds = h2_seeds - retrieved1

      mu1 = torch.zeros_like(h1)
      mu2 = torch.zeros_like(h2)
      mu1[mask1] = mu1_seeds
      mu2[mask2] = mu2_seeds

      # Eliminate bias leakage from non-seed nodes
      proj_mu1 = self.proj_match(mu1)
      proj_mu2 = self.proj_match(mu2)
      proj_mu1[~mask1] = 0.0
      proj_mu2[~mask2] = 0.0

      # C. Joint node state updates
      h1 = self.node_update(torch.cat([h1, proj_mu1], dim=-1))
      h2 = self.node_update(torch.cat([h2, proj_mu2], dim=-1))

      h1 = h1 + h1_in
      h2 = h2 + h2_in

      if self.use_jk:
        layer_outputs1.append(h1)
        layer_outputs2.append(h2)

    h1_final = torch.cat(layer_outputs1, dim=-1) if self.use_jk else h1
    h2_final = torch.cat(layer_outputs2, dim=-1) if self.use_jk else h2

    # Execute pooling & projection
    pooled1 = self._apply_pooling(h1_final, edge_index1, e1, batch1)
    pooled2 = self._apply_pooling(h2_final, edge_index2, e2, batch2)

    z1 = self.projector(pooled1)
    z2 = self.projector(pooled2)

    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    dist = torch.sum((z1_norm - z2_norm) ** 2, dim=-1)

    return dist, z1_norm, z2_norm
