# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    GATv2Conv,
    ResGatedGraphConv,
    GPSConv,
    PNAConv,
    TAGConv,
    global_mean_pool,
    global_max_pool,
    GraphNorm,
)

try:
  from torch_scatter import scatter_add

  USE_TORCH_SCATTER = True
except ImportError:
  USE_TORCH_SCATTER = False


class TAGCN_EdgeAugmented(nn.Module):
  """Edge-Augmented TAGCN (from Narayanan et al., ICCV 2021)."""

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

    if self.use_jk:
      return torch.cat(layer_outputs, dim=-1)
    return x


class GraphEncoder(nn.Module):

  def __init__(
      self,
      node_feat_dim=11,
      edge_feat_dim=14,
      hidden_dim=128,
      proj_feat_dim=640,
      gnn_type='GINE',
      num_layers=3,
      dropout=0.1,
      norm_type='graph',
      use_dual_pool=False,
      train_eps=True,
      use_input_mlp=True,
      use_jk=True,
      **kwargs,
  ):
    super(GraphEncoder, self).__init__()
    self.gnn_type = gnn_type
    self.use_dual_pool = use_dual_pool
    self.use_jk = use_jk
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim

    effective_dim = (hidden_dim * num_layers) if use_jk else hidden_dim
    in_dim = effective_dim * 2 if use_dual_pool else effective_dim

    # 1. TAGCN Branch
    if gnn_type == 'TAGCN':
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
      self.projector = nn.Sequential(
          nn.Linear(in_dim, proj_feat_dim), nn.BatchNorm1d(proj_feat_dim)
      )
      return

    # 2. Input Projections
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

    # 3. Sequential Backbone Layers
    self.layers = nn.ModuleList()
    self.norms = nn.ModuleList()

    # Register degree histogram as a model buffer so PyTorch moves it to CUDA automatically
    self.register_buffer(
        'deg_histogram',
        torch.tensor(
            [120, 8500, 3200, 14000, 1800, 450, 120, 20, 5], dtype=torch.float
        ),
    )

    for _ in range(num_layers):
      if gnn_type == 'GINE':
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers.append(
            GINEConv(nn_mlp, edge_dim=hidden_dim, train_eps=train_eps)
        )

      elif gnn_type == 'GPS':
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
                attn_type='multihead',
            )
        )

      elif gnn_type == 'PNA':
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.layers.append(
            PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=self.deg_histogram,
                edge_dim=hidden_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        )

      elif gnn_type == 'GATv2':
        self.layers.append(
            GATv2Conv(
                hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim
            )
        )

      elif gnn_type == 'ResGated':
        self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim))

      else:
        raise ValueError(
            f'Unknown GNN type: {gnn_type}. Choose from'
            " ['GINE', 'GPS', 'PNA', 'GATv2', 'TAGCN']."
        )

      # Normalization
      if norm_type == 'graph':
        self.norms.append(GraphNorm(hidden_dim))
      elif norm_type == 'layer':
        self.norms.append(nn.LayerNorm(hidden_dim))
      elif norm_type == 'batch':
        self.norms.append(nn.BatchNorm1d(hidden_dim))

    self.projector = nn.Sequential(
        nn.Linear(in_dim, proj_feat_dim), nn.BatchNorm1d(proj_feat_dim)
    )

  def forward(self, data):
    if self.gnn_type == 'TAGCN':
      x_nodes = self.tagcn(data)
      pooled = (
          torch.cat(
              [
                  global_mean_pool(x_nodes, data.batch),
                  global_max_pool(x_nodes, data.batch),
              ],
              dim=-1,
          )
          if self.use_dual_pool
          else global_mean_pool(x_nodes, data.batch)
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
      if self.gnn_type in ['GINE', 'GATv2', 'PNA']:
        x = layer(x, edge_index, edge_attr=edge_attr)
      elif self.gnn_type == 'GPS':
        x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
      elif self.gnn_type == 'ResGated':
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

    pooled = (
        torch.cat(
            [global_mean_pool(x_final, batch), global_max_pool(x_final, batch)],
            dim=-1,
        )
        if self.use_dual_pool
        else global_mean_pool(x_final, batch)
    )

    return self.projector(pooled)
