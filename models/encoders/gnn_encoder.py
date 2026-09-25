# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    GATv2Conv,
    ResGatedGraphConv,
    GPSConv,
    TAGConv,
    global_mean_pool,
    global_max_pool,
    GraphNorm
)

# Defensive Import: Use compiled torch-scatter if available, fall back to native if not
try:
    from torch_scatter import scatter_add
    USE_TORCH_SCATTER = True
except ImportError:
    USE_TORCH_SCATTER = False


class TAGCN_EdgeAugmented(nn.Module):
    """
    Edge-Augmented TAGCN (from Narayanan et al., ICCV 2021).
    Integrates 14D differential edge features into the node state before 
    applying Topology Adaptive Graph Convolutions (arXiv:1710.10370).
    """
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, dropout, K_hops=2, use_input_mlp=False):
        super(TAGCN_EdgeAugmented, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. Input Projections (1-Layer or 2-Layer MLP)
        if use_input_mlp:
            self.node_embed = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.edge_embed = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        else:
            self.node_embed = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.edge_embed = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        # 2. Node-Edge Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 3. Standard Isotropic TAGCN Convolution Layers (Du et al., arXiv:1710.10370)
        self.convs = nn.ModuleList([
            TAGConv(hidden_dim, hidden_dim, K=K_hops) for _ in range(num_layers)
        ])
        
        # 4. Graph Normalization
        self.norms = nn.ModuleList([
            GraphNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        col = edge_index[1]

        # Embed nodes and edges independently
        x_proj = self.node_embed(x)
        edge_proj = self.edge_embed(edge_attr)

        # Aggregate edge features at target nodes
        if USE_TORCH_SCATTER:
            edge_context = scatter_add(edge_proj, col, dim=0, dim_size=x.size(0))
        else:
            edge_context = torch.zeros(x.size(0), edge_proj.size(1), device=x.device).index_add_(0, col, edge_proj)

        # Non-linear node-edge fusion via concatenation
        x = self.fusion(torch.cat([x_proj, edge_context], dim=-1))

        # Multi-scale Convolutions with Additive Residuals
        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = x + x_in
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim=11,
        edge_feat_dim=14,
        hidden_dim=128,
        proj_feat_dim=128,
        gnn_type='GINE',
        num_layers=3,
        dropout=0.1,
        norm_type='graph',
        use_dual_pool=False,      # Default to Mean-Only (Our 40.23% Champion!)
        train_eps=False,          # NEW: Learnable epsilon in GINE
        use_input_mlp=False       # NEW: 2-layer MLP input projection
    ):
        super(GraphEncoder, self).__init__()
        self.gnn_type = gnn_type
        self.use_dual_pool = use_dual_pool

        # 1. Custom TAGCN Branch Handling
        if gnn_type == 'TAGCN':
            self.tagcn = TAGCN_EdgeAugmented(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                K_hops=2,
                use_input_mlp=use_input_mlp
            )
            in_dim = hidden_dim * 2 if use_dual_pool else hidden_dim
            self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_feat_dim),
                nn.BatchNorm1d(proj_feat_dim)
            )
            return

        # 2. Input Projections (1-Layer Linear vs. 2-Layer Non-Linear MLP)
        if use_input_mlp:
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        else:
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )

        # 3. Sequential Backbone Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if gnn_type == 'GINE':
                nn_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                # Pass train_eps to GINEConv (trainable epsilon scalar)
                self.layers.append(GINEConv(nn_mlp, edge_dim=hidden_dim, train_eps=train_eps))
            elif gnn_type == 'GATv2':
                self.layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim))
            elif gnn_type == 'ResGated':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GPS':
                local_conv = GINEConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ), edge_dim=hidden_dim)
                self.layers.append(GPSConv(hidden_dim, local_conv, heads=4, dropout=dropout))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            # Normalization layers
            if norm_type == 'graph':
                self.norms.append(GraphNorm(hidden_dim))
            elif norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        in_dim = hidden_dim * 2 if use_dual_pool else hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_feat_dim),
            nn.BatchNorm1d(proj_feat_dim)
        )

    def forward(self, data):
        # 1. TAGCN Forward Path
        if self.gnn_type == 'TAGCN':
            x_nodes = self.tagcn(data)
            pooled = torch.cat([
                global_mean_pool(x_nodes, data.batch),
                global_max_pool(x_nodes, data.batch)
            ], dim=-1) if self.use_dual_pool else global_mean_pool(x_nodes, data.batch)
            return self.projector(pooled)

        # 2. Standard GNN Forward Path (GINE, ResGated, GATv2, GPS)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for i, layer in enumerate(self.layers):
            x_in = x
            if self.gnn_type in ['GINE', 'GATv2']:
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif self.gnn_type in ['ResGated', 'GPS']:
                x = layer(x, edge_index)
            
            x = self.norms[i](x, batch) if isinstance(self.norms[i], GraphNorm) else self.norms[i](x)
            x = F.relu(x)
            x = x + x_in

        pooled = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=-1) if self.use_dual_pool else global_mean_pool(x, batch)

        return self.projector(pooled)
