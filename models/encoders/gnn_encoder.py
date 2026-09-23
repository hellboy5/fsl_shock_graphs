# models/encoders/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, 
    GATv2Conv, 
    ResGatedGraphConv, 
    GPSConv, 
    global_mean_pool,
    global_max_pool,
    GraphNorm,
    LayerNorm
)


class GraphEncoder(nn.Module):
    """
    Optimized Graph Neural Network for Continuous Differential Shock Graphs.
    
    Architectural Foundations:
      1. Continuous Feature Embeddings:
         - Gilmer et al., "Neural Message Passing for Quantum Chemistry", ICML 2017.
         - Hu et al., "Strategies for Pre-training Graph Neural Networks", ICLR 2020.
      2. Identity Residual Skip Connections:
         - Li et al., "DeepGCNs: Can GCNs Go as Deep as CNNs?", ICCV 2019.
         - Luo et al., "Can Classic GNNs Be Strong Baselines for Graph-level Tasks?", ICML 2025.
      3. Graph-Adaptive Normalization:
         - Cai et al., "GraphNorm: A Principled Approach to Accelerating GNN Training", ICML 2021.
         - Ba et al., "Layer Normalization", 2016.
      4. Dual-Pooling Global Readout [Mean || Max]:
         - Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019 (GIN).
         - Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017.
      5. Edge-Conditioned Convolutions:
         - GINE: Hu et al., ICLR 2020.
         - GATv2: Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022.
         - ResGated: Bresson & Laurent, "Residual Gated Graph ConvNets", 2017.
         - GraphGPS: Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022.
    """
    def __init__(
        self, 
        node_feat_dim: int = 11, 
        edge_feat_dim: int = 14, 
        hidden_dim: int = 128, 
        proj_feat_dim: int = 128, 
        gnn_type: str = 'GINE', 
        num_layers: int = 3, 
        dropout: float = 0.1, 
        norm_type: str = 'graph',    # Options: 'graph' (Cai et al., 2021), 'layer', or 'batch'
        use_dual_pool: bool = True   # Options: True for [Mean || Max] (Xu et al., 2019), False for Mean only
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type.lower()
        self.use_dual_pool = use_dual_pool

        # -------------------------------------------------------------------
        # 1. Feature Pre-Encoders (Gilmer et al., ICML 2017; Hu et al., 2020)
        # Projects raw 11D continuous nodes and 14D continuous edge descriptors 
        # into a shared, well-conditioned latent space.
        # -------------------------------------------------------------------
        self.node_emb = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # -------------------------------------------------------------------
        # 2. Message-Passing & Normalization Layers
        # -------------------------------------------------------------------
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # A. Message Passing Operator
            if gnn_type == 'GINE':
                # GINEConv (Hu et al., ICLR 2020): Injects edge attributes into a 2-layer MLP
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.layers.append(GINEConv(mlp, edge_dim=hidden_dim))

            elif gnn_type == 'GATv2':
                # GATv2Conv (Brody et al., ICLR 2022): Dynamic edge-conditioned attention
                self.layers.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False))

            elif gnn_type == 'ResGated':
                # ResGatedGraphConv (Bresson & Laurent, 2017; Dwivedi et al., 2020)
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))

            elif gnn_type == 'GPS':
                # GraphGPS (Rampasek et al., NeurIPS 2022)
                local_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                local_conv = GINEConv(local_mlp, edge_dim=hidden_dim)
                self.layers.append(GPSConv(hidden_dim, local_conv, heads=4, dropout=dropout, attn_dropout=dropout))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

            # B. Normalization Layer (Cai et al., ICML 2021; Ba et al., 2016)
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == 'layer':
                self.norms.append(LayerNorm(hidden_dim))
            elif self.norm_type == 'graph':
                self.norms.append(GraphNorm(hidden_dim))
            else:
                raise ValueError(f"Unsupported normalization: {norm_type}")

        # -------------------------------------------------------------------
        # 3. Readout & Final Projection (Xu et al., ICLR 2019)
        # Dual Pooling [Mean || Max] captures both global volume and sharp features.
        # -------------------------------------------------------------------
        pool_mult = 2 if self.use_dual_pool else 1
        readout_dim = hidden_dim * pool_mult
        
        self.proj = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_feat_dim)
        )

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # 1. Project input continuous features
        x = self.node_emb(x)
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        # 2. Message Passing with Residuals & Normalization (Li et al., 2019; Luo et al., 2025)
        for i in range(self.num_layers):
            x_in = x
            
            if self.gnn_type in ['GINE', 'GATv2', 'ResGated']:
                x = self.layers[i](x, edge_index, edge_attr=edge_attr)
            elif self.gnn_type == 'GPS':
                x = self.layers[i](x, edge_index, batch_idx, edge_attr=edge_attr)
                
            # Normalization (handles GraphNorm/LayerNorm signature with batch_idx)
            if self.norm_type in ['layer', 'graph']:
                x = self.norms[i](x, batch_idx)
            else:
                x = self.norms[i](x)
                
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Additive Residual Skip Connection
            x = x + x_in

        # 3. Global Graph Readout (Xu et al., ICLR 2019)
        if self.use_dual_pool:
            pooled = torch.cat([global_mean_pool(x, batch_idx), global_max_pool(x, batch_idx)], dim=-1)
        else:
            pooled = global_mean_pool(x, batch_idx)

        # 4. Final Projection to target multimodal dimension (128-dim)
        return self.proj(pooled)
