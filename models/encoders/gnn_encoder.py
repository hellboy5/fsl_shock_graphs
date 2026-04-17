import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, 
    GATv2Conv, 
    ResGatedGraphConv, 
    GPSConv, 
    global_mean_pool
)

class GraphEncoder(nn.Module):
    def __init__(self, node_feat_dim=11, edge_feat_dim=14, hidden_dim=128, proj_feat_dim=128, gnn_type='GINE', num_layers=3, dropout=0.1):
        """
        Args:
            node_feat_dim: 11 (x, y, t, flow_slots, struct_slots)
            edge_feat_dim: 14 (geometric continuous features)
            hidden_dim: Internal dimension for message passing
            proj_feat_dim: Final dimension to match the CNN (128)
            gnn_type: 'GINE', 'GATv2', 'ResGated', or 'GPS'
            num_layers: Number of message passing layers
            dropout: Dropout probability
        """
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. Initial Linear Embedders
        # Projects raw nodes and edges up to the hidden_dim immediately.
        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_feat_dim, hidden_dim)

        # 2. Build the GNN Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'GINE':
                # GINE uses an MLP to update nodes, adding edge attributes natively
                nn_callable = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.layers.append(GINEConv(nn_callable, edge_dim=hidden_dim))

            elif gnn_type == 'GATv2':
                # GATv2 pays dynamic attention to neighbors, modulated by edge geometry
                self.layers.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False))

            elif gnn_type == 'ResGated':
                # ResGated explicitly gates information flow based on edge geometry
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))

            elif gnn_type == 'GPS':
                # GraphGPS requires a "local" MPNN. We use GINE as the local feature extractor.
                local_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                local_mpnn = GINEConv(local_nn, edge_dim=hidden_dim)
                # GPSConv combines the local MPNN with a global multi-head attention Transformer
                self.layers.append(GPSConv(hidden_dim, local_mpnn, heads=4, dropout=dropout, attn_dropout=dropout))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

        # 3. Projection Head
        # Projects the pooled graph vector to match the Multimodal Fusion requirement
        self.proj1 = nn.Linear(hidden_dim, proj_feat_dim)

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # 1. Embed raw features into latent space
        x = self.node_emb(x)
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
            
        # 2. Message Passing with Residual Connections
        for i in range(self.num_layers):
            # Save the input state for the residual connection
            x_residual = x  
            
            # Pass through the specific layer
            if self.gnn_type in ['GINE', 'GATv2', 'ResGated']:
                x = self.layers[i](x, edge_index, edge_attr=edge_attr)
            elif self.gnn_type == 'GPS':
                # GPS handles batching internally for its global attention
                x = self.layers[i](x, edge_index, batch_idx, edge_attr=edge_attr)
                
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply the residual connection (GNN+ Trick #5)
            # GPS actually applies its own internal residuals, but adding it here 
            # for the classic architectures guarantees stability.
            x = x + x_residual

        # 3. Global Graph Pooling (Compress graph into a single vector)
        pooled_graph = global_mean_pool(x, batch_idx)

        # 4. Final Projection (to 128-dim for FSL matching/fusion)
        projected_graph = self.proj1(pooled_graph)

        return projected_graph
