import torch
import torch.nn as nn

# Ensure these import paths match your exact file structure
from .cnn_encoder import VisionEncoder
from .gnn_encoder import GraphEncoder
from .fusion import MultimodalFusion
from .heads import FewShotClassifier

class MultimodalFewShotNetwork(nn.Module):
    """
    Top-level network for Multimodal Few-Shot Learning.
    Routes data through unimodal encoders, fuses them if required, 
    and passes the resulting features to the Prototypical Network head.
    """
    def __init__(self, cfg):
        super().__init__()
        self.modality = cfg.model.modality
        
        # --- 1. Vision Pathway ---
        if self.modality in ['vision', 'multimodal']:
            self.vision_encoder = VisionEncoder(proj_feat_dim=cfg.model.hidden_dim)
            
        # --- 2. Graph Pathway ---
        if self.modality in ['graph', 'multimodal']:
            self.graph_encoder = GraphEncoder(
                node_feat_dim=cfg.model.node_feat_dim,
                edge_feat_dim=cfg.model.edge_feat_dim,
                hidden_dim=cfg.model.hidden_dim,
                proj_feat_dim=cfg.model.hidden_dim,
                gnn_type=cfg.model.gnn_type,
                num_layers=cfg.model.num_layers,
                dropout=cfg.model.dropout
            )
            
        # --- 3. Fusion Block ---
        if self.modality == 'multimodal':
            # FIX: Unpack the config explicitly
            self.fusion = MultimodalFusion(
                proj_feat_dim=cfg.model.hidden_dim, 
                fusion_type=cfg.model.fusion_type
            )
            
        # --- 4. Few-Shot Head ---
        self.classifier = FewShotClassifier(
            method=cfg.model.fsl_method,
            distance=cfg.model.distance_metric,
            use_simpleshot=cfg.model.use_simpleshot
        )

    def forward(self, vision_batch, graph_batch, n_way, k_shot):
        """
        Forward pass for an entire FSL episode.
        """
        # --- A. Feature Extraction & Modality Routing ---
        if self.modality == 'vision':
            features = self.vision_encoder(vision_batch)
            
        elif self.modality == 'graph':
            features = self.graph_encoder(graph_batch)
            
        elif self.modality == 'multimodal':
            v_feat = self.vision_encoder(vision_batch)
            g_feat = self.graph_encoder(graph_batch)
            features = self.fusion(v_feat, g_feat)
            
        else:
            raise ValueError(f"Unknown modality configured: {self.modality}")
            
        # --- B. Episodic Splitting ---
        # Because the EpisodicBatchSampler perfectly ordered the batch as 
        # [Support_Class1... Support_ClassN, Query_Class1... Query_ClassN],
        # we can safely slice the tensor purely by index.
        k_total = n_way * k_shot
        support_features = features[:k_total]
        query_features = features[k_total:]
        
        # --- C. Prototypical Classification ---
        # The FewShotClassifier handles the .view() reshaping, prototype 
        # centroid calculation, and distance metric computation.
        logits = self.classifier(support_features, query_features, n_way, k_shot)
        
        return logits
`
