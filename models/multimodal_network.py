import torch
import torch.nn as nn
from models.encoders.cnn_encoder import VisionEncoder
from models.encoders.gnn_encoder import GraphEncoder
from models.fusion import MultimodalFusion
from models.prototypical import FewShotClassifier

class MultimodalFewShotNetwork(nn.Module):
    """
    The end-to-end architecture unifying vision extraction, topological 
    graph extraction, multimodal fusion, and metric-based classification.
    """
    def __init__(self, cfg):
        super().__init__()
        self.modality = cfg.model.modality # 'vision', 'graph', or 'fusion'
        
        # 1. Feature Extractors (The Engines)
        if self.modality in ['vision', 'fusion']:
            self.vision_encoder = VisionEncoder(proj_feat_dim=cfg.model.hidden_dim)
            
        if self.modality in ['graph', 'fusion']:
            self.graph_encoder = GraphEncoder(
                node_feat_dim=cfg.model.node_dim,
                edge_feat_dim=cfg.model.edge_dim,
                hidden_dim=cfg.model.hidden_dim,
                proj_feat_dim=cfg.model.hidden_dim,
                gnn_type=cfg.model.gnn_type
            )
            
        # 2. Multimodal Fusion Module
        if self.modality == 'fusion':
            self.fusion = MultimodalFusion(
                proj_feat_dim=cfg.model.hidden_dim, 
                fusion_type=cfg.model.fusion_type
            )
            
        # 3. Metric Learning Head (The Classifier)
        self.classifier = FewShotClassifier(
            method=cfg.task.fsl_method, 
            distance=cfg.task.distance_metric,
            use_simpleshot=cfg.task.use_simpleshot
        )

    def forward(self, vision_batch, graph_batch, n_way, k_shot):
        # Step 1: Modality Routing & Extraction
        if self.modality == 'vision':
            features = self.vision_encoder(vision_batch)
        elif self.modality == 'graph':
            features = self.graph_encoder(graph_batch)
        elif self.modality == 'fusion':
            v_feat = self.vision_encoder(vision_batch)
            g_feat = self.graph_encoder(graph_batch)
            features = self.fusion(v_feat, g_feat)

        # Step 2: Task Formulation (Splitting the batch into Support & Query)
        k_total = n_way * k_shot
        support = features[:k_total]
        query = features[k_total:]

        # Step 3: Metric Classification
        logits = self.classifier(support, query, n_way, k_shot)
        
        return logits
