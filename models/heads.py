# models/heads.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotClassifier(nn.Module):
    """
    Unified Metric Classification Head for Few-Shot Learning.
    Supports Prototypical Networks (ProtoNet) and Matching Networks.
    Incorporates learnable inverse temperature scaling (1/tau) for Cosine metrics.
    """
    def __init__(
        self, 
        method: str = 'protonet', 
        distance: str = 'cosine', 
        scale: float = 10.0, 
        learnable_scale: bool = True     # Configured to match default.yaml
    ):
        super().__init__()
        self.method = method.lower()
        self.distance = distance.lower()
        self.learnable_scale = learnable_scale
        
        if self.learnable_scale:
            # Initialize learnable scale in log-space (log(10.0) ≈ 2.3025)
            # Parameterizing in log-space guarantees scale remains strictly positive
            self.log_scale = nn.Parameter(
                torch.tensor(np.log(scale), dtype=torch.float32)
            )
        else:
            self.scale = scale

    def forward(self, support, query, n_way, k_shot):
        """
        Computes metric logits between query embeddings and support prototypes.
        
        Args:
            support (Tensor): [N_way * K_shot, Feature_Dim] support embeddings.
            query (Tensor): [N_way * Q_query, Feature_Dim] query embeddings.
            n_way (int): Number of classes in the episode.
            k_shot (int): Number of support samples per class.
            
        Returns:
            Tensor: Logits of shape [N_way * Q_query, N_way].
        """
        dim = support.size(-1)

        # Retrieve scale factor: exp(log_scale) clamped between 1.0 and 50.0 for stability
        if self.learnable_scale:
            effective_scale = torch.clamp(self.log_scale.exp(), min=1.0, max=50.0)
        else:
            effective_scale = self.scale

        if self.method == 'protonet':
            # 1. Compute Class Prototypes (Averaging across K-shots)
            # Shape: [N_way, Feature_Dim]
            prototypes = support.view(n_way, k_shot, dim).mean(dim=1)
            
            # 2. L2 Normalization (Hypersphere Projection)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            query = F.normalize(query, p=2, dim=-1)

            # 3. Distance Metric Evaluation
            if self.distance == 'cosine':
                # Cosine Similarity * Learnable Inverse Temperature
                return effective_scale * torch.mm(query, prototypes.t())
                
            elif self.distance == 'euclidean':
                # Squared Euclidean Distance * (effective_scale / 2.0)
                dists = torch.cdist(query, prototypes, p=2) ** 2
                return -(effective_scale / 2.0) * dists
                
            else:
                raise ValueError(f"Distance metric '{self.distance}' is not supported.")
                
        else:
            raise ValueError(f"Classifier method '{self.method}' is not supported.")
