import torch
import torch.nn as nn
import torch.nn.functional as F

class FewShotClassifier(nn.Module):
    def __init__(self, method='protonet', distance='euclidean', use_simpleshot=False):
        """
        Args:
            method: 'protonet' (nearest centroid) or 'matching' (nearest neighbor)
            distance: 'euclidean' or 'cosine'
            use_simpleshot: Applies CL2N (Centered L2-Normalization) from arXiv:1911.04623
        """
        super().__init__()
        self.method = method
        self.distance = distance
        self.use_simpleshot = use_simpleshot

    def forward(self, support, query, n_way, k_shot):
        """
        Args:
            support: [n_way * k_shot, dim]
            query: [n_query_total, dim]
        """
        # ---------------------------------------------------------
        # 1. SimpleShot Feature Transformation (CL2N)
        # ---------------------------------------------------------
        if self.use_simpleshot:
            # Centering: Subtract the mean of the support set (Task-level centering)
            # (Note: For exact SimpleShot, you can pass in the global train-set mean here)
            support_mean = support.mean(dim=0, keepdim=True)
            support = support - support_mean
            query = query - support_mean

            # L2-Normalization
            support = F.normalize(support, p=2, dim=-1)
            query = F.normalize(query, p=2, dim=-1)

        # ---------------------------------------------------------
        # 2. Few-Shot Evaluation
        # ---------------------------------------------------------
        if self.method == 'protonet':
            # Prototype Approach (Nearest Centroid)
            prototypes = support.view(n_way, k_shot, -1).mean(1) # [n_way, dim]
            
            if self.distance == 'euclidean':
                logits = -torch.cdist(query, prototypes) ** 2
            elif self.distance == 'cosine':
                logits = self.cosine_sim(query, prototypes)
                
        elif self.method == 'matching':
            # Matching Network Approach (Nearest Neighbor Pairwise)
            if self.distance == 'euclidean':
                sims = -torch.cdist(query, support) ** 2
            elif self.distance == 'cosine':
                sims = self.cosine_sim(query, support) 
            
            # Aggregate pairwise similarities by class
            sims = sims.view(query.size(0), n_way, k_shot)
            logits = sims.mean(2)

        return logits

    def cosine_sim(self, x, y):
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        return torch.mm(x_norm, y_norm.t())
