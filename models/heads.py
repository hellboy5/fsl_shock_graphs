# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotClassifier(nn.Module):
    """
    Benchmark-Compliant Few-Shot Classification Head with L2 Normalization.
    
    Academic Foundations:
      1. Prototypical Centroids:
         - Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
      2. Unit-Hypersphere Normalization (SimpleShot):
         - Wang et al., "SimpleShot: Revisiting Nearest-Neighbor Classification 
           for Few-Shot Learning", arXiv:1911.04623.
      3. Temperature-Scaled Cosine Classification:
         - Chen et al., "A Closer Look at Few-Shot Classification", ICLR 2019.
         - DeepEMD: Zhang et al., CVPR 2020.
         - FRN: Wertheimer et al., CVPR 2021.
    """
    def __init__(
        self, 
        method: str = 'protonet', 
        distance: str = 'cosine', 
        scale: float = 10.0, 
        use_simpleshot: bool = False
    ):
        """
        Args:
            method: 'protonet' (nearest centroid) or 'matching' (pairwise comparison)
            distance: 'cosine' (recommended) or 'euclidean' (normalized)
            scale: Inverse temperature scaling factor (standard 10.0 in FSL)
            use_simpleshot: Optional task-level mean subtraction before normalization
        """
        super().__init__()
        self.method = method.lower()
        self.distance = distance.lower()
        self.scale = scale
        self.use_simpleshot = use_simpleshot

    def forward(
        self, 
        support: torch.Tensor, 
        query: torch.Tensor, 
        n_way: int, 
        k_shot: int
    ) -> torch.Tensor:
        """
        Args:
            support: [n_way * k_shot, dim]
            query:   [n_query_total, dim]
            n_way:   Number of classes in the episode
            k_shot:  Number of support shots per class
            
        Returns:
            logits:  [n_query_total, n_way]
        """
        # 1. Optional Task-Level Centering (SimpleShot / Wang et al., 2019)
        if self.use_simpleshot:
            support_mean = support.mean(dim=0, keepdim=True)
            support = support - support_mean
            query = query - support_mean

        # 2. Mandatory L2 Feature Normalization
        # Projects all support and query vectors onto the unit hypersphere: ||z||_2 = 1.0
        # Prevents initial loss spikes and eliminates magnitude bias.
        support = F.normalize(support, p=2, dim=-1)
        query = F.normalize(query, p=2, dim=-1)

        dim = support.size(-1)

        # 3. Metric Evaluation
        if self.method == 'protonet':
            # Class Centroid c_k as the average of support vectors
            prototypes = support.view(n_way, k_shot, dim).mean(dim=1)
            # Re-normalize class prototype onto unit hypersphere (SimpleShot standard)
            prototypes = F.normalize(prototypes, p=2, dim=-1)

            if self.distance == 'cosine':
                # Scaled Cosine Similarity (DeepEMD / Chen et al.)
                logits = self.scale * torch.mm(query, prototypes.t())
            elif self.distance == 'euclidean':
                # Normalized Euclidean Distance: ||u - v||^2 in range
                dists = torch.cdist(query, prototypes, p=2) ** 2
                logits = -(self.scale / 2.0) * dists
            else:
                raise ValueError(f"Unknown distance metric: {self.distance}")

        elif self.method == 'matching':
            # Matching Network: Pairwise nearest-neighbor evaluation
            if self.distance == 'cosine':
                sims = self.scale * torch.mm(query, support.t())
            elif self.distance == 'euclidean':
                dists = torch.cdist(query, support, p=2) ** 2
                sims = -(self.scale / 2.0) * dists
            else:
                raise ValueError(f"Unknown distance metric: {self.distance}")

            # Average similarities across support exemplars per class
            sims = sims.view(query.size(0), n_way, k_shot)
            logits = sims.mean(dim=2)

        else:
            raise ValueError(f"Unknown FSL method: {self.method}")

        return logits
