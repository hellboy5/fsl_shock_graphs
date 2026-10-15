# losses/gmn_loss.py
"""Loss functions for Graph Matching Network (GMN) Episodic Meta-Training.

Implements Episodic Cross-Entropy (Multi-Class InfoNCE) with an optional
Batch-Hard Triplet Margin regularizer for metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMNEpisodicLoss(nn.Module):
  """Episodic Cross-Entropy Loss with optional Batch-Hard Negative Mining.

  Computes Softmax NLL over the negative scaled distance matrix:
      logits = -dist_matrix / tau
  """

  def __init__(
      self,
      tau: float = 0.5,
      margin: float = 0.2,
      use_hard_mining: bool = False,
      lambda_triplet: float = 0.1,
  ):
    super().__init__()
    self.tau = tau
    self.margin = margin
    self.use_hard_mining = use_hard_mining
    self.lambda_triplet = lambda_triplet

  def forward(
      self, dist_or_logits: torch.Tensor, target_labels: torch.Tensor, is_distance: bool = True
  ) -> torch.Tensor:
    """Args:
        dist_or_logits: [N_query, N_way] -> Pairwise distance matrix or pre-scaled logits.
        target_labels: [N_query] -> Ground-truth class index (0 to N_way - 1).
        is_distance: bool -> If True, computes logits = -dist / tau. If False, treats as logits.

    Returns:
        Scalar loss tensor.
    """
    if is_distance:
      logits = -dist_or_logits / self.tau
      dist_matrix = dist_or_logits
    else:
      logits = dist_or_logits
      dist_matrix = -logits * self.tau

    # 1. Primary Objective: Multi-Class Episodic Cross-Entropy
    loss_ce = F.cross_entropy(logits, target_labels)

    if not self.use_hard_mining:
      return loss_ce

    # 2. Optional Regularizer: Batch-Hard Negative Triplet Margin
    N_q, N_way = dist_matrix.shape
    device = dist_matrix.device

    # Positive distance for each query sample
    d_pos = dist_matrix[torch.arange(N_q, device=device), target_labels]

    # Mask positive class to isolate all negative classes in the episode
    mask_neg = torch.ones_like(dist_matrix, dtype=torch.bool)
    mask_neg[torch.arange(N_q, device=device), target_labels] = False

    # Hardest negative: minimum distance among non-matching classes
    d_neg_hard = dist_matrix.masked_fill(~mask_neg, 1e9).min(dim=-1).values

    # Batch-hard margin penalty
    loss_triplet = F.relu(d_pos - d_neg_hard + self.margin).mean()

    return loss_ce + self.lambda_triplet * loss_triplet
