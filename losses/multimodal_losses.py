# losses/multimodal_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalInfoNCELoss(nn.Module):
  """Symmetric in-batch contrastive loss between image and graph embeddings (CLIP style)."""

  def __init__(self, temperature=0.07):
    super().__init__()
    self.temp = nn.Parameter(torch.tensor(temperature))

  def forward(self, z_v, z_g):
    # z_v, z_g are expected to be L2-normalized: [B, 640]
    t = self.temp.clamp(min=0.01, max=1.0)
    sim = torch.mm(z_v, z_g.t()) / t  # [B, B]
    labels = torch.arange(z_v.size(0), device=z_v.device)
    loss_v2g = F.cross_entropy(sim, labels)
    loss_g2v = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_v2g + loss_g2v)


class SupConLoss(nn.Module):
  """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

  Pulls features of the same class together across modalities.
  """

  def __init__(self, temperature=0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, features, labels):
    # features: [2*B, D], labels: [B]
    device = features.device
    batch_size = labels.size(0)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

    # Tile mask for 2 views (vision + graph): [2*B, 2*B]
    mask = mask.repeat(2, 2)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * 2, device=device).view(-1, 1),
        0,
    )
    mask = mask * logits_mask

    sim = torch.div(torch.matmul(features, features.T), self.temperature)
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss


class MultimodalLossEngine(nn.Module):
  """Master loss dispatcher for joint multimodal pre-training."""

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.loss_type = getattr(cfg.training, "loss_type", "ce_multitask").lower()
    self.lambda_aux = float(getattr(cfg.training, "lambda_aux", 0.5))
    self.lambda_contrast = float(getattr(cfg.training, "lambda_contrast", 0.1))

    smoothing = float(getattr(cfg.training, "label_smoothing", 0.0))
    self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)

    if self.loss_type in ["clip_ce", "infonce"]:
      self.contrastive = CrossModalInfoNCELoss(temperature=0.07)
    elif self.loss_type in ["supcon", "supcon_ce"]:
      self.supcon = SupConLoss(temperature=0.07)

  def forward(
      self, logits_fuse, logits_v, logits_g, z_v_norm, z_g_norm, labels
  ):
    loss_fuse = self.ce(logits_fuse, labels)
    loss_v = self.ce(logits_v, labels)
    loss_g = self.ce(logits_g, labels)

    # 1. Multi-Task Cross-Entropy
    if self.loss_type == "ce_multitask":
      total_loss = (
          loss_fuse + self.lambda_aux * loss_v + self.lambda_aux * loss_g
      )

    # 2. CLIP / InfoNCE Cross-Modal Contrastive
    elif self.loss_type in ["clip_ce", "infonce"]:
      loss_clip = self.contrastive(z_v_norm, z_g_norm)
      total_loss = (
          loss_fuse
          + self.lambda_contrast * loss_clip
          + self.lambda_aux * (loss_v + loss_g)
      )

    # 3. Supervised Contrastive (SupCon)
    elif self.loss_type in ["supcon", "supcon_ce"]:
      features_all = torch.cat([z_v_norm, z_g_norm], dim=0)  # [2*B, 640]
      loss_sc = self.supcon(features_all, labels)
      total_loss = (
          loss_fuse
          + self.lambda_contrast * loss_sc
          + self.lambda_aux * (loss_v + loss_g)
      )

    # 4. Pairwise Cosine Alignment
    elif self.loss_type in ["cosine_ce", "align"]:
      loss_align = (
          1.0 - F.cosine_similarity(z_v_norm, z_g_norm, dim=-1)
      ).mean()
      total_loss = (
          loss_fuse
          + self.lambda_contrast * loss_align
          + self.lambda_aux * (loss_v + loss_g)
      )

    else:
      raise ValueError(f"Unknown loss_type: {self.loss_type}")

    return total_loss, loss_fuse, loss_v, loss_g
