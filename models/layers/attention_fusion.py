# models/layers/attention_fusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper: Format Tensor to Token Sequence [B, L, D]
# =============================================================================
def _ensure_token_sequence(x):
  """Ensures tensor has shape [Batch_Size, Seq_Len, Dim].

  If x is [B, D], it is unsqueezed to [B, 1, D]. If x is [B, D, H, W], it is
  flattened to [B, H*W, D].
  """
  if x.dim() == 2:
    return x.unsqueeze(1)  # [B, 1, D]
  elif x.dim() == 4:
    B, D, H, W = x.shape
    return x.view(B, D, H * W).transpose(1, 2)  # [B, H*W, D]
  elif x.dim() == 3:
    return x  # [B, L, D]
  else:
    raise ValueError(f"Unsupported input tensor dimension: {x.shape}")


# =============================================================================
# 1. Dense Bidirectional Cross-Attention
# =============================================================================
class DenseCrossAttention(nn.Module):
  """Bidirectional token-to-token cross-attention between Vision and Graph.

  Vision attends to Graph, Graph attends to Vision, followed by residual
  projection.
  """

  def __init__(self, dim=640, num_heads=8, dropout=0.1):
    super().__init__()
    self.dim = dim
    self.v_to_g = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )
    self.g_to_v = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )

    self.ln_v1 = nn.LayerNorm(dim)
    self.ln_g1 = nn.LayerNorm(dim)

    # Cross-modal fusion projection
    self.proj = nn.Sequential(
        nn.Linear(dim * 2, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
    )
    self.ln_out = nn.LayerNorm(dim)

  def forward(self, z_v, z_g):
    t_v = _ensure_token_sequence(z_v)  # [B, L_v, 640]
    t_g = _ensure_token_sequence(z_g)  # [B, L_g, 640]

    # Cross-Attention passes
    attn_v, _ = self.v_to_g(query=t_v, key=t_g, value=t_g)
    attn_g, _ = self.g_to_v(query=t_g, key=t_v, value=t_v)

    out_v = self.ln_v1(t_v + attn_v).mean(dim=1)  # [B, 640]
    out_g = self.ln_g1(t_g + attn_g).mean(dim=1)  # [B, 640]

    # Fused representation
    fused = self.proj(torch.cat([out_v, out_g], dim=-1))
    z_out = self.ln_out(fused + (z_v if z_v.dim() == 2 else z_v.mean(dim=1)))
    return F.normalize(z_out, p=2, dim=-1)


# =============================================================================
# 2. Multimodal Bottleneck Transformer (MBTB - Top Recommendation)
# =============================================================================
class MultimodalBottleneckAttention(nn.Module):
  """Multimodal Bottleneck Transformer (Nagrani et al., NeurIPS 2021).

  Uses B=4 learnable latent tokens to constrain cross-modal communication,
  preventing base-class overfitting and noise propagation.
  """

  def __init__(self, dim=640, num_bottlenecks=4, num_heads=8, dropout=0.1):
    super().__init__()
    self.num_bottlenecks = num_bottlenecks
    self.dim = dim

    # Learnable latent bottleneck tokens shared across the batch
    self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, dim))
    nn.init.trunc_normal_(self.bottlenecks, std=0.02)

    # Modalities write to bottleneck
    self.v_to_b = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )
    self.g_to_b = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )

    # Modalities read from updated bottleneck
    self.b_to_v = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )
    self.b_to_g = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )

    self.ln_b = nn.LayerNorm(dim)
    self.ln_v = nn.LayerNorm(dim)
    self.ln_g = nn.LayerNorm(dim)
    self.ln_out = nn.LayerNorm(dim)

    # Learnable gating scalar
    self.alpha = nn.Parameter(torch.tensor(0.1))

  def forward(self, z_v, z_g):
    B = z_v.size(0)
    t_v = _ensure_token_sequence(z_v)  # [B, L_v, 640]
    t_g = _ensure_token_sequence(z_g)  # [B, L_g, 640]

    b = self.bottlenecks.to(z_v.device).expand(B, -1, -1)

    # Step 1: Modalities deposit context into the shared bottleneck
    b_from_v, _ = self.v_to_b(query=b, key=t_v, value=t_v)
    b_from_g, _ = self.g_to_b(query=b, key=t_g, value=t_g)
    b_updated = self.ln_b(b + b_from_v + b_from_g)

    # Step 2: Modalities read consensus back from the bottleneck
    t_v_up, _ = self.b_to_v(query=t_v, key=b_updated, value=b_updated)
    t_g_up, _ = self.b_to_g(query=t_g, key=b_updated, value=b_updated)

    v_pool = self.ln_v(t_v + t_v_up).mean(dim=1)  # [B, 640]
    g_pool = self.ln_g(t_g + t_g_up).mean(dim=1)  # [B, 640]

    # Step 3: Residual fusion anchored on Vision
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    z_fused = self.ln_out(base_v + self.alpha * (v_pool + g_pool))
    return F.normalize(z_fused, p=2, dim=-1)


# =============================================================================
# 3. Low-Rank Attention (Rank Attention)
# =============================================================================
class LowRankAttention(nn.Module):
  """Low-Rank Factorized Cross-Attention.

  Projects 640D features into an R-dimensional subspace (R=32) before attention,
  filtering out high-frequency noise and pebble artifacts.
  """

  def __init__(self, dim=640, rank=32, num_heads=4, dropout=0.1):
    super().__init__()
    self.dim = dim
    self.rank = rank

    # Low-rank factorized down-projectors
    self.down_v = nn.Linear(dim, rank, bias=False)
    self.down_g = nn.Linear(dim, rank, bias=False)

    # Subspace Cross-Attention
    self.subspace_attn = nn.MultiheadAttention(
        embed_dim=rank, num_heads=num_heads, dropout=dropout, batch_first=True
    )

    # Up-projector back to 640D
    self.up_proj = nn.Sequential(
        nn.Linear(rank, dim),
        nn.GELU(),
        nn.LayerNorm(dim),
    )

    self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
    self.ln_out = nn.LayerNorm(dim)

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    base_g = z_g if z_g.dim() == 2 else z_g.mean(dim=1)

    t_v = _ensure_token_sequence(z_v)  # [B, L_v, 640]
    t_g = _ensure_token_sequence(z_g)  # [B, L_g, 640]

    # Project into rank-R subspace [B, L, 32]
    r_v = self.down_v(t_v)
    r_g = self.down_g(t_g)

    # Attend in low-rank space
    r_fused, _ = self.subspace_attn(query=r_v, key=r_g, value=r_g)
    delta_z = self.up_proj(r_fused.mean(dim=1))  # [B, 640]

    # Contextual gate
    g = self.gate(torch.cat([base_v, delta_z], dim=-1))
    z_out = self.ln_out(base_v + g * delta_z)
    return F.normalize(z_out, p=2, dim=-1)


# =============================================================================
# 4. Asymmetric Residual Cross-Attention (Vision Anchor)
# =============================================================================
class AsymmetricResidualCrossAttention(nn.Module):
  """Vision-anchored residual attention.

  ResNet-12 acts as Query vector, querying GINE node/subgraph tokens. Includes a
  sigmoid-confidence gate and learnable scale alpha.
  """

  def __init__(self, dim=640, num_heads=8, dropout=0.1):
    super().__init__()
    self.dim = dim
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )

    self.gate_mlp = nn.Sequential(
        nn.Linear(dim * 2, dim // 2),
        nn.ReLU(),
        nn.Linear(dim // 2, dim),
        nn.Sigmoid(),
    )

    self.ln_delta = nn.LayerNorm(dim)
    self.ln_out = nn.LayerNorm(dim)

    # Initialized to zero so training starts exactly at Vision Baseline (59.71%)
    self.alpha = nn.Parameter(torch.zeros(1))

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    q_v = base_v.unsqueeze(1)  # [B, 1, 640]
    t_g = _ensure_token_sequence(z_g)  # [B, L_g, 640]

    # Query graph tokens using Vision representation
    attn_out, _ = self.cross_attn(query=q_v, key=t_g, value=t_g)
    delta_g = self.ln_delta(attn_out.squeeze(1))  # [B, 640]

    # Dynamic element-wise gate deciding how much shape context to inject
    gate = self.gate_mlp(torch.cat([base_v, delta_g], dim=-1))

    z_fused = self.ln_out(base_v + self.alpha * (gate * delta_g))
    return F.normalize(z_fused, p=2, dim=-1)


# =============================================================================
# 5. Gated Cross-Attention (Flamingo-style Tanh Gating)
# =============================================================================
class GatedCrossAttention(nn.Module):
  """Flamingo-style gated cross-attention with zero-initialized tanh gates."""

  def __init__(self, dim=640, num_heads=8, dropout=0.1):
    super().__init__()
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
    )
    self.ln_q = nn.LayerNorm(dim)
    self.ln_kv = nn.LayerNorm(dim)

    # Tanh gate initialized at 0.0
    self.tanh_gate = nn.Parameter(torch.zeros(1))
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 2, dim),
    )
    self.ln_ffn = nn.LayerNorm(dim)
    self.ffn_gate = nn.Parameter(torch.zeros(1))

  def forward(self, z_v, z_g):
    base_v = z_v if z_v.dim() == 2 else z_v.mean(dim=1)
    q = self.ln_q(_ensure_token_sequence(z_v))
    kv = self.ln_kv(_ensure_token_sequence(z_g))

    attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
    attn_pool = attn_out.mean(dim=1)

    # Residual with tanh gating
    h = base_v + torch.tanh(self.tanh_gate) * attn_pool
    out = h + torch.tanh(self.ffn_gate) * self.ffn(self.ln_ffn(h))
    return F.normalize(out, p=2, dim=-1)


# =============================================================================
# Factory Dispatcher
# =============================================================================
def build_attention_fusion(fusion_type: str, dim: int = 640, **kwargs):
  """Builds the requested attention fusion module."""
  f_type = fusion_type.lower()
  if f_type in ["bottleneck", "mbtb"]:
    return MultimodalBottleneckAttention(dim=dim, **kwargs)
  elif f_type in ["asymmetric", "asym"]:
    return AsymmetricResidualCrossAttention(dim=dim, **kwargs)
  elif f_type in ["low_rank", "rank"]:
    return LowRankAttention(dim=dim, **kwargs)
  elif f_type in ["cross_attention", "cross_attn", "dense"]:
    return DenseCrossAttention(dim=dim, **kwargs)
  elif f_type in ["gated_cross", "flamingo"]:
    return GatedCrossAttention(dim=dim, **kwargs)
  else:
    raise ValueError(f"Unknown attention fusion_type: {fusion_type}")
