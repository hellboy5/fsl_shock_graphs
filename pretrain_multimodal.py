# pretrain_multimodal.py
import math
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from losses.multimodal_losses import MultimodalLossEngine
from models.multimodal_pretrain import MultimodalPretrainModel
from train import seed_worker
from utils.helpers import seed_everything


def apply_ogm_gradient_modulation(
    model: torch.nn.Module,
    loss_v: torch.Tensor,
    loss_g: torch.Tensor,
    alpha: float = 1.0,
):
  """Applies On-the-Fly Gradient Modulation to prevent modality dominance.

  If the auxiliary vision loss is lower than the auxiliary graph loss (vision
  is dominating), we scale down the vision encoder's gradients before the
  optimizer step.
  """
  if loss_v.item() < loss_g.item():
    # 1. Compute the discrepancy ratio
    ratio = (loss_g.item() - loss_v.item()) / (loss_g.item() + 1e-8)

    # 2. Compute the throttling coefficient via hyperbolic tangent decay
    # alpha controls the steepness of the dampening curve
    coeff = 1.0 - math.tanh(alpha * ratio)

    # 3. Intercept and scale down the gradients of all ResNet-12 parameters
    # This prevents the visual stream from overtaking the GNN's learning paths
    for name, p in model.vision_encoder.named_parameters():
      if p.grad is not None:
        p.grad.data.mul_(coeff)


def train_collate(data_list):
  """Collates both vision images and PyG shock graphs into a unified batch."""
  images = torch.stack([d.x_img for d in data_list])
  
  # Strip x_img to save GPU memory during graph message passing
  graphs_clean = []
  for d in data_list:
    g = d.clone()
    if hasattr(g, "x_img"):
      del g.x_img
    graphs_clean.append(g)
    
  graphs = Batch.from_data_list(graphs_clean)
  labels = torch.tensor([d.y.item() for d in data_list], dtype=torch.long)
  return {"image": images, "graph": graphs, "label": labels}


def val_episodic_collate(data_list):
  """Collates episodic evaluation batches for few-shot validation."""
  images = torch.stack([d.x_img for d in data_list])
  
  graphs_clean = []
  for d in data_list:
    g = d.clone()
    if hasattr(g, "x_img"):
      del g.x_img
    graphs_clean.append(g)
    
  graphs = Batch.from_data_list(graphs_clean)
  return {"image": images, "graph": graphs}


def run_episodic_val(
    model, val_loader, device, n_way=5, n_shot=1, n_query=15
):
  """Evaluates few-shot accuracy using unified cosine prototypes."""
  model.eval()
  targets = (
      torch.arange(n_way)
      .repeat_interleave(n_query)
      .long()
      .to(device, non_blocking=True)
  )
  accs = []
  k_total = n_way * n_shot

  with torch.no_grad():
    for batch in val_loader:
      imgs = batch["image"].to(device, non_blocking=True)
      graphs = batch["graph"].to(device, non_blocking=True)

      # 1. Extract unified 640D feature embeddings for all episode items
      z = model.extract_fused_features(imgs, graphs)  # [80, 640]

      # 2. Form class prototypes from support items and L2-normalize
      z_supp = (
          z[:k_total].view(n_way, n_shot, -1).mean(dim=1)
      )  # Class centroids [n_way, 640]
      z_supp = F.normalize(z_supp, p=2, dim=-1)

      # 3. L2-normalize query features and compute metric cosine similarities
      z_query = F.normalize(z[k_total:], p=2, dim=-1)  # Query items [n_query_total, 640]
      cos_sim = torch.mm(z_query, z_supp.t())  # [n_query_total, n_way]

      preds = cos_sim.argmax(dim=-1)
      accs.append((preds == targets).float().mean().item() * 100.0)

  mean_acc = float(np.mean(accs))
  ci_acc = float(1.96 * (np.std(accs) / math.sqrt(len(accs))))
  return mean_acc, ci_acc


def build_optimizers(model, cfg, epochs):
  """Constructs disentangled optimizers and schedulers tailored per modality."""
  opt_mode = getattr(cfg.training, "optimizer", "disentangled").lower()

  if opt_mode == "disentangled":
    # 1. Vision Stream: SGD with Nesterov Momentum (Prevents sharp minima on 2D pixels)
    v_lr = float(getattr(cfg.training, "vision_lr", 0.05))
    v_wd = float(getattr(cfg.training, "vision_weight_decay", 0.0005))
    opt_v = torch.optim.SGD(
        model.vision_encoder.parameters(),
        lr=v_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=v_wd,
    )
    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_v, T_max=epochs, eta_min=1e-5
    )

    # 2. Graph Stream: AdamW (Per-coordinate preconditioning for degree-skewed graphs)
    g_lr = float(getattr(cfg.training, "graph_lr", 0.001))
    g_wd = float(getattr(cfg.training, "graph_weight_decay", 0.0001))
    opt_g = torch.optim.AdamW(
        model.graph_encoder.parameters(), lr=g_lr, weight_decay=g_wd
    )
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=epochs, eta_min=1e-6
    )

    # 3. Fusion Block & Classification Heads: AdamW (Fast coordinate alignment)
    f_lr = float(getattr(cfg.training, "fusion_lr", 0.001))
    f_wd = float(getattr(cfg.training, "fusion_weight_decay", 0.0001))
    fusion_params = (
        list(model.classifier_fused.parameters())
        + list(model.classifier_v.parameters())
        + list(model.classifier_g.parameters())
    )
    if model.fusion is not None:
      fusion_params += list(model.fusion.parameters())

    opt_f = torch.optim.AdamW(fusion_params, lr=f_lr, weight_decay=f_wd)
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_f, T_max=epochs, eta_min=1e-6
    )

    optimizers = [opt_v, opt_g, opt_f]
    schedulers = [sched_v, sched_g, sched_f]
    print(
        f"--> [Disentangled Optimizers Active]\n"
        f"    • Vision Stream : SGD(lr={v_lr}, momentum=0.9, nesterov=True,"
        f" wd={v_wd})\n"
        f"    • Graph Stream  : AdamW(lr={g_lr}, wd={g_wd})\n"
        f"    • Fusion & Heads: AdamW(lr={f_lr}, wd={f_wd})"
    )

  else:
    # Single unified optimizer fallback if explicitly configured
    lr = float(getattr(cfg.training, "lr", 0.001))
    wd = float(getattr(cfg.training, "weight_decay", 0.0001))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )
    optimizers = [opt]
    schedulers = [sched]
    print(f"--> [Unified Optimizer Active] AdamW(lr={lr}, wd={wd})")

  return optimizers, schedulers


def run_joint_pretraining(cfg: DictConfig, device: torch.device):
  loss_type = getattr(cfg.training, "loss_type", "ce_multitask")
  fusion_type = getattr(cfg.model, "fusion_type", "asymmetric")
  epochs = int(getattr(cfg.training, "epochs", 35))

  print("=" * 80)
  print(
      f"STARTING JOINT PRE-TRAINING | Fusion: {fusion_type} | Loss: {loss_type}"
  )
  print("=" * 80)

  v_transform = get_vision_transform(cfg)
  g_transform = get_graph_transform(cfg)

  # 1. Base Training Set (64 Classes, 383K Samples)
  train_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="multimodal",
      split="train",
      vision_transform=v_transform,
      graph_transform=g_transform,
  )

  batch_size = int(getattr(cfg.training, "batch_size", 128))
  num_workers = int(getattr(cfg.training, "num_workers", 4))

  train_loader = DataLoader(
      train_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      collate_fn=train_collate,
      pin_memory=True,
      persistent_workers=(num_workers > 0),
      prefetch_factor=2 if num_workers > 0 else None,
      worker_init_fn=seed_worker,
  )

  # 2. Novel Validation Set (16 Classes, Episodic)
  val_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="multimodal",
      split="val",
      vision_transform=v_transform,
      graph_transform=g_transform,
  )

  val_n_way = int(
      getattr(cfg.task, "val_n_way", getattr(cfg.task, "n_way", 5))
  )
  val_n_shot = int(
      getattr(cfg.task, "val_n_shot", getattr(cfg.task, "n_shot", 1))
  )
  val_n_query = int(
      getattr(cfg.task, "val_n_query", getattr(cfg.task, "n_query", 15))
  )
  val_episodes = int(
      getattr(cfg.task, "val_episodes", 200)
  )

  val_sampler = EpisodicBatchSampler(
      val_set.labels,
      val_set.base_names,
      val_n_way,
      val_n_shot,
      val_n_query,
      val_episodes,
  )

  val_loader = DataLoader(
      val_set,
      batch_sampler=val_sampler,
      collate_fn=val_episodic_collate,
      num_workers=num_workers,
      pin_memory=True,
  )

  # 3. Model & Loss Engine
  model = MultimodalPretrainModel(cfg, num_classes=64).to(device)
  loss_engine = MultimodalLossEngine(cfg).to(device)

  # 4. Build Disentangled Optimizers & Schedulers
  optimizers, schedulers = build_optimizers(model, cfg, epochs)

  save_dir = getattr(cfg.training, "save_dir", "experiments/pretrain_multimodal")
  ckpt_dir = os.path.join(save_dir, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)

  best_val_acc = 0.0
  use_ogm = bool(getattr(cfg.training, "use_ogm", False))

  for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct_fused, total_samples = 0.0, 0, 0
    t0 = time.time()

    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Joint Pretrain]"
    )
    for batch in pbar:
      imgs = batch["image"].to(device, non_blocking=True)
      graphs = batch["graph"].to(device, non_blocking=True)
      labels = batch["label"].to(device, non_blocking=True)

      # Zero all active optimizers
      for opt in optimizers:
        opt.zero_grad(set_to_none=True)

      # Forward pass with modality dropout protection
      logits_fuse, logits_v, logits_g, z_v_norm, z_g_norm = model(imgs, graphs)

      loss, loss_fuse, loss_v, loss_g = loss_engine(
          logits_fuse, logits_v, logits_g, z_v_norm, z_g_norm, labels
      )

      loss.backward()

      # On-the-Fly Gradient Modulation: Throttles vision if dominating
      if use_ogm:
        apply_ogm_gradient_modulation(
            model=model,
            loss_v=loss_v,
            loss_g=loss_g,
            alpha=float(getattr(cfg.training, "ogm_alpha", 1.0)),
        )

      # Global gradient clipping to preserve GINE numerical stability
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

      # Step all active optimizers independently
      for opt in optimizers:
        opt.step()

      total_loss += loss.item() * len(labels)
      correct_fused += (logits_fuse.argmax(dim=-1) == labels).sum().item()
      total_samples += len(labels)

      pbar.set_postfix({
          "Loss": f"{loss.item():.4f}",
          "Acc": f"{(correct_fused / total_samples) * 100.0:.2f}%",
      })

    # Step independent learning rate schedulers
    for sched in schedulers:
      sched.step()

    train_loss = total_loss / total_samples
    train_acc = (correct_fused / total_samples) * 100.0

    # 5. Episodic Validation on Novel Classes
    val_acc, val_ci = run_episodic_val(
        model,
        val_loader,
        device,
        n_way=val_n_way,
        n_shot=val_n_shot,
        n_query=val_n_query,
    )

    if len(schedulers) == 3:
      lr_str = (
          f"LR(v: {schedulers[0].get_last_lr()[0]:.5f} | g:"
          f" {schedulers[1].get_last_lr()[0]:.6f} | f:"
          f" {schedulers[2].get_last_lr()[0]:.6f})"
      )
    else:
      lr_str = f"LR: {schedulers[0].get_last_lr()[0]:.6f}"

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train"
        f" 64-Acc: {train_acc:.2f}% | Novel Val 1-Shot: {val_acc:.2f}% ±"
        f" {val_ci:.2f}% | {lr_str} | Elapsed: {time.time()-t0:.1f}s"
    )

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_path = os.path.join(ckpt_dir, "best_model.pth")
      torch.save(
          {
              "epoch": epoch,
              "model_state_dict": model.state_dict(),
              "val_acc": val_acc,
              "cfg": cfg,
          },
          best_path,
      )
      print(
          f"  -> [SAVED NEW BEST] Novel Val 1-Shot: {val_acc:.2f}% to {best_path}"
      )

  print("=" * 80)
  print(f"Pre-Training Complete! Peak Novel Val 1-Shot: {best_val_acc:.2f}%")
  print("=" * 80)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
  seed_everything(cfg.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  run_joint_pretraining(cfg, device)


if __name__ == "__main__":
  main()
