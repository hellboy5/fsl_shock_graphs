# pretrain_multimodal.py
import math
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
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


def train_collate(data_list):
  images = torch.stack([d["image"] for d in data_list])
  graphs = Batch.from_data_list([d["graph"] for d in data_list])
  labels = torch.tensor([d["label"] for d in data_list], dtype=torch.long)
  return {"image": images, "graph": graphs, "label": labels}


def val_episodic_collate(data_list):
  images = torch.stack([d["image"] for d in data_list])
  graphs = Batch.from_data_list([d["graph"] for d in data_list])
  return {"image": images, "graph": graphs}


def run_episodic_val(
    model, val_loader, device, n_way=5, n_shot=1, n_query=15, episodes=200
):
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

      z = model.extract_fused_features(imgs, graphs)
      z_supp = z[:k_total].view(n_way, n_shot, -1).mean(dim=1)
      z_supp = F.normalize(z_supp, p=2, dim=-1)

      z_query = F.normalize(z[k_total:], p=2, dim=-1)
      cos_sim = torch.mm(z_query, z_supp.t())
      preds = cos_sim.argmax(dim=-1)
      accs.append((preds == targets).float().mean().item() * 100.0)

  mean_acc = float(np.mean(accs))
  ci_acc = float(1.96 * (np.std(accs) / math.sqrt(len(accs))))
  return mean_acc, ci_acc


def run_joint_pretraining(cfg: DictConfig, device: torch.device):
  loss_type = getattr(cfg.training, "loss_type", "ce_multitask")
  print("=" * 80)
  print(
      f"STARTING JOINT PRE-TRAINING | Fusion: {cfg.model.fusion_type} | Loss:"
      f" {loss_type}"
  )
  print("=" * 80)

  v_transform = get_vision_transform(cfg)
  g_transform = get_graph_transform(cfg)

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

  val_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="multimodal",
      split="val",
      vision_transform=v_transform,
      graph_transform=g_transform,
  )

  val_sampler = EpisodicBatchSampler(
      val_set.labels,
      val_set.base_names,
      n_way=5,
      n_shot=1,
      n_query=15,
      n_episodes=200,
  )

  val_loader = DataLoader(
      val_set,
      batch_sampler=val_sampler,
      collate_fn=val_episodic_collate,
      num_workers=num_workers,
      pin_memory=True,
  )

  model = MultimodalPretrainModel(cfg, num_classes=64).to(device)
  loss_engine = MultimodalLossEngine(cfg).to(device)

  lr = float(getattr(cfg.training, "lr", 0.001))
  weight_decay = float(getattr(cfg.training, "weight_decay", 0.0001))
  epochs = int(getattr(cfg.training, "epochs", 35))

  optimizer = torch.optim.AdamW(
      model.parameters(), lr=lr, weight_decay=weight_decay
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-6
  )

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

      optimizer.zero_grad(set_to_none=True)

      logits_fuse, logits_v, logits_g, z_v_norm, z_g_norm = model(imgs, graphs)

      loss, loss_fuse, loss_v, loss_g = loss_engine(
          logits_fuse, logits_v, logits_g, z_v_norm, z_g_norm, labels
      )

      loss.backward()

      # On-the-Fly Gradient Modulation: Dampens vision if dominating
      if use_ogm and (loss_v.item() < loss_g.item()):
        ratio = (loss_g.item() - loss_v.item()) / (loss_g.item() + 1e-6)
        coeff = 1.0 - math.tanh(ratio)
        for p in model.vision_encoder.parameters():
          if p.grad is not None:
            p.grad.data.mul_(coeff)

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
      optimizer.step()

      total_loss += loss.item() * len(labels)
      correct_fused += (logits_fuse.argmax(dim=-1) == labels).sum().item()
      total_samples += len(labels)

      pbar.set_postfix({
          "Loss": f"{loss.item():.4f}",
          "Acc": f"{(correct_fused / total_samples) * 100.0:.2f}%",
      })

    scheduler.step()

    train_loss = total_loss / total_samples
    train_acc = (correct_fused / total_samples) * 100.0

    val_acc, val_ci = run_episodic_val(
        model, val_loader, device, n_way=5, n_shot=1, n_query=15, episodes=200
    )

    cur_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train"
        f" 64-Acc: {train_acc:.2f}% | Novel Val 1-Shot: {val_acc:.2f}% ±"
        f" {val_ci:.2f}% | LR: {cur_lr:.6f} | Elapsed: {time.time()-t0:.1f}s"
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
