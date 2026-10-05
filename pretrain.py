# pretrain.py
import os
import random
from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform
from models.encoders.gnn_encoder import GraphEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % (2**32)
  np.random.seed(worker_seed)
  random.seed(worker_seed)


# 1. Mini-Batch Collate Function for Standard Pre-Training (64-Class CE)
def pretrain_graph_collate(data_list):
  targets = torch.tensor([d.y.item() for d in data_list], dtype=torch.long)
  batched_graphs = Batch.from_data_list(data_list)
  return batched_graphs, targets


# 2. Episodic Collate Function for Validation (5-Way 1-Shot)
def val_episodic_collate(data_list):
  return Batch.from_data_list(data_list)


# 3. Model Wrapper (Backbone + 64-Class Linear Head)
class GraphClassificationModel(nn.Module):

  def __init__(self, cfg, num_classes=64):
    super(GraphClassificationModel, self).__init__()
    g_hidden_dim = getattr(cfg.model, "graph_hidden_dim", 128)
    g_proj_dim = getattr(cfg.model, "graph_proj_dim", 640)

    self.encoder = GraphEncoder(
        node_feat_dim=cfg.model.node_feat_dim,
        edge_feat_dim=cfg.model.edge_feat_dim,
        hidden_dim=g_hidden_dim,
        proj_feat_dim=g_proj_dim,
        gnn_type=cfg.model.gnn_type,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        norm_type=getattr(cfg.model, "norm_type", "graph"),
        pooling_method=getattr(cfg.model, "pooling_method", "global_attention"),
        train_eps=getattr(cfg.model, "train_eps", True),
        use_input_mlp=getattr(cfg.model, "use_input_mlp", True),
        use_jk=getattr(cfg.model, "use_jk", True),
    )
    self.classifier = nn.Linear(g_proj_dim, num_classes)

  def forward(self, graph_batch):
    z = self.encoder(graph_batch)  # [B, 640]
    logits = self.classifier(z)  # [B, 64]
    return logits, z


# 4. Novel-Class Validation Step (5-Way 1-Shot Cosine Matching)
def validate_few_shot(
    encoder, val_loader, n_way, n_query, eval_episodes, device
):
  encoder.eval()
  targets = (
      torch.arange(n_way)
      .repeat_interleave(n_query)
      .long()
      .to(device, non_blocking=True)
  )
  episode_accs = []

  with torch.no_grad():
    for graph_batch in val_loader:
      graph_batch = graph_batch.to(device, non_blocking=True)
      embeddings = encoder(graph_batch)  # [Total_Samples, 640]
      embeddings = F.normalize(embeddings, p=2, dim=-1)

      n_support = n_way * 1  # 1-shot
      prototypes = embeddings[:n_support]  # [5, 640]
      queries = embeddings[n_support:]  # [75, 640]

      # Cosine similarity scaled by 10.0
      logits = torch.mm(queries, prototypes.t()) * 10.0
      preds = logits.argmax(dim=-1)
      acc = (preds == targets).float().mean().item() * 100.0
      episode_accs.append(acc)

  mean_acc = float(np.mean(episode_accs))
  ci = float(1.96 * (np.std(episode_accs) / np.sqrt(len(episode_accs))))
  return mean_acc, ci


# 5. Main Pre-Training Function
def run_pretraining(cfg, device):
  save_dir = cfg.training.save_dir
  os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

  epochs = int(getattr(cfg.training, "epochs", 60))
  batch_size = int(getattr(cfg.training, "batch_size", 128))
  num_workers = int(getattr(cfg.training, "num_workers", 4))

  # Validation task parameters reused cleanly from cfg.task
  val_n_way = 5
  val_n_shot = 1
  val_n_query = cfg.task.n_query
  val_episodes = int(getattr(cfg.task, "val_episodes", 200))

  # 1. Enforce Un-Coarsened mode
  cfg.dataset.graph.use_coarse = False
  g_transform = get_graph_transform(cfg)

  # 2. Build Datasets
  print("--> [Data] Loading Base Training Set (64 Classes, 383K Graphs)...")
  train_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="graph",
      split="train",
      vision_transform=None,
      graph_transform=g_transform,
  )

  print(
      "--> [Data] Loading Novel Validation Set (16 Classes, 5-Way 1-Shot)..."
  )
  val_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="graph",
      split="val",
      vision_transform=None,
      graph_transform=g_transform,
  )

  g = torch.Generator()
  g.manual_seed(cfg.seed)

  # Standard Mini-Batch DataLoader for Training (100% Data Exposure)
  train_loader = DataLoader(
      train_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      collate_fn=pretrain_graph_collate,
      pin_memory=True,
      persistent_workers=(num_workers > 0),
      worker_init_fn=seed_worker,
      generator=g,
  )

  # Episodic Sampler for Novel-Class Validation
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

  # 3. Model Setup
  model = GraphClassificationModel(cfg, num_classes=64).to(device)

  # 4. Adaptive Optimizer Setup: AdamW for GNNs, SGD fallback if specified
  opt_type = getattr(cfg.training, "optimizer", "auto")
  use_sgd = (opt_type == "sgd") or (cfg.training.lr >= 0.05)

  if use_sgd:
    lr = cfg.training.lr
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=cfg.training.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    print(
        f"--> [Pretrain Optimizer: SGD Nesterov] lr={lr}, momentum=0.9,"
        f" weight_decay={cfg.training.weight_decay}"
    )
  else:
    # GNN Gold Standard: AdamW with lr=0.001
    lr = (
        0.001
        if (cfg.training.lr == 0.001 or opt_type == "auto")
        else cfg.training.lr
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(getattr(cfg.training, "weight_decay", 0.0001)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    print(
        f"--> [Pretrain Optimizer: AdamW] lr={lr},"
        f" weight_decay={optimizer.param_groups[0]['weight_decay']}"
    )

  criterion = nn.CrossEntropyLoss()
  best_val_acc = 0.0

  print("\n" + "=" * 80)
  print(
      f"   STARTING 64-CLASS SUPERVISED PRE-TRAINING ({epochs} Epochs, Batch:"
      f" {batch_size})   "
  )
  print("=" * 80)

  for epoch in range(1, epochs + 1):
    model.train()
    train_losses, train_accs = [], []

    pbar = tqdm(
        train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train 64-Class]"
    )
    for graphs, targets in pbar:
      graphs = graphs.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      logits, _ = model(graphs)
      loss = criterion(logits, targets)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      acc = (logits.argmax(dim=-1) == targets).float().mean().item() * 100.0
      train_losses.append(loss.item())
      train_accs.append(acc)

      pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

    scheduler.step()

    # 5. Novel-Class Validation Step
    val_1shot, val_ci = validate_few_shot(
        model.encoder,
        val_loader,
        val_n_way,
        val_n_query,
        val_episodes,
        device,
    )

    epoch_loss = float(np.mean(train_losses))
    epoch_train_acc = float(np.mean(train_accs))
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} |"
        f" Train 64-Class Acc: {epoch_train_acc:.2f}% | Novel Val 1-Shot:"
        f" {val_1shot:.2f}% ± {val_ci:.2f}% | LR: {current_lr:.6f}"
    )

    # 6. Save Checkpoint When Transferability Peaks
    if val_1shot > best_val_acc:
      best_val_acc = val_1shot
      ckpt_path = os.path.join(save_dir, "checkpoints", "best_model.pth")
      torch.save(
          {
              "epoch": epoch,
              "model_state_dict": {
                  f"graph_encoder.{k}": v
                  for k, v in model.encoder.state_dict().items()
              },
              "best_val_acc": best_val_acc,
              "cfg": cfg,
          },
          ckpt_path,
      )
      print(f"  -> [SAVED NEW BEST] Novel Val 1-Shot: {best_val_acc:.2f}%\n")

  print("\n" + "=" * 80)
  print(f"Pre-Training Complete! Peak Novel Val 1-Shot: {best_val_acc:.2f}%")
  print(
      f"Best Model Saved to: {os.path.join(save_dir, 'checkpoints', 'best_model.pth')}"
  )
  print("=" * 80)
