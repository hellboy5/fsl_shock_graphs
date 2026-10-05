# train.py
import os
import random
from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % (2**32)
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def collate_fn(data_list):
  has_images = (
      hasattr(data_list[0], 'x_img') and data_list[0].x_img is not None
  )

  if has_images:
    images = torch.stack([data.x_img for data in data_list])
    for data in data_list:
      del data.x_img
  else:
    images = None

  batched_graphs = Batch.from_data_list(data_list)
  return {'image': images, 'graph': batched_graphs}


def calculate_accuracy(logits, targets):
  pred = logits.argmax(dim=1)
  return (pred == targets).float().mean().item() * 100.0


def run_training(cfg, device):
  save_dir = cfg.training.save_dir
  os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

  v_transform = get_vision_transform(cfg)
  g_transform = get_graph_transform(cfg)

  train_set = MultimodalFSLDataset(
      cfg.dataset,
      modality=cfg.model.modality,
      split='train',
      vision_transform=v_transform,
      graph_transform=g_transform,
  )
  val_set = MultimodalFSLDataset(
      cfg.dataset,
      modality=cfg.model.modality,
      split='val',
      vision_transform=v_transform,
      graph_transform=g_transform,
  )

  n_way, n_shot, n_query = cfg.task.n_way, cfg.task.n_shot, cfg.task.n_query

  train_sampler = EpisodicBatchSampler(
      train_set.labels,
      train_set.base_names,
      n_way,
      n_shot,
      n_query,
      cfg.task.train_episodes,
  )
  val_sampler = EpisodicBatchSampler(
      val_set.labels,
      val_set.base_names,
      n_way,
      n_shot,
      n_query,
      cfg.task.val_episodes,
  )

  num_workers = int(getattr(cfg.training, 'num_workers', 4))
  g = torch.Generator()
  g.manual_seed(cfg.seed)

  train_loader = DataLoader(
      train_set,
      batch_sampler=train_sampler,
      collate_fn=collate_fn,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=(num_workers > 0),
      prefetch_factor=2 if num_workers > 0 else None,
      worker_init_fn=seed_worker,
      generator=g,
  )
  val_loader = DataLoader(
      val_set,
      batch_sampler=val_sampler,
      collate_fn=collate_fn,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=(num_workers > 0),
      prefetch_factor=2 if num_workers > 0 else None,
      worker_init_fn=seed_worker,
      generator=g,
  )

  model = MultimodalFewShotNetwork(cfg).to(device)

  # --- Warm-Start Checkpoint Loading (Strategy 3 / Meta-Baseline) ---
  ckpt_path = getattr(cfg, 'checkpoint_path', None) or getattr(
      cfg.evaluation, 'checkpoint_path', None
  )
  if ckpt_path:
    print(f'--> [Warm-Start] Loading pre-trained weights from: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Allow loading state_dict whether saved as full model or encoder submodule
    model_dict = model.state_dict()
    matched_dict = {}
    for k, v in state_dict.items():
      if k in model_dict and v.shape == model_dict[k].shape:
        matched_dict[k] = v
      elif (
          f'graph_encoder.{k}' in model_dict
          and v.shape == model_dict[f'graph_encoder.{k}'].shape
      ):
        matched_dict[f'graph_encoder.{k}'] = v
      elif (
          k.replace('graph_encoder.', '') in model_dict
          and v.shape == model_dict[k.replace('graph_encoder.', '')].shape
      ):
        matched_dict[k.replace('graph_encoder.', '')] = v

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    print(
        f'--> [Warm-Start] Successfully loaded {len(matched_dict)} /'
        f' {len(model_dict)} tensors.'
    )

  # Optimizer: SGD Nesterov for Vision, AdamW for Graph and Fusion
  opt_type = getattr(cfg.training, 'optimizer', 'auto')
  use_sgd = (
      (opt_type == 'sgd')
      or (cfg.model.modality == 'vision' and opt_type == 'auto')
      or (cfg.training.lr >= 0.05)
  )

  if use_sgd:
    lr = (
        0.1
        if (cfg.model.modality == 'vision' and cfg.training.lr == 0.001)
        else cfg.training.lr
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=cfg.training.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-5
    )
    print(
        f'--> [Optimizer: SGD Nesterov] lr={lr}, momentum=0.9,'
        f' weight_decay={cfg.training.weight_decay} | Scheduler:'
        ' CosineAnnealingLR'
    )
  else:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    print(
        f'--> [Optimizer: AdamW] lr={cfg.training.lr},'
        f' weight_decay={cfg.training.weight_decay} | Scheduler:'
        ' CosineAnnealingLR'
    )

  targets = (
      torch.arange(n_way)
      .repeat_interleave(n_query)
      .long()
      .to(device, non_blocking=True)
  )
  best_val_acc = 0.0

  print(
      f'--- Starting Training: {cfg.model.modality} modality | Seed: {cfg.seed}'
      f' | Workers: {num_workers} ---'
  )

  for epoch in range(1, cfg.training.epochs + 1):
    model.train()
    train_accs, train_losses = [], []

    for batch in train_loader:
      optimizer.zero_grad(set_to_none=True)
      img_batch = (
          batch['image'].to(device, non_blocking=True)
          if batch['image'] is not None
          else None
      )
      graph_batch = (
          batch['graph'].to(device, non_blocking=True)
          if batch['graph'] is not None
          else None
      )

      logits = model(img_batch, graph_batch, n_way, n_shot)
      loss = F.cross_entropy(logits, targets)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      train_losses.append(loss.item())
      train_accs.append(calculate_accuracy(logits, targets))

    model.eval()
    val_accs = []
    with torch.no_grad():
      for batch in val_loader:
        img_batch = (
            batch['image'].to(device, non_blocking=True)
            if batch['image'] is not None
            else None
        )
        graph_batch = (
            batch['graph'].to(device, non_blocking=True)
            if batch['graph'] is not None
            else None
        )

        logits = model(img_batch, graph_batch, n_way, n_shot)
        val_accs.append(calculate_accuracy(logits, targets))

    mean_val_acc = float(np.mean(val_accs))
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f'Epoch {epoch:03d} | Train Loss: {np.mean(train_losses):.4f} | Train'
        f' Acc: {np.mean(train_accs):.2f}% | Val Acc: {mean_val_acc:.2f}% | LR:'
        f' {current_lr:.6f}',
        flush=True,
    )

    if mean_val_acc > best_val_acc:
      best_val_acc = mean_val_acc
      torch.save(
          {
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'best_val_acc': best_val_acc,
              'cfg': cfg,
          },
          os.path.join(save_dir, 'checkpoints', 'best_model.pth'),
      )
      print(f'  -> New Best Model Saved! ({best_val_acc:.2f}%)', flush=True)

    scheduler.step()
