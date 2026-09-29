# train.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork


def seed_worker(worker_id):
    """
    Ensures every DataLoader worker process has an independent, reproducible seed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(data_list):
    """
    Collate function to assemble batched images and batched PyG graphs.
    """
    has_images = hasattr(data_list[0], 'x_img') and data_list[0].x_img is not None

    if has_images:
        images = torch.stack([data.x_img for data in data_list])
        for data in data_list:
            del data.x_img  # Free memory and prevent PyG from merging images
    else:
        images = None

    batched_graphs = Batch.from_data_list(data_list)

    return {
        'image': images,
        'graph': batched_graphs
    }


def calculate_accuracy(logits, targets):
    """
    Computes top-1 accuracy percentage.
    """
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def run_training(cfg, device):
    """
    Main training and validation loop for episodic few-shot learning.
    Strictly deterministic, pure FP32, with optimized DataLoader throughput.
    """
    save_dir = cfg.training.save_dir
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # 1. Setup Data Transforms & Datasets
    v_transform = get_vision_transform(cfg)
    g_transform = get_graph_transform(cfg)

    train_set = MultimodalFSLDataset(
        cfg.dataset,
        modality=cfg.model.modality,
        split='train',
        vision_transform=v_transform,
        graph_transform=g_transform
    )

    val_set = MultimodalFSLDataset(
        cfg.dataset,
        modality=cfg.model.modality,
        split='val',
        vision_transform=v_transform,
        graph_transform=g_transform
    )

    n_way, n_shot, n_query = cfg.task.n_way, cfg.task.n_shot, cfg.task.n_query

    # 2. Episodic Samplers (Seeded via cfg.seed in main.py)
    train_sampler = EpisodicBatchSampler(
        train_set.labels, train_set.base_names, n_way, n_shot, n_query, cfg.task.train_episodes
    )
    val_sampler = EpisodicBatchSampler(
        val_set.labels, val_set.base_names, n_way, n_shot, n_query, cfg.task.val_episodes
    )

    num_workers = int(getattr(cfg.training, 'num_workers', 4))

    # Dedicated PyTorch Generator to guarantee DataLoader determinism
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    # 3. High-Throughput DataLoaders (Pinned Host Memory, Persistent Workers, Prefetching)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g
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
        generator=g
    )

    # 4. Initialize Model
    model = MultimodalFewShotNetwork(cfg).to(device)

    # 5. Optimizer & Scheduler Configuration
    opt_type = getattr(cfg.training, 'optimizer', 'auto')
    use_sgd = (
        (opt_type == 'sgd')
        or (cfg.model.modality == 'vision' and opt_type == 'auto')
        or (cfg.training.lr >= 0.05)
    )

    if use_sgd:
        lr = 0.1 if (cfg.model.modality == 'vision' and cfg.training.lr == 0.001) else cfg.training.lr
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=cfg.training.weight_decay,
            nesterov=True
        )

        sched_type = getattr(cfg.training, 'scheduler', 'cosine')
        if sched_type == 'multistep':
            milestones = list(getattr(cfg.training, 'milestones', [40, 70, 90]))
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=0.1
            )
            print(f"--> [Optimizer: SGD Nesterov] lr={lr}, momentum=0.9, weight_decay={cfg.training.weight_decay} | Scheduler: MultiStepLR milestones={milestones}")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.training.epochs,
                eta_min=1e-5
            )
            print(f"--> [Optimizer: SGD Nesterov] lr={lr}, momentum=0.9, weight_decay={cfg.training.weight_decay} | Scheduler: CosineAnnealingLR T_max={cfg.training.epochs}")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=1e-6
        )
        print(f"--> [Optimizer: AdamW] lr={cfg.training.lr}, weight_decay={cfg.training.weight_decay} | Scheduler: CosineAnnealingLR")

    # 6. Training Variables (Pure FP32)
    targets = torch.arange(n_way).repeat_interleave(n_query).long().to(device, non_blocking=True)
    label_smoothing = float(getattr(cfg.training, 'label_smoothing', 0.0))
    accum_steps = max(1, int(getattr(cfg.training, 'accum_steps', 1)))
    best_val_acc = 0.0

    print(f"--- Starting Training: {cfg.model.modality} modality | Seed: {cfg.seed} | Accum Steps: {accum_steps} | Label Smoothing: {label_smoothing} | Precision: FP32 | Workers: {num_workers} ---")

    for epoch in range(1, cfg.training.epochs + 1):
        # --- A. Training Phase ---
        model.train()
        train_accs, train_losses = [], []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            img_batch = batch['image'].to(device, non_blocking=True) if batch['image'] is not None else None
            graph_batch = batch['graph'].to(device, non_blocking=True) if batch['graph'] is not None else None

            # Standard Pure FP32 Forward Pass
            logits = model(img_batch, graph_batch, n_way, n_shot)
            loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

            # Scale loss for gradient accumulation
            (loss / accum_steps).backward()

            # Execute optimizer step every accum_steps episodes (or at epoch end)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            train_accs.append(calculate_accuracy(logits, targets))

        # --- B. Validation Phase ---
        model.eval()
        val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                img_batch = batch['image'].to(device, non_blocking=True) if batch['image'] is not None else None
                graph_batch = batch['graph'].to(device, non_blocking=True) if batch['graph'] is not None else None

                logits = model(img_batch, graph_batch, n_way, n_shot)
                val_accs.append(calculate_accuracy(logits, targets))

        mean_val_acc = float(np.mean(val_accs))
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {np.mean(train_accs):.2f}% | Val Acc: {mean_val_acc:.2f}% | LR: {current_lr:.6f}")

        # --- C. Checkpointing ---
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'cfg': cfg
            }, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
            print(f"  -> New Best Model Saved! ({best_val_acc:.2f}%)")

        scheduler.step()
