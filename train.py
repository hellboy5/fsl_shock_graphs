# train.py
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
from torch_geometric.data import Batch


def collate_fn(data_list):
    # 1. Safely check if images exist in the batch (multimodal/vision mode)
    has_images = hasattr(data_list[0], 'x_img')
    
    if has_images:
        images = torch.stack([data.x_img for data in data_list])
        for data in data_list:
            del data.x_img  # Prevent PyG from merging images
    else:
        images = None
        
    # 2. Use PyG's native collater for the graphs
    batched_graphs = Batch.from_data_list(data_list)
    
    return {
        'image': images,
        'graph': batched_graphs
    }


def calculate_accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def run_training(cfg, device):
    # Setup Dirs: Save directly into the exact path configured by Hydra
    save_dir = cfg.training.save_dir
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    # Setup Data Transforms (Driven entirely by config)
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

    train_sampler = EpisodicBatchSampler(train_set.labels, train_set.base_names, n_way, n_shot, n_query, cfg.task.train_episodes)
    val_sampler = EpisodicBatchSampler(val_set.labels, val_set.base_names, n_way, n_shot, n_query, cfg.task.val_episodes)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=getattr(cfg.training, 'num_workers', 2))
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=getattr(cfg.training, 'num_workers', 2))

    # Setup Model
    model = MultimodalFewShotNetwork(cfg).to(device)

    # 1. AdamW Optimizer (Loshchilov & Hutter, ICLR 2019)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    # 2. Cosine Annealing Schedule (Smooth decay to 1e-6 floor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=1e-6
    )

    targets = torch.arange(n_way).repeat_interleave(n_query).long().to(device)
    best_val_acc = 0.0
    
    print(f"--- Starting Training: {cfg.model.modality} modality ---")
    for epoch in range(1, cfg.training.epochs + 1):
        # Train Loop
        model.train()
        train_accs, train_losses = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            img_batch = batch['image'].to(device) if batch['image'] is not None else None
            graph_batch = batch['graph'].to(device)
            logits = model(img_batch, graph_batch, n_way, n_shot)

            loss = F.cross_entropy(logits, targets)
            loss.backward()

            # 3. Gradient Clipping (max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(calculate_accuracy(logits, targets))

        # Validation Loop
        model.eval()
        val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                img_batch = batch['image'].to(device) if batch['image'] is not None else None
                graph_batch = batch['graph'].to(device)
                logits = model(img_batch, graph_batch, n_way, n_shot)
                val_accs.append(calculate_accuracy(logits, targets))
                
        mean_val_acc = np.mean(val_accs)
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {np.mean(train_accs):.2f}% | Val Acc: {mean_val_acc:.2f}%")

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
