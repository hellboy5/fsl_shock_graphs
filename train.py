import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
from torch_geometric.data import Batch

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'image': images, 'graph': graphs, 'label': labels}

def calculate_accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0

def run_training(cfg, device):
    # Setup Dirs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(cfg.training.save_dir, f"run_{timestamp}")
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    # Setup Data
    v_transform_train = get_vision_transform(split='train')
    v_transform_val = get_vision_transform(split='val')
    g_transform = get_graph_transform(cfg)
    
    train_set = MultimodalFSLDataset(cfg.dataset.data_root, split='train', vision_transform=v_transform_train, graph_transform=g_transform)
    val_set = MultimodalFSLDataset(cfg.dataset.data_root, split='val', vision_transform=v_transform_val, graph_transform=g_transform)

    n_way, n_shot, n_query = cfg.task.n_way, cfg.task.n_shot, cfg.task.n_query
    
    train_sampler = EpisodicBatchSampler(train_set.labels, n_way, n_shot, n_query, cfg.task.train_episodes)
    val_sampler = EpisodicBatchSampler(val_set.labels, n_way, n_shot, n_query, cfg.task.val_episodes)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=collate_fn)

    # Setup Model & Optim
    model = MultimodalFewShotNetwork(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=0.9, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    targets = torch.arange(n_way).repeat(n_query).long().to(device)
    best_val_acc = 0.0
    
    print(f"--- Starting Training: {cfg.model.modality} modality ---")
    for epoch in range(1, cfg.training.epochs + 1):
        # Train Loop
        model.train()
        train_accs, train_losses = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['image'].to(device), batch['graph'].to(device), n_way, n_shot)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(calculate_accuracy(logits, targets))

        # Validation Loop
        model.eval()
        val_accs = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['image'].to(device), batch['graph'].to(device), n_way, n_shot)
                val_accs.append(calculate_accuracy(logits, targets))
                
        mean_val_acc = np.mean(val_accs)
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {np.mean(train_accs):.2f}% | Val Acc: {mean_val_acc:.2f}%")

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'cfg': cfg # Save the config so eval knows what model to build
            }, os.path.join(save_dir, 'checkpoints', 'best_model.pth'))
            print("  -> New Best Model Saved!")

        scheduler.step()
