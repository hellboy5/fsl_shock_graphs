import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
from train import collate_fn, calculate_accuracy

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    pm = 1.96 * (np.std(a) / np.sqrt(len(a)))
    return m, pm

def run_evaluation(cfg, device):
    # Extract evaluation parameters from the runtime config
    checkpoint_path = cfg.checkpoint_path
    data_dir = cfg.dataset.data_root
    eval_n_way = cfg.task.n_way
    eval_n_shot = cfg.task.n_shot
    eval_n_query = cfg.task.n_query
    
    # Use 2000 episodes for testing, unless overridden in config
    eval_episodes = cfg.task.get('eval_episodes', 2000)
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_cfg = checkpoint['cfg'] # Retrieve the config used during training to rebuild model
    
    print(f"Loaded checkpoint from Epoch {checkpoint['epoch']} (Original Val Acc: {checkpoint['best_val_acc']:.2f}%)")
    print(f"Evaluating on {eval_n_way}-Way {eval_n_shot}-Shot ({eval_episodes} episodes)")

    # 2. Setup Test Data (Using the NEW evaluation parameters)
    test_set = MultimodalFSLDataset(data_dir, split='test', 
                                    vision_transform=get_vision_transform('test'), 
                                    graph_transform=get_graph_transform(train_cfg))
    
    test_sampler = EpisodicBatchSampler(test_set.labels, eval_n_way, eval_n_shot, eval_n_query, eval_episodes)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=collate_fn)

    # 3. Setup Model (Built with train_cfg to match weights exactly)
    model = MultimodalFewShotNetwork(train_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    targets = torch.arange(eval_n_way).repeat(eval_n_query).long().to(device)
    test_accs = []

    # 4. Evaluation Loop
    with torch.no_grad():
        for batch in test_loader:
            # We pass the NEW runtime n_way and k_shot so the FSL Engine splits the tensor correctly
            logits = model(batch['image'].to(device), batch['graph'].to(device), eval_n_way, eval_n_shot)
            acc = calculate_accuracy(logits, targets)
            test_accs.append(acc)

    mean, ci = compute_confidence_interval(test_accs)
    print(f"\nFinal Test Results: {mean:.2f}% ± {ci:.2f}%\n")

if __name__ == "__main__":
    pass
