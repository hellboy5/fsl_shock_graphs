# eval.py
from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import calculate_accuracy, collate_fn, seed_worker


def compute_confidence_interval(data):
  a = 1.0 * np.array(data)
  m = np.mean(a)
  pm = 1.96 * (np.std(a) / np.sqrt(len(a)))
  return m, pm


def run_evaluation(cfg, device):
  checkpoint_path = getattr(cfg, 'checkpoint_path', None)
  if not checkpoint_path:
    raise ValueError(
        'No checkpoint_path specified! Pass checkpoint_path=... on the command'
        ' line.'
    )

  eval_n_way = cfg.task.n_way
  eval_n_shot = cfg.task.n_shot
  eval_n_query = cfg.task.n_query
  eval_episodes = int(getattr(cfg.task, 'test_episodes', 10000))
  num_workers = int(getattr(cfg.training, 'num_workers', 4))

  # 1. Load Checkpoint
  checkpoint = torch.load(
      checkpoint_path, map_location=device, weights_only=False
  )
  train_cfg = checkpoint['cfg']

  print(
      f"Loaded checkpoint from Epoch {checkpoint['epoch']} (Original Val Acc:"
      f" {checkpoint['best_val_acc']:.2f}%)"
  )
  print(
      f'Evaluating on {eval_n_way}-Way {eval_n_shot}-Shot ({eval_episodes}'
      ' episodes)'
  )

  # 2. Setup Test Data (Using deterministic test transforms)
  test_set = MultimodalFSLDataset(
      cfg.dataset,
      modality=train_cfg.model.modality,
      split='test',
      vision_transform=get_vision_transform(train_cfg),
      graph_transform=get_graph_transform(train_cfg),
  )

  test_sampler = EpisodicBatchSampler(
      test_set.labels,
      test_set.base_names,
      eval_n_way,
      eval_n_shot,
      eval_n_query,
      eval_episodes,
  )

  g = torch.Generator()
  g.manual_seed(cfg.seed)

  test_loader = DataLoader(
      test_set,
      batch_sampler=test_sampler,
      collate_fn=collate_fn,
      num_workers=num_workers,
      pin_memory=True,
      worker_init_fn=seed_worker,
      generator=g,
  )

  # 3. Setup Model (Built with train_cfg to match weights; strict=False ignores legacy buffers)
  model = MultimodalFewShotNetwork(train_cfg).to(device)
  model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  model.eval()

  targets = (
      torch.arange(eval_n_way)
      .repeat_interleave(eval_n_query)
      .long()
      .to(device, non_blocking=True)
  )
  test_accs = []

  # 4. Evaluation Loop
  with torch.no_grad():
    for batch in test_loader:
      img_batch = (
          batch['image'].to(device, non_blocking=True)
          if batch['image'] is not None
          else None
      )
      graph_batch = batch['graph'].to(device, non_blocking=True)

      logits = model(img_batch, graph_batch, eval_n_way, eval_n_shot)
      acc = calculate_accuracy(logits, targets)
      test_accs.append(acc)

  mean, ci = compute_confidence_interval(test_accs)
  print(f'\nFinal Test Results: {mean:.2f}% ± {ci:.2f}%\n')


if __name__ == '__main__':
  pass
