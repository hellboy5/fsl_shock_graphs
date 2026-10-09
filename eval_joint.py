# eval_joint.py
import argparse
import math
import os

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
from train import calculate_accuracy, collate_fn, seed_worker
from utils.helpers import seed_everything


def compute_confidence_interval(data):
  """Calculates the mean and 95% confidence interval across episodes."""
  a = 1.0 * np.array(data)
  m = float(np.mean(a))
  pm = float(1.96 * (np.std(a) / math.sqrt(len(a))))
  return m, pm


def evaluate_checkpoint(ckpt_path: str, episodes: int = 10000, device_str: str = "cuda"):
  device = torch.device(device_str if torch.cuda.is_available() else "cpu")

  if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

  print("=" * 80)
  print(f"EVALUATING JOINT CHECKPOINT: {ckpt_path}")
  print("=" * 80)

  # 1. Load Checkpoint & Recover Config (Matches eval.py)
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  train_cfg = checkpoint['cfg']

  seed_everything(getattr(train_cfg, "seed", 42))

  # 2. Re-create Dataset & Loader on the 20 Novel Test Classes (Matches eval.py)
  test_set = MultimodalFSLDataset(
      train_cfg.dataset,
      modality="multimodal",
      split="test",
      vision_transform=get_vision_transform(train_cfg),
      graph_transform=get_graph_transform(train_cfg),
  )

  test_sampler = EpisodicBatchSampler(
      test_set.labels,
      test_set.base_names,
      5,
      1,
      15,
      episodes,
  )

  g = torch.Generator()
  g.manual_seed(getattr(train_cfg, "seed", 42))

  test_loader = DataLoader(
      test_set,
      batch_sampler=test_sampler,
      collate_fn=collate_fn,
      num_workers=4,
      pin_memory=True,
      persistent_workers=True,
      prefetch_factor=2,
      worker_init_fn=seed_worker,
      generator=g,
  )

  # 3. Setup Model & Load Weights (Strict=False filters out aux heads safely)
  model = MultimodalFewShotNetwork(train_cfg).to(device)
  model.load_state_dict(checkpoint['model_state_dict'], strict=False)
  model.eval()

  print(f"--> Successfully restored target layers from pre-trained checkpoint.")
  print(f"--> Starting 5-Way 1-Shot Test Evaluation ({episodes} Episodes)...")

  targets = (
      torch.arange(5)
      .repeat_interleave(15)
      .long()
      .to(device, non_blocking=True)
  )

  accs = []

  # 4. Evaluation Loop
  with torch.no_grad():
    for batch in tqdm(test_loader, desc="10k Test Evaluation"):
      imgs = batch["image"].to(device, non_blocking=True) if batch["image"] is not None else None
      graphs = batch["graph"].to(device, non_blocking=True)

      logits = model(imgs, graphs, 5, 1)
      accs.append(calculate_accuracy(logits, targets))

  mean_acc, ci = compute_confidence_interval(accs)
  print("\n" + "=" * 80)
  print(f"FINAL RESULT ({episodes} Episodes): {mean_acc:6.2f}% ± {ci:.2f}%")
  print("=" * 80)
  return mean_acc, ci


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
  parser.add_argument("--episodes", type=int, default=10000, help="Number of test episodes")
  parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
  args = parser.parse_args()

  evaluate_checkpoint(args.checkpoint, args.episodes, args.device)
