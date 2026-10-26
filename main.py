# main.py
from eval import run_evaluation
import hydra
from pretrain import run_pretraining
import torch
from train import run_training
from utils.helpers import seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
  # 1. Global Setup
  seed_everything(cfg.seed)
  device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

  # 2. Routing
  if cfg.mode == "train":
    print(f"Starting Episodic Training Run for Modality: {cfg.model.modality}")
    run_training(cfg, device)

  elif cfg.mode == "pretrain":
    print(
        f"Starting {cfg.dataset.num_classes}-Class Supervised Pre-Training"
        f" for: {cfg.model.modality}"
    )
    run_pretraining(cfg, device)

  elif cfg.mode == "eval":
    print(f"Starting Evaluation for Checkpoint: {cfg.checkpoint_path}")
    run_evaluation(cfg, device)

  else:
    raise ValueError(
        f"Unknown mode: '{cfg.mode}'. Expected 'pretrain', 'train', or 'eval'."
    )


if __name__ == "__main__":
  main()
