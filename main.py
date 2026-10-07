# main.py
import hydra
from pretrain import run_pretraining
import torch
from utils.helpers import seed_everything
from pretrain_multimodal import run_joint_pretraining

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
  # 1. Global Setup
  seed_everything(cfg.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 2. Routing
  if cfg.mode == "train":
    from train import run_training

    print(f"Starting Episodic Training Run for Modality: {cfg.model.modality}")
    run_training(cfg, device)

  elif cfg.mode == "pretrain":
    print(
        f"Starting 64-Class Supervised Pre-Training for: {cfg.model.modality}"
    )
    run_pretraining(cfg, device)
  elif cfg.mode in ["pretrain_multimodal", "joint_pretrain"]:
    run_joint_pretraining(cfg, device)
  elif cfg.mode == "eval":
    from eval import run_evaluation

    print(f"Starting Evaluation for Checkpoint: {cfg.checkpoint_path}")
    run_evaluation(cfg, device)


if __name__ == "__main__":
  main()
