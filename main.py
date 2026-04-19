import hydra
import torch
from utils.helpers import seed_everything
from train import run_training
from eval import run_evaluation

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    # 1. Global Setup
    seed_everything(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Modality/Experiment Routing
    if cfg.mode == 'train':
        print(f"Starting Training Run for Modality: {cfg.model.modality}")
        run_training(cfg, device)
        
    elif cfg.mode == 'eval':
        print(f"Starting Evaluation for Checkpoint: {cfg.checkpoint_path}")
        run_evaluation(cfg, device)

if __name__ == "__main__":
    main()
