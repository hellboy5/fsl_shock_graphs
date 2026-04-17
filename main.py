import hydra
import torch
from utils.seed import set_global_seeds
from train import run_training
from eval import run_evaluation

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    # 1. Global Setup
    set_global_seeds(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Modality/Experiment Routing
    if cfg.mode == 'train':
        print(f"Starting Training Run for Modality: {cfg.model.modality}")
        run_training(cfg, device)
        
    elif cfg.mode == 'eval':
        print(f"Starting Evaluation for Checkpoint: {cfg.checkpoint_path}")
        run_evaluation(cfg.checkpoint_path, cfg.dataset.data_root, 
                       cfg.task.n_way, cfg.task.n_shot, cfg.task.n_query)

if __name__ == "__main__":
    main()
