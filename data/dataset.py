# data/dataset.py
import os
import glob
import random
import torch
from torch_geometric.data import Dataset
from PIL import Image
from torchvision import transforms
from .transforms import NormalizeGraphContinuous

class MultimodalFSLDataset(Dataset):
    def __init__(self, cfg, split='train', transform=None, pre_transform=None):
        super().__init__(root=cfg.data_root, transform=transform, pre_transform=pre_transform)
        self.cfg = cfg
        self.split = split
        self.modality = cfg.modality
        self.split_dir = os.path.join(cfg.data_root, split)
        
        # Image Transform
        self.image_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalize_img.mean, std=cfg.normalize_img.std)
        ])

        # PyG Graph Transform
        self.graph_transform = NormalizeGraphContinuous(
            node_mean=cfg.normalize_graph.node_mean,
            node_std=cfg.normalize_graph.node_std,
            edge_mean=cfg.normalize_graph.edge_mean,
            edge_std=cfg.normalize_graph.edge_std
        )

        self.samples = self._load_and_group_files()

    def _load_and_group_files(self):
        samples = []
        class_folders = sorted([d for d in os.listdir(self.split_dir) 
                                if os.path.isdir(os.path.join(self.split_dir, d))])
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.split_dir, class_name)
            pth_files = glob.glob(os.path.join(class_path, "*.pth"))
            
            grouped_files = {}
            for pth_path in pth_files:
                filename = os.path.basename(pth_path)
                base_name = filename.split('_aug')[0].split('_eval')[0]
                
                if base_name not in grouped_files:
                    grouped_files[base_name] = []
                grouped_files[base_name].append(pth_path)
                
            for base_name, file_list in grouped_files.items():
                samples.append({
                    'file_group': file_list,
                    'class_idx': class_idx,
                    'class_name': class_name
                })
        return samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample_info = self.samples[idx]
        
        if self.split == 'train':
            pth_path = random.choice(sample_info['file_group'])
        else:
            eval_files = [f for f in sample_info['file_group'] if '_eval' in f]
            if not eval_files:
                raise FileNotFoundError(f"Missing _eval file for {sample_info['class_name']}")
            pth_path = eval_files[0]
        
        # 1. Load and transform graph
        data = torch.load(pth_path)
        data.y = torch.tensor([sample_info['class_idx']], dtype=torch.long)
        data = self.graph_transform(data)
        
        # 2. Load multimodal image if required
        if self.modality == 'multimodal':
            img_path = pth_path.replace('.pth', '.jpg')
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image for graph: {pth_path}")
                
            image = Image.open(img_path).convert('RGB')
            data.x_img = self.image_transform(image)
            
        return data
