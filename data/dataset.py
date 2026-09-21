# data/dataset.py
import os
import torch
from torch_geometric.data import Dataset
from PIL import Image


class MultimodalFSLDataset(Dataset):
    """
    Multimodal Few-Shot Learning Dataset.
    Integrates PyTorch Geometric graph structures with torchvision images.
    Supports on-the-fly toggling between coarsened and uncoarsened shock graphs.
    """
    def __init__(self, cfg, modality: str, split='train', vision_transform=None, graph_transform=None):
        super().__init__(root=None, transform=None, pre_transform=None)
        
        self.cfg = cfg
        self.split = split
        self.modality = modality
        self.split_dir = os.path.join(cfg.data_root, split)
        
        self.vision_transform = vision_transform
        self.graph_transform = graph_transform

        self.samples = self._load_and_group_files()
        
        self.labels = [sample['class_idx'] for sample in self.samples]
        self.base_names = [sample['base_name'] for sample in self.samples]

    def _find_image_path(self, class_path, pt_filename):
        """Universal image resolver: works with any graph suffix and image format."""
        if pt_filename.endswith("_coarse.pt"):
            stem = pt_filename[:-10]
        else:
            stem = pt_filename[:-3]

        # 1. Direct stem match (e.g. foobar.png or foobar.jpg)
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(class_path, stem + ext)
            if os.path.exists(p):
                return p

        # 2. Tagged match: strip trailing algorithm tag (e.g. 'dog_aug_00_se_tcg' -> 'dog_aug_00')
        if '_' in stem:
            clean_stem = stem.rsplit('_', 1)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(class_path, clean_stem + ext)
                if os.path.exists(p):
                    return p

        return None

    def _load_and_group_files(self):
        samples = []
        
        # Check whether to load coarsened or uncoarsened graphs
        use_coarse = False
        if hasattr(self.cfg, 'graph') and hasattr(self.cfg.graph, 'use_coarse'):
            use_coarse = bool(self.cfg.graph.use_coarse)
        elif hasattr(self.cfg, 'use_coarse'):
            use_coarse = bool(self.cfg.use_coarse)

        with os.scandir(self.split_dir) as dir_entries:
            class_folders = sorted([e.name for e in dir_entries if e.is_dir()])

        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.split_dir, class_name)
            
            with os.scandir(class_path) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                        
                    fname = entry.name
                    
                    # Generic filter: coarse vs uncoarse by exclusion
                    if use_coarse:
                        if not fname.endswith("_coarse.pt"):
                            continue
                    else:
                        if not (fname.endswith(".pt") and not fname.endswith("_coarse.pt")):
                            continue

                    pt_path = entry.path

                    # Resolve image path
                    img_path = self._find_image_path(class_path, fname)
                    if self.modality in ['multimodal', 'vision'] and (img_path is None or not os.path.exists(img_path)):
                        raise FileNotFoundError(f"Could not find an image for graph: {pt_path}")

                    # Determine base sample ID for Few-Shot episodic sampler
                    if '_aug_' in fname:
                        base_name = fname.split('_aug_')[0]
                    else:
                        base_name = fname.split('_')[0].split('.')[0]

                    samples.append({
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'base_name': base_name,
                        'graph_path': pt_path,
                        'image_path': img_path
                    })

        samples.sort(key=lambda s: s['graph_path'])
        print(f"[{self.split.upper()} Set] Indexed {len(samples):,} graphs (Mode: {'COARSE' if use_coarse else 'UNCOARSE'}).")
        return samples
    
    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]
        
        # --- A. Load Graph Data ---
        data = torch.load(sample['graph_path'], weights_only=False)
        data.y = torch.tensor([sample['class_idx']], dtype=torch.long)
        
        if self.graph_transform is not None:
            data = self.graph_transform(data)
            
        # --- B. Load Vision Data ---
        if self.modality in ['multimodal', 'vision']:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.vision_transform is not None:
                data.x_img = self.vision_transform(image)
            else:
                raise ValueError("vision_transform cannot be None when modality includes vision.")
                
        return data
