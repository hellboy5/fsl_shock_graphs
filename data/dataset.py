# data/dataset.py
import os
import torch
from torch_geometric.data import Dataset
from PIL import Image


class MultimodalFSLDataset(Dataset):
    """
    High-Speed Multimodal Few-Shot Learning Dataset.
    
    Optimizations:
      - In-memory set resolution: Eliminates slow network stat/exists calls on HPC storage.
      - Modality-aware bypass: Skips image discovery completely when modality == 'graph'.
      - Suffix-agnostic pairing: Robust to arbitrary graph naming conventions and image extensions.
    """
    def __init__(self, cfg, modality: str, split='train', vision_transform=None, graph_transform=None):
        super().__init__(root=None, transform=None, pre_transform=None)
        
        self.cfg = cfg
        self.split = split
        self.modality = modality
        self.split_dir = os.path.join(cfg.data_root, split)
        
        self.vision_transform = vision_transform
        self.graph_transform = graph_transform

        # Load file paths and index the split
        self.samples = self._load_and_group_files()
        
        # Expose flat lists required by EpisodicBatchSampler
        self.labels = [sample['class_idx'] for sample in self.samples]
        self.base_names = [sample['base_name'] for sample in self.samples]

    def _resolve_image_name(self, pt_filename, class_files_set):
        """In-memory resolution: checks RAM set rather than issuing network stat calls."""
        # 1. Augmented datasets: direct prefix + aug token extraction
        if '_aug_' in pt_filename:
            prefix, rest = pt_filename.split('_aug_', 1)
            aug_idx = rest.split('_')[0]
            base_img = f"{prefix}_aug_{aug_idx}"
            for ext in ['.png', '.jpg', '.jpeg']:
                if (base_img + ext) in class_files_set:
                    return base_img + ext

        # 2. Strip graph extensions (_coarse.pt or .pt)
        if pt_filename.endswith("_coarse.pt"):
            stem = pt_filename[:-10]
        else:
            stem = pt_filename[:-3]

        # 3. Direct match (e.g. foobar.png or foobar.jpg)
        for ext in ['.png', '.jpg', '.jpeg']:
            if (stem + ext) in class_files_set:
                return stem + ext

        # 4. Multi-tag stripping fallback (e.g. foobar_se_tcg -> foobar_se -> foobar)
        parts = stem.split('_')
        for i in range(len(parts) - 1, 0, -1):
            candidate_stem = "_".join(parts[:i])
            for ext in ['.png', '.jpg', '.jpeg']:
                if (candidate_stem + ext) in class_files_set:
                    return candidate_stem + ext

        return None

    def _load_and_group_files(self):
        samples = []
        
        # Check whether to load coarsened or uncoarsened graphs
        use_coarse = False
        if hasattr(self.cfg, 'graph') and hasattr(self.cfg.graph, 'use_coarse'):
            use_coarse = bool(self.cfg.graph.use_coarse)
        elif hasattr(self.cfg, 'use_coarse'):
            use_coarse = bool(self.cfg.use_coarse)

        mode_str = "COARSE" if use_coarse else "UNCOARSE"

        with os.scandir(self.split_dir) as dir_entries:
            class_folders = sorted([e.name for e in dir_entries if e.is_dir()])

        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.split_dir, class_name)
            
            # Read directory entries into memory ONCE
            file_names = os.listdir(class_path)
            class_files_set = set(file_names)

            for fname in file_names:
                # Suffix filtering: coarse vs uncoarse by negative exclusion
                if use_coarse:
                    if not fname.endswith("_coarse.pt"):
                        continue
                else:
                    if not (fname.endswith(".pt") and not fname.endswith("_coarse.pt")):
                        continue

                pt_path = os.path.join(class_path, fname)

                # Only resolve images if the modality requires them
                img_path = None
                if self.modality in ['multimodal', 'vision']:
                    img_name = self._resolve_image_name(fname, class_files_set)
                    if img_name is not None:
                        img_path = os.path.join(class_path, img_name)
                    else:
                        raise FileNotFoundError(f"Could not find matching image for {pt_path}")

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

        # Deterministic sort for multi-worker DataLoader reproducibility
        samples.sort(key=lambda s: s['graph_path'])
        
        print(f"[{self.split.upper()} Set] Indexed {len(samples):,} graphs (Mode: {mode_str}).")
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
            if sample['image_path'] is None or not os.path.exists(sample['image_path']):
                raise FileNotFoundError(f"Image not found at {sample['image_path']}.")
                
            image = Image.open(sample['image_path']).convert('RGB')
            if self.vision_transform is not None:
                data.x_img = self.vision_transform(image)
            else:
                raise ValueError("vision_transform cannot be None when modality includes vision.")
                
        return data
