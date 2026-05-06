import os
import glob
import torch
from torch_geometric.data import Dataset, Data
from PIL import Image

class MultimodalFSLDataset(Dataset):
    """
    Multimodal Few-Shot Learning Dataset.
    Integrates PyTorch Geometric graph data with standard torchvision images.
    """
    def __init__(self, cfg, modality: str, split='train', vision_transform=None, graph_transform=None):
        # Pass None to PyG's superclass so we can manually handle our custom transforms
        super().__init__(root=None, transform=None, pre_transform=None)
        
        self.cfg = cfg
        self.split = split
        self.modality = modality
        self.split_dir = os.path.join(cfg.data_root, split)
        
        # Map the injected transforms directly
        self.vision_transform = vision_transform
        self.graph_transform = graph_transform

        # Load file paths and build the dataset index
        self.samples = self._load_and_group_files()
        
        # Expose a flat list of labels for the EpisodicBatchSampler to use!
        self.labels = [sample['class_idx'] for sample in self.samples]

    def _load_and_group_files(self):
        """
        Scans the directory structure to pair images with their corresponding graph files.
        Enforces a strict 1-to-1 mapping based on the offline augmentation strategy.
        """
        samples = []
        class_folders = sorted([d for d in os.listdir(self.split_dir) 
                                if os.path.isdir(os.path.join(self.split_dir, d))])
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.split_dir, class_name)
            
            # 1. Find all .pt graph files (e.g., dog_001_aug_00.pt)
            pt_files = glob.glob(os.path.join(class_path, "*.pt"))
            
            for pt_path in pt_files:
                # 2. Swap the extension to find the exact matching augmented image
                image_path = pt_path.replace('.pt', '.jpg')
                
                samples.append({
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'graph_path': pt_path,   # The .pt file
                    'image_path': image_path # The perfectly aligned .jpg file
                })
                
        return samples
    
    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]
        
        # --- A. Load Graph Data ---
        # Load the pre-packaged PyG Data object directly from the single file
        data = torch.load(sample['graph_path'], weights_only=False)
        
        # Explicitly set/override the dataset label for your Few-Shot Sampler
        data.y = torch.tensor([sample['class_idx']], dtype=torch.long)
        
        # Apply graph transform if provided
        if self.graph_transform is not None:
            data = self.graph_transform(data)
            
        # --- B. Load Vision Data ---
        if self.modality in ['multimodal', 'vision']:
            if not os.path.exists(sample['image_path']):
                raise FileNotFoundError(f"Image not found at {sample['image_path']}")
                
            image = Image.open(sample['image_path']).convert('RGB')
            
            if self.vision_transform is not None:
                data.x_img = self.vision_transform(image)
            else:
                raise ValueError("vision_transform cannot be None if modality includes vision.")
                
        return data
