import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler

class EpisodicBatchSampler(Sampler):
    """
    Yields batches of indices for Few-Shot Learning episodes.
    Strictly guarantees that no two augmentations of the same base image 
    appear in the same episode, preventing data leakage.
    """
    def __init__(self, labels, base_names, n_way, k_shot, q_query, episodes_per_epoch):
        super().__init__(None)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        self.classes = np.unique(labels)
        
        # Build a nested dictionary: { class_idx: { base_name: [list_of_dataset_indices] } }
        self.class_to_base_to_indices = {c: defaultdict(list) for c in self.classes}
        
        for idx, (label, base_name) in enumerate(zip(labels, base_names)):
            self.class_to_base_to_indices[label][base_name].append(idx)
            
        # Keep a fast-lookup list of unique base names per class
        self.class_to_base_names = {
            c: list(self.class_to_base_to_indices[c].keys()) for c in self.classes
        }

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_indices = []
            query_indices = []
            
            # 1. Randomly sample classes for this episode
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            for c in selected_classes:
                base_names_for_class = self.class_to_base_names[c]
                samples_needed = self.k_shot + self.q_query
                
                if len(base_names_for_class) < samples_needed:
                    raise ValueError(f"Class {c} only has {len(base_names_for_class)} unique images, need {samples_needed}.")
                
                # 2. Randomly select unique BASE images (preventing leakage)
                selected_bases = np.random.choice(base_names_for_class, samples_needed, replace=False)
                
                selected_indices = []
                # 3. For each chosen base image, randomly pick EXACTLY ONE augmentation index
                for base in selected_bases:
                    aug_indices = self.class_to_base_to_indices[c][base]
                    chosen_aug_idx = np.random.choice(aug_indices)
                    selected_indices.append(chosen_aug_idx)
                
                # 4. Split into Support and Query
                support_indices.extend(selected_indices[:self.k_shot])
                query_indices.extend(selected_indices[self.k_shot:])
                
            yield support_indices + query_indices

    def __len__(self):
        return self.episodes_per_epoch
