# data/samplers.py
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class EpisodicBatchSampler(Sampler):
    """
    Leakage-Proof Hierarchical Episodic Sampler for Few-Shot Learning.
    
    Guarantees:
      1. Zero Base-Image Collisions: Enforces that no two augmentations of the 
         same physical base image appear in the same episode (strictly disjoint 
         Support and Query sets).
      2. Dynamic Augmentation Handling: Safely handles images with varying numbers 
         of augmentations (e.g. if extraction errors left some images with 6 or 8 
         instead of 10).
      3. Deterministic Evaluation: When evaluated on val/test splits with only 1 
         canonical image (_aug_00), it deterministically collapses to standard 
         few-shot episodic benchmarking.
      4. Safe Way-Clamping: Prevents crashes if n_way exceeds the available class 
         count in a split (e.g., in validation sets with only 16 classes).
    """
    def __init__(self, labels, base_names, n_way, k_shot, q_query, episodes_per_epoch):
        super().__init__(None)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        self.classes = np.unique(labels)
        
        # Build nested mapping: { class_idx: { base_name: [list_of_dataset_indices] } }
        self.class_to_base_to_indices = {c: defaultdict(list) for c in self.classes}
        for idx, (label, base_name) in enumerate(zip(labels, base_names)):
            self.class_to_base_to_indices[label][base_name].append(idx)
            
        # Fast-lookup list of unique base objects per class
        self.class_to_base_names = {
            c: list(self.class_to_base_to_indices[c].keys()) for c in self.classes
        }

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_indices = []
            query_indices = []
            
            # 1. Safely clamp n_way to available classes (prevents val-split crashes)
            effective_n_way = min(self.n_way, len(self.classes))
            selected_classes = np.random.choice(self.classes, effective_n_way, replace=False)
            
            for c in selected_classes:
                base_names_for_class = self.class_to_base_names[c]
                samples_needed = self.k_shot + self.q_query
                
                if len(base_names_for_class) < samples_needed:
                    raise ValueError(
                        f"Class {c} only has {len(base_names_for_class)} unique base images, "
                        f"but {samples_needed} are required per episode!"
                    )
                
                # 2. Draw distinct BASE images without replacement (Zero physical collisions)
                selected_bases = np.random.choice(
                    base_names_for_class, 
                    samples_needed, 
                    replace=False
                )
                
                # 3. For each unique base image, dynamically draw ONE available augmentation
                selected_indices = []
                for base in selected_bases:
                    aug_indices = self.class_to_base_to_indices[c][base]
                    # Uniformly picks 1 from whichever augmentations successfully exist on disk
                    chosen_aug_idx = np.random.choice(aug_indices)
                    selected_indices.append(chosen_aug_idx)
                
                # 4. Partition into Support and Query
                support_indices.extend(selected_indices[:self.k_shot])
                query_indices.extend(selected_indices[self.k_shot:])
                
            # Yield batch: [Support_Class1... Support_ClassN, Query_Class1... Query_ClassN]
            yield support_indices + query_indices

    def __len__(self):
        return self.episodes_per_epoch
