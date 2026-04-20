import numpy as np
from torch.utils.data import Sampler

class EpisodicBatchSampler(Sampler):
    """
    Yields batches of indices corresponding to Few-Shot Learning episodes.
    Ensures that the output batch is strictly ordered: 
    All Support images (grouped by class), followed by All Query images (grouped by class).
    """
    def __init__(self, labels, n_way, k_shot, q_query, episodes_per_epoch):
        super().__init__(None)
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch

        # Identify all unique classes in the dataset
        self.classes = np.unique(self.labels)
        
        # Map each class to a list of its corresponding data indices
        self.class_to_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            support_indices = []
            query_indices = []
            
            # 1. Randomly sample 'n_way' classes for this episode
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            for c in selected_classes:
                class_indices = self.class_to_indices[c]
                samples_needed = self.k_shot + self.q_query
                
                # Safety check: Ensure the class has enough data
                if len(class_indices) < samples_needed:
                    raise ValueError(f"Class {c} only has {len(class_indices)} samples, but {samples_needed} are needed.")
                
                # 2. Randomly select the required number of images for this class
                selected_indices = np.random.choice(class_indices, samples_needed, replace=False)
                
                # 3. CRITICAL FIX: Split indices into Support and Query BEFORE aggregating
                support_indices.extend(selected_indices[:self.k_shot].tolist())
                query_indices.extend(selected_indices[self.k_shot:].tolist())
                
            # 4. Combine so the model can safely slice the first (n_way * k_shot) elements
            episode_indices = support_indices + query_indices
            
            # Yield the final list of indices to the DataLoader
            yield episode_indices

    def __len__(self):
        return self.episodes_per_epoch
