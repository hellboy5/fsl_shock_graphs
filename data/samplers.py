# data/samplers.py
import numpy as np
from torch.utils.data import Sampler

class EpisodicBatchSampler(Sampler):
    def __init__(self, dataset, n_way, k_shot, q_query, episodes_per_epoch):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        
        self.class_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            c = sample['class_idx']
            if c not in self.class_to_indices:
                self.class_to_indices[c] = []
            self.class_to_indices[c].append(idx)
            
        self.classes = list(self.class_to_indices.keys())

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            episode_indices = []
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            for c in selected_classes:
                class_indices = self.class_to_indices[c]
                samples_needed = self.k_shot + self.q_query
                
                if len(class_indices) < samples_needed:
                    raise ValueError(f"Class {c} only has {len(class_indices)} unique samples.")
                
                selected_indices = np.random.choice(class_indices, samples_needed, replace=False)
                episode_indices.extend(selected_indices.tolist())
                
            yield episode_indices

    def __len__(self):
        return self.episodes_per_epoch
