import h5py
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split
from .graph_dataset import H5GraphDataset, GraphDataProcessor
import torch

class H5DynamicGraphDataset(H5GraphDataset):
    def __init__(self, h5_path, mode='train', train_ratio=0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.h5_path = h5_path
        self.sample_paths = []
        
        with h5py.File(h5_path, 'r') as hf:
            for iter_key in hf.keys():
                if iter_key.startswith('iter_'):
                    iter_grp = hf[iter_key]
                    self.sample_paths.extend([
                        f"{iter_key}/{round_key}" 
                        for round_key in iter_grp.keys()
                        if round_key.startswith('round_')
                    ])
            

        self.total_samples = len(self.sample_paths)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            sample_path = self.sample_paths[idx]
            group = hf[sample_path]
            
            features = torch.tensor(
                group['features'][:], 
                dtype=torch.float32, 
                device=self.device
            )

            edge_index = torch.from_numpy(
                group['edge_index'][:]
            ).long().contiguous()
            
            labels = torch.tensor(
                group['labels'][:], 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(-1)

            if torch.isnan(features).any() or torch.isnan(labels).any():
                raise ValueError(f"NaN in sample {sample_path}")

            return Data(
                x=features,
                edge_index=edge_index.to(self.device),
                y=labels
            )

class DynamicGraphDataProcessor(GraphDataProcessor):
    def prepare_datasets(self, train_ratio=0.8):
        full_dataset = H5DynamicGraphDataset(self.h5_path, mode='full')
        train_size = int(len(full_dataset) * train_ratio)
        eval_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, eval_size])