import h5py
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset, random_split
import torch

class H5GraphDataset(Dataset):
    def __init__(self, h5_path, mode='train', train_ratio=0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.h5_path = h5_path
        
        with h5py.File(h5_path, 'r') as hf:
            self.total_samples = hf.attrs['total_samples']
        
        if mode == 'full':
            self.indices = np.arange(self.total_samples)
        else:
            indices = np.random.permutation(self.total_samples)
            split_idx = int(self.total_samples * train_ratio)
            self.indices = indices[:split_idx] if mode == 'train' else indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            sample_id = self.indices[idx]
            group = hf[f'sample_{sample_id}']
            
            node_ds = group['node_features']
            node_matrix = node_ds[:]
            x = torch.tensor(node_matrix, dtype=torch.float32, device=self.device)
            
            edge_array = group['edge_index'][:]
            edge_index = torch.from_numpy(edge_array).long()
            if edge_index.dim() == 1:
                edge_index = edge_index.view(2, -1)
            
            labels = torch.from_numpy(group['labels'][:]).float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)

            if torch.isnan(x).any() or torch.isnan(labels).any():
                raise ValueError(f"NaN Data in {sample_id}")
                
            return Data(
                x=x.to(self.device),
                edge_index=edge_index.to(self.device),
                y=labels.to(self.device),
            )

class GraphDataProcessor:    
    def __init__(self, h5_path):
        self.h5_path = h5_path
        
    def prepare_datasets(self, train_ratio=0.8):
        full_dataset = H5GraphDataset(self.h5_path, mode='full')
        train_size = int(len(full_dataset) * train_ratio)
        eval_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, eval_size])
