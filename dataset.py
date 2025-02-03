import torch
import os
import csv
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

# class BinPackingDataset(Dataset):
#     def __init__(self, data_dir = "data"):
#         self.data_dir = data_dir
#         self.instances = []
        
#         with open(os.path.join(data_dir, "data.csv"), 'r') as f:
#             csv_reader = csv.reader(f)
#             for row in csv_reader:
#                 instance_id = int(float(row[0]))
#                 weights = np.array([float(x) for x in row[1:]], dtype=np.float32)
#                 self.instances.append((instance_id, weights))
    
#     def __len__(self):
#         return len(self.instances)
    
#     def load_sparse_adj(self, instance_id, n_items):
#         indices_list = []
#         values_list = []
        
#         with open(os.path.join(self.data_dir, f"{instance_id}.csv"), 'r') as f:
#             csv_reader = csv.reader(f)
#             for row in csv_reader:
#                 i, j, value = map(float, row)
#                 indices_list.append([int(i), int(j)])
#                 values_list.append(value)
        
#         if indices_list:  # If there are any entries
#             indices = torch.tensor(indices_list, dtype=torch.long).t()
#             values = torch.tensor(values_list, dtype=torch.float32)
#         else:  # Empty adjacency matrix
#             indices = torch.empty((2, 0), dtype=torch.long)
#             values = torch.empty(0, dtype=torch.float32)
        
#         size = (n_items, n_items)
#         return torch.sparse_coo_tensor(indices, values, size)
    
#     def __getitem__(self, idx):
#         instance_id, weights = self.instances[idx]
#         weights_tensor = torch.tensor(weights, dtype=torch.float32)
#         n_items = len(weights)
        
#         adj_matrix = self.load_sparse_adj(instance_id, n_items)
        
#         return weights_tensor, adj_matrix, n_items

class BinPackingDataset(Dataset):
    def __init__(self, filename=os.path.join("data", "train.h5")):
        self.h5 = h5py.File(filename, "r")
        self.instances = list(self.h5["weights"].keys())  # Instance IDs are stored as string keys
    
    def __len__(self):
        return len(self.instances)
    
    def load_sparse_adj(self, instance_id, n_items):
        if instance_id not in self.h5["adjacency"]:
            return torch.sparse_coo_tensor(([], []), [], (n_items, n_items))

        adjacency_data = self.h5["adjacency"][instance_id][:]
        
        if adjacency_data.size == 0:
            return torch.sparse_coo_tensor(([], []), [], (n_items, n_items))

        indices_list = adjacency_data[:, :2].astype(np.int64)
        values_list = adjacency_data[:, 2]

        indices = torch.tensor(indices_list.T, dtype=torch.long)
        values = torch.tensor(values_list, dtype=torch.float32)

        return torch.sparse_coo_tensor(indices, values, (n_items, n_items))
    
    def __getitem__(self, idx):
        instance_id = self.instances[idx]
        
        weights = self.h5["weights"][instance_id][:]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        n_items = len(weights)
        
        adj_matrix = self.load_sparse_adj(instance_id, n_items)

        return weights_tensor, adj_matrix, n_items
    
    def close(self):
        self.h5.close()


def collate_binpacking(batch):
    max_items = max(items for _, _, items in batch)
    
    weights_batch = []
    adj_batch = []
    mask_batch = []
    
    for weights, adj, n_items in batch:
        padded_weights = torch.zeros(max_items, dtype=torch.float32)
        padded_weights[:n_items] = weights
        weights_batch.append(padded_weights)
        
        padded_adj = torch.zeros((max_items, max_items), dtype=torch.float32)
        padded_adj[:n_items, :n_items] = adj.to_dense()
        adj_batch.append(padded_adj)
        
        mask = torch.zeros(max_items, dtype=torch.bool)
        mask[:n_items] = True
        mask_batch.append(mask)
    
    weights_batch = torch.stack(weights_batch)
    adj_batch = torch.stack(adj_batch)
    mask_batch = torch.stack(mask_batch)
    
    return weights_batch.unsqueeze(-1), adj_batch, mask_batch


if __name__ == "__main__":
    dataset = BinPackingDataset("data")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_binpacking,
        pin_memory=torch.cuda.is_available()
    )

    for weights, adj, mask in loader:
        print(weights)
        print(adj)
        break
        # print(f"Weights: {weights.shape}")
        # print(f"Adjacency: {adj.shape}")
        # print(f"Mask: {mask.shape}")
        # print(f"Valid items per instance: {mask.sum(dim=1)}")