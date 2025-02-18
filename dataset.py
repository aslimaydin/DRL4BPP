import shutil
import torch
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def generate_single_set_h5py(set_id, range_weights, range_values):
    n_weights = np.random.randint(range_weights[0], range_weights[1])
    weights = np.sort(np.random.uniform(low=range_values[0], high=range_values[1], size=n_weights)).astype(np.float32)

    adjacency_list = []
    for i in range(n_weights):
        for j in range(i + 1, n_weights):
            normalized_sum = weights[i] + weights[j]
            if normalized_sum <= 1.0 and i != j:
                adjacency_list.append([i, j, normalized_sum])
                adjacency_list.append([j, i, normalized_sum])

    return set_id, weights, np.array(adjacency_list, dtype=np.float32)


def generate_data_1D(data_dir="data", num_workers=os.cpu_count(), n_sets=1000, range_weights=(100, 500), range_values=(0.01, 0.99)):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    os.mkdir(data_dir)

    with h5py.File(os.path.join(data_dir, "train.h5"), "w") as h5file:
        grp_weights = h5file.create_group("weights")
        grp_adj = h5file.create_group("adjacency")

        print("Generating and saving data...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for set_id, weights, adjacency in tqdm(
                executor.map(generate_single_set_h5py, 
                            range(n_sets), 
                            [range_weights] * n_sets, 
                            [range_values] * n_sets), total=n_sets, ncols=100):

                grp_weights.create_dataset(str(set_id), data=weights)
                grp_adj.create_dataset(str(set_id), data=adjacency)
    
    n_sets = n_sets // 10

    with h5py.File(os.path.join(data_dir, "test.h5"), "w") as h5file:
        grp_weights = h5file.create_group("weights")
        grp_adj = h5file.create_group("adjacency")

        print("Generating and saving data...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for set_id, weights, adjacency in tqdm(
                executor.map(generate_single_set_h5py, 
                            range(n_sets), 
                            [range_weights] * n_sets, 
                            [range_values] * n_sets), total=n_sets, ncols=100):

                grp_weights.create_dataset(str(set_id), data=weights)
                grp_adj.create_dataset(str(set_id), data=adjacency)


class BinPackingDataset(Dataset):
    def __init__(self, filename=os.path.join("data", "train.h5")):
        self.h5 = h5py.File(filename, "r")
        self.instances = list(self.h5["weights"].keys())
    
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
    generate_data_1D(n_sets=1000, range_weights=(100, 200))

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