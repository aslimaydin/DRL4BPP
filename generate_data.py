import os
import shutil
import csv
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def generate_single_set_csv(set_id, data_dir, range_weights, range_values):
    n_weights = np.random.randint(range_weights[0], range_weights[1])
    weights = np.sort(np.random.uniform(low=range_values[0], high=range_values[1], size=n_weights)).astype(np.float32)

    data_path = os.path.join(data_dir, "data.csv")
    with open(data_path, 'a', newline='') as data_file:
        data_wr = csv.writer(data_file)
        data_wr.writerow(np.insert(weights, 0, set_id))

    sparse_path = os.path.join(data_dir, f"{set_id}.csv")
    with open(sparse_path, 'w', newline='') as sparse_file:
        sparse_wr = csv.writer(sparse_file)
        for i in range(n_weights):
            for j in range(i + 1, n_weights):
                normalized_sum = weights[i] + weights[j]
                if normalized_sum <= 1.0 and i != j:
                    sparse_wr.writerow([i, j, normalized_sum])
                    sparse_wr.writerow([j, i, normalized_sum])


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


def generate_data(data_dir="data", num_workers=os.cpu_count(), n_sets=1000, datatype="h5", range_weights=(100, 500), range_values=(0.01, 0.99)):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    os.mkdir(data_dir)

    if datatype == "csv":
        with open(os.path.join(data_dir, "train.csv"), 'w', newline='') as f:
            pass  

        print("Generating weights...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(generate_single_set_csv, 
                                range(n_sets), 
                                [data_dir] * n_sets, 
                                [range_weights] * n_sets, 
                                [range_values] * n_sets), total=n_sets, ncols=100))
    
        with open(os.path.join(data_dir, "test.csv"), 'w', newline='') as f:
            pass  
        
        n_sets = n_sets // 10

        print("Generating weights...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(generate_single_set_csv, 
                                range(n_sets), 
                                [data_dir] * n_sets, 
                                [range_weights] * n_sets, 
                                [range_values] * n_sets), total=n_sets, ncols=100))
    
    elif datatype == "h5":
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


if __name__ == "__main__":
    generate_data(n_sets=1000, range_weights=(100, 200))