import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from random import shuffle
from matplotlib.patches import Rectangle


class BinPacking2D(Dataset):
    def __init__(self, file_name):
        self.items = []
        self.packing = []
        self.bins_info = []

        self.process_data(file_name)
        
    def process_data(self, file_name):
        with open(file_name, 'r') as f:
            group = []
            items = []
            bin_size = None

            for line in f:
                stripped = line.strip()
                if stripped:
                    values = list(map(float, stripped.split()))
                    if len(values) == 3:
                        n_bins = values[0]
                        bin_size = values[1:]
                        self.bins_info.append((n_bins, *bin_size))
                    else:
                        if bin_size is None:
                            raise ValueError("Bin size must be defined before normalizing values.")

                        normalized_values = [
                            values[0] / bin_size[0],
                            values[1] / bin_size[1],
                            values[2] / bin_size[0],
                            values[3] / bin_size[1]
                        ]

                        group.append(normalized_values)
                        items.append(normalized_values[:2])
                else:
                    if group:
                        self.items.append(items)
                        self.packing.append(group)

                        group = []
                        items = []

            if group:
                self.items.append(items)
                self.packing.append(group)
        
    def construct_graphs(self, packing):
        G_x = nx.Graph()
        G_y = nx.Graph()

        for i, (x1, y1, w1, h1) in enumerate(packing):
            G_x.add_node(i, pos=(x1, y1), size=(w1, h1))
            G_y.add_node(i, pos=(x1, y1), size=(w1, h1))

            x1_end = x1 + w1
            y1_end = y1 + h1

            for j, (x2, y2, w2, h2) in enumerate(packing):
                if i == j:
                    continue
                
                x2_end = x2 + w2
                y2_end = y2 + h2

                if not (x1_end <= x2 or x2_end <= x1):
                    G_y.add_edge(i, j)
                
                if not (y1_end <= y2 or y2_end <= y1):
                    G_x.add_edge(i, j)

        return G_x, G_y

    def nx_to_pyg(self, G_x, G_y, indices):
        num_nodes = G_x.number_of_nodes()
        x = torch.tensor(
            [
                [G_x.nodes[i]['pos'][0], G_x.nodes[i]['pos'][1], 
                 G_x.nodes[i]['size'][0], G_x.nodes[i]['size'][1]] for i in range(num_nodes)
            ], dtype=torch.float
        )

        edge_index_x = torch.tensor(list(G_x.edges), dtype=torch.long).t().contiguous()
        edge_index_y = torch.tensor(list(G_y.edges), dtype=torch.long).t().contiguous()
        indices = torch.tensor(indices, dtype=torch.long)

        return Data(x=x, edge_index_x=edge_index_x, edge_index_y=edge_index_y, indices=indices)

    def random_packing(self, items):
        packing = []
        indexed_items = list(enumerate(items))
        shuffle(indexed_items)

        x_cursor, y_cursor = 0, 0
        max_row_height = 0
        new_indices = []

        for original_index, (w, h) in indexed_items:
            if x_cursor + w > 1.0:
                x_cursor = 0
                y_cursor += max_row_height
                max_row_height = 0

            packing.append((x_cursor, y_cursor, w, h))
            new_indices.append(original_index)

            x_cursor += w
            max_row_height = max(max_row_height, h)

        return packing, new_indices

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label_packing = self.packing[idx]
        indices = range(len(label_packing))
        G_x, G_y = self.construct_graphs(label_packing)

        random_packing, random_indices = self.random_packing(self.items[idx])
        random_G_x, random_G_y = self.construct_graphs(random_packing)

        return self.nx_to_pyg(G_x, G_y, indices), self.nx_to_pyg(random_G_x, random_G_y, random_indices)


class BinPackingGraph:
    def __init__(self, items, container_width):
        self.packing = []
        self.container_width = container_width
        
        self.G_x = nx.Graph()
        self.G_y = nx.Graph()

        self.bottom_left_packing(items)
        self.construct_graphs()
        
    def bottom_left_packing(self, items):
        is_packable = False
        for i, (w1, _) in enumerate(items):
            for j, (w2, _) in enumerate(items):
                if i == j:
                    continue

                if w1 + w2 <= self.container_width:
                    is_packable = True
                    break
                
        if not is_packable:
            for i, (_, h1) in enumerate(items):
                for j, (w2, _) in enumerate(items):
                    if i == j:
                        continue

                    if h1 + w2 <= self.container_width:
                        items[i] = (items[i][1], items[i][0])
                        is_packable = True
                        break
                
                if is_packable:
                    break

        sorted_items_with_indices = sorted(enumerate(items), key=lambda x: (-x[1][1], -x[1][0]))
        x_cursor, y_cursor = 0, 0
        max_row_height = 0
        
        for i, (w, h) in sorted_items_with_indices:
            if x_cursor + w > self.container_width:
                x_cursor = 0
                y_cursor += max_row_height
                max_row_height = 0
            
            self.packing.append((i, x_cursor, y_cursor, w, h))
            x_cursor += w
            max_row_height = max(max_row_height, h)

    def random_packing(self, items):
        is_packable = False
        for i, (w1, _) in enumerate(items):
            for j, (w2, _) in enumerate(items):
                if i == j:
                    continue

                if w1 + w2 <= self.container_width:
                    is_packable = True
                    break
                
        if not is_packable:
            for i, (_, h1) in enumerate(items):
                for j, (w2, _) in enumerate(items):
                    if i == j:
                        continue

                    if h1 + w2 <= self.container_width:
                        items[i] = (items[i][1], items[i][0])
                        is_packable = True
                        break
                
                if is_packable:
                    break
        
        shuffled_items = items.copy()
        shuffle(shuffled_items)
        x_cursor, y_cursor = 0, 0
        max_row_height = 0

        for w, h in shuffled_items:
            if x_cursor + w > self.container_width:
                x_cursor = 0
                y_cursor += max_row_height
                max_row_height = 0
            
            self.packing.append((x_cursor, y_cursor, w, h))
            x_cursor += w
            max_row_height = max(max_row_height, h)
        
    def construct_graphs(self):
        for i, x1, y1, w1, h1 in self.packing:
            self.G_x.add_node(i, pos=(x1, y1), size=(w1, h1))
            self.G_y.add_node(i, pos=(x1, y1), size=(w1, h1))

            x1_end = x1 + w1
            y1_end = y1 + h1
            
            for j, x2, y2, w2, h2 in self.packing:
                if i == j:
                    continue
                
                x2_end = x2 + w2
                y2_end = y2 + h2

                if not (x1_end <= x2 or x2_end <= x1):
                    self.G_y.add_edge(i, j)
                
                if not (y1_end <= y2 or y2_end <= y1):
                    self.G_x.add_edge(i, j)

    def nx_to_pyg(self):
        num_nodes_x = self.G_x.number_of_nodes()
        x = torch.tensor(
            [[self.G_x.nodes[i]['pos'][0], self.G_x.nodes[i]['pos'][1], self.G_x.nodes[i]['size'][0], self.G_x.nodes[i]['size'][1]] for i in range(num_nodes_x)], 
            dtype=torch.float)

        edge_index_x = torch.tensor(list(self.G_x.edges), dtype=torch.long).t().contiguous()
        edge_index_y = torch.tensor(list(self.G_y.edges), dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index_x=edge_index_x, edge_index_y=edge_index_y)

    def plot_packing(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, x, y, w, h in self.packing:
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)

            if x == int(x): x = int(x)
            if y == int(y): y = int(y)
            if w == int(w): w = int(w)
            if h == int(h): h = int(h)
            ax.text(x + w / 2, y + h / 2, f'({w}, {h})', fontsize=6, ha='center', va='center', color='black')

        ax.set_xlim(0, max(x + w for _, x, _, w, _ in self.packing) + 1)
        ax.set_ylim(0, max(y + h for _, _, y, _, h in self.packing) + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Packed Rectangles")
        ax.set_aspect('equal')

        fig.set_size_inches(18.5, 10.5)
        plt.show()
        plt.close()

    def plot_graphs(self):
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        nx.draw(self.G_x, with_labels=True)
        plt.title("G_x")

        plt.axvline(x = 5, color = 'b')
        
        plt.subplot(1, 2, 2)
        nx.draw(self.G_y, with_labels=True)
        plt.title("G_y")

        fig.set_size_inches(18.5, 10.5)
        plt.show()
        plt.close()


if __name__ == "__main__":
    dataset = BinPacking2D("data/gcut/dataset.txt")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # for labels, inputs in loader:
    #     print(labels.edge_index_x)
    #     print(inputs.edge_index_x)
    #     break