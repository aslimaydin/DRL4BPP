import torch
import networkx as nx
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.nn import GATConv, GINConv, GAE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
    

class GINEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_dim, hidden_dim))

        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_channels), 
            torch.nn.ReLU())

        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        return x

class BinPackingGAE(GAE):
    def __init__(self, in_channels, hidden_dim, out_channels):
        encoder = GATEncoder(in_channels, hidden_dim, out_channels)
        super().__init__(encoder)


class BinPackingGraph:
    def __init__(self, items, container_width, container_height):
        self.packing = []
        self.container_width = container_width
        self.container_height = container_height
        
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
            [[self.G_x.nodes[i]['pos'][0], self.G_x.nodes[i]['pos'][1], self.G_x.nodes[i]['size'][0], self.G_x.nodes[i]['size'][1]] 
            for i in range(num_nodes_x)], dtype=torch.float)

        edge_index_x = torch.tensor(list(self.G_x.edges), dtype=torch.long).t().contiguous()
        edge_index_y = torch.tensor(list(self.G_y.edges), dtype=torch.long).t().contiguous()

        edge_index = torch.cat([edge_index_x, edge_index_y], dim=1) if edge_index_x.numel() and edge_index_y.numel() else edge_index_x

        return Data(x=x, edge_index=edge_index, edge_index_x=edge_index_x, edge_index_y=edge_index_y)


def attribute_masking(masked_x):
    mask = torch.rand(data.x.shape, device=device) < mask_ratio
    masked_x[mask] = 0
    return masked_x


items_list = [
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
]

container_size = (10, 10)

epochs = 1
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = [BinPackingGraph(items, container_size[0], container_size[1]).nx_to_pyg() for items in items_list]
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

in_channels = 4
hidden_dim = 16
out_channels = 8
mask_ratio = 0.3

model = BinPackingGAE(in_channels, hidden_dim, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    # loop = tqdm(loader, leave=True, ncols=100)

    for data in loader:
        data = data.to(device)

        masked_x = attribute_masking(data.x.clone())

        print(masked_x)
        print(data.x)
        break

        # optimizer.zero_grad()
        
        # z = model.encode(data.x, data.edge_index)
        # print(z.shape, data.edge_index.shape)

        # loss = model.recon_loss(z, data.edge_index)
        # loss.backward()
        # optimizer.step()
        
        # total_loss += loss.item()
    
    # loop.postfix(epoch=epoch+1, loss=total_loss / len(loader))

torch.save(model.state_dict(), "data/gae_bin_packing.pth")