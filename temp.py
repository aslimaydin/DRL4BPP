import torch
import torch.nn.functional as F
import networkx as nx

from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
    

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
                    
        sorted_items = sorted(items, key=lambda x: (-x[1], -x[0]))
        x_cursor, y_cursor = 0, 0
        max_row_height = 0
        
        for w, h in sorted_items:
            if x_cursor + w > self.container_width:
                x_cursor = 0
                y_cursor += max_row_height
                max_row_height = 0
            
            self.packing.append((x_cursor, y_cursor, w, h))
            x_cursor += w
            max_row_height = max(max_row_height, h)

    def construct_graphs(self):
        for i, (x1, y1, w1, h1) in enumerate(self.packing):
            self.G_x.add_node(i, pos=(x1, y1), size=(w1, h1))
            self.G_y.add_node(i, pos=(x1, y1), size=(w1, h1))

            x1_end = x1 + w1
            y1_end = y1 + h1
            
            for j, (x2, y2, w2, h2) in enumerate(self.packing):
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


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1_x = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()))
        
        self.conv2_x = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()))
        
        self.conv1_y = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()))
        
        self.conv2_y = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()))
        
        self.lin1 = nn.Linear(hidden_channels*4, hidden_channels*2)
        self.lin2 = nn.Linear(hidden_channels*2, out_channels)

    def forward(self, data):
        x, edge_index_x, edge_index_y, batch = data.x, data.edge_index_x, data.edge_index_y, data.batch

        h1_x = self.conv1_x(x, edge_index_x)
        h2_x = self.conv2_x(h1_x, edge_index_x)

        h1_y = self.conv1_y(x, edge_index_y)
        h2_y = self.conv2_y(h1_y, edge_index_y)

        h1_x = global_add_pool(h1_x, batch)
        h2_x = global_add_pool(h2_x, batch)
        h1_y = global_add_pool(h1_y, batch)
        h2_y = global_add_pool(h2_y, batch)

        h = torch.cat((h1_x, h2_x, h1_y, h2_y), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)
    

items_list = [
    [(3, 2), (5, 3), (2, 4), (4, 2), (3, 3)],
    [(2, 2), (3, 4), (4, 2), (1, 5), (3, 3)],
    [(8, 7), (7, 2), (6, 6), (9, 1), (7, 4)],
    [(6, 1), (6, 4), (3, 5), (4, 8), (3, 7)],
]

container_width = 10

dataset = []
for items in items_list:
    bp = BinPackingGraph(items, container_width)
    data = bp.nx_to_pyg()
    dataset.append(data)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = GIN(in_channels=4, hidden_channels=16, out_channels=8).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# for data in loader:
#     data = data.to('cuda')
#     print(data)
#     embeddings = model(data)
#     print(embeddings)