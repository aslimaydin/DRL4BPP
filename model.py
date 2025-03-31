import torch
from torch_geometric.nn import GATConv, GINConv


class BinPackingGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(BinPackingGAT, self).__init__()

        self.gnn_x1 = GATConv(in_channels, hidden_dim)
        self.gnn_x2 = GATConv(hidden_dim, hidden_dim)

        self.gnn_y1 = GATConv(in_channels, hidden_dim)
        self.gnn_y2 = GATConv(hidden_dim, hidden_dim)

        self.fc = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index_x, edge_index_y, indices = data.x, data.edge_index_x, data.edge_index_y, data.indices

        x_x = self.gnn_x1(x, edge_index_x).relu()
        x_x = self.gnn_x2(x_x, edge_index_x).relu()

        x_y = self.gnn_y1(x, edge_index_y).relu()
        x_y = self.gnn_y2(x_y, edge_index_y).relu()

        x_combined = torch.cat([x_x, x_y], dim=1)

        output = self.fc(x_combined).relu()
        output = self.out(output)

        return output[indices]


if __name__ == "__main__":
    from dataset import *

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BinPacking2D("data/gcut/dataset.txt")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = BinPackingGAT(4, 256, 2).to(device)

    for labels, inputs in loader:
        labels = labels.to(device)
        inputs = inputs.to(device)

        predictions = model(inputs)

        print(inputs)
        print(labels.x[:, :2])
        print(predictions.shape, '\n')