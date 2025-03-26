import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.conv1_x = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()))
        
        self.conv1_y = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()))
        
        self.conv2_x = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()))
        
        self.conv2_y = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()))
        
        self.conv3_x = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels), 
                nn.ReLU()))
        
        self.conv3_y = GINConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels), 
                nn.ReLU()))
        
        self.lin1 = nn.Linear(hidden_channels*3*2, hidden_channels*3)
        self.lin2 = nn.Linear(hidden_channels*3, out_channels)

    def forward(self, data):
        x, edge_index_x, edge_index_y, batch = data.x, data.edge_index_x, data.edge_index_y, data.batch
        print(x.shape, edge_index_x.shape, edge_index_y.shape)
        if edge_index_x.numel() == 0:
            h1_x = x.clone()
            h2_x = x.clone()
            h3_x = x.clone()
        else:
            h1_x = self.conv1_x(x, edge_index_x)
            h2_x = self.conv2_x(h1_x, edge_index_x)
            h3_x = self.conv3_x(h2_x, edge_index_x)

        if edge_index_y.numel() == 0:
            h1_y = x.clone()
            h2_y = x.clone()
            h3_y = x.clone()
        else:
            h1_y = self.conv1_y(x, edge_index_y)
            h2_y = self.conv2_y(h1_y, edge_index_y)
            h3_y = self.conv3_y(h2_y, edge_index_y)

        h1_x = global_add_pool(h1_x, batch)
        h2_x = global_add_pool(h2_x, batch)
        h3_x = global_add_pool(h3_x, batch)

        h1_y = global_add_pool(h1_y, batch)
        h2_y = global_add_pool(h2_y, batch)
        h3_y = global_add_pool(h3_y, batch)

        h = torch.cat((h1_x, h2_x, h3_x, h1_y, h2_y, h3_y), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h, F.log_softmax(h, dim=1)


# if __name__ == "__main__":
    