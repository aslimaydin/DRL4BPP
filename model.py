import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class GATEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, embedding_dim=16, n_layers=1, n_heads=4, dropout=0.1):
        super(GATEncoder, self).__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gat = nn.ModuleList([])

        for i in range(n_layers):
            mult = n_heads if i > 0 else 1
            self.gat.append(
                nn.ModuleList([
                    GraphAttentionLayer(hidden_dim * mult, hidden_dim, dropout) 
                    for _ in range(n_heads)
                ])
            )
        
        self.proj = nn.Linear(hidden_dim * n_heads, embedding_dim)
        
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = self.input_proj(x)
        
        batch_size, N, _ = h.size()
        adj = torch.ones(batch_size, N, N).to(x.device)
        
        for layer in self.gat:
            h = torch.cat([att(h, adj) for att in layer], dim=-1)
            h = self.dropout(h)

        embeddings = self.proj(h)
        
        return embeddings
    
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        Wh = self.W(h)
        
        N = Wh.size(1)
        
        a_input = torch.cat([Wh.repeat_interleave(N, dim=1), 
                           Wh.repeat(1, N, 1)], dim=2)
        e = self.leakyrelu(self.a(a_input)).squeeze(2)
        e = e.view(-1, N, N)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.bmm(attention, Wh)
        
        return h_prime
    

class Decoder(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=32):
        super(Decoder, self).__init__()
        
        self.hidden = nn.Linear(2 * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, embeddings):
        batch_size, N, _ = embeddings.size()
        
        emb1 = embeddings.repeat_interleave(N, dim=1)
        emb2 = embeddings.repeat(1, N, 1)
        
        pairs = torch.cat([emb1, emb2], dim=2)
        
        h = F.relu(self.hidden(pairs))
        adj_pred = torch.sigmoid(self.output(h))
        
        return adj_pred.view(batch_size, N, N)

class GATAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, embedding_dim=16, n_layers=1, n_heads=4, dropout=0.1):
        super(GATAutoencoder, self).__init__()
        
        self.encoder = GATEncoder(input_dim, hidden_dim, embedding_dim, n_layers, n_heads, dropout)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        
    def forward(self, x):
        embeddings = self.encoder(x)
        
        adj_pred = self.decoder(embeddings)
        
        return embeddings, adj_pred
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    min_weights = 100
    max_weights = 200

    batch_size = 1
    
    model = GATAutoencoder(input_dim=1, hidden_dim=16, embedding_dim=8, n_layers=3, n_heads=4, dropout=0.1).to(device)
    print(model)
    num_weights = random.randint(min_weights, max_weights)
    weights, _ = torch.sort(torch.rand(batch_size, num_weights, 1, dtype=torch.float32, device=device), dim=1)
    adj_matrix = torch.zeros((batch_size, num_weights, num_weights), dtype=torch.float32).to(device)
    
    print("Data generation...")
    for batch in range(batch_size):
        for i in range(num_weights):
            for j in range(num_weights):
                normalized_sum = weights[batch, i] + weights[batch, j]
                if normalized_sum <= 1.0 and i != j:
                    adj_matrix[batch, i, j] = normalized_sum

    print(weights.shape, adj_matrix.shape)
    
    embeddings, adj_matrix_pred = model(weights)