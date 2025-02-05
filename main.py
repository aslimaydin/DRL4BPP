import os
import time
import shutil
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists, join
from torch.utils.data import DataLoader


from model import GATAutoencoder, save_checkpoint
from dataset import BinPackingDataset, collate_binpacking
from generate_data import generate_data

n_epochs = 1000
batch_size = 1
num_workers = os.cpu_count()
lr = 3e-4

input_dim = 1
hidden_dim = 128
embedding_dim = 32
n_layers = 1
n_heads = 4
dropout= 0.3

n_sets = 1000               # The number of sets of weights to generate
range_weights = (100, 200)  # The range of the number of weights in the sets 
range_values=(0.01, 0.99)   # The range of the values of weights

data_dir = "data"
results_dir = "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_losses(avg_losses, avg_validation_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(avg_losses) + 1), avg_losses, label="Training Loss", marker='o')
    plt.plot(range(1, len(avg_validation_losses) + 1), avg_validation_losses, label="Validation Loss", marker='s')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def validate(test_loader, model, loss_fn):
    model.eval()
    avg_loss = 0
    total_loss = 0
    loop = tqdm(test_loader, leave=True, ncols=100)
    for idx, (x, y, _) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        _, adj_pred = model(x)

        recon_loss = loss_fn(adj_pred, y)
        sparsity_loss = torch.mean(adj_pred)
        curr_loss = recon_loss + 0.1 * sparsity_loss
        
        total_loss += curr_loss.item()
        avg_loss = total_loss / (idx + 1)

        loop.set_postfix(loss=total_loss / (idx + 1))
    
    return avg_loss


def train(train_loader, test_loader, model, optimizer, loss_fn):
    avg_losses = []
    avg_validation_losses = []

    for epoch in range(n_epochs):
        model.train()
        
        total_loss = 0
        avg_loss = 0

        y = 0
        embeddings = 0
        
        loop = tqdm(train_loader, leave=True, ncols=100)
        for idx, (x, y, _) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            embeddings, adj_pred = model(x)

            recon_loss = loss_fn(adj_pred, y)
            sparsity_loss = torch.mean(adj_pred)
            curr_loss = recon_loss + 0.1 * sparsity_loss
            
            curr_loss.backward()
            optimizer.step()
            
            total_loss += curr_loss.item()
            avg_loss = total_loss / (idx + 1)

            loop.set_postfix(epoch=epoch+1, loss=total_loss / (idx + 1))
        
        avg_losses.append(avg_loss)

        avg_validation_loss = validate(test_loader, model, loss_fn)
        avg_validation_losses.append(avg_validation_loss)
        
        plot_losses(avg_losses, avg_validation_losses, save_path=join(results_dir, "loss_plot.png"))

        if len(avg_losses) > 5 and avg_losses[-1] < avg_losses[-2]:
            save_checkpoint(model, optimizer, join(results_dir, "checkpoints", f"checkpoint_{epoch}.pth.tar"))

if __name__ == "__main__":
    if exists(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)
    os.mkdir(join(results_dir, "checkpoints"))
        
    if not exists(data_dir):
        generate_data(
            data_dir=data_dir, 
            num_workers=os.cpu_count(), 
            n_sets=n_sets, 
            datatype="h5", 
            range_weights=range_weights, 
            range_values=range_values
            )

    train_dataset = BinPackingDataset(join(data_dir, "train.h5"))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_binpacking, 
        pin_memory=torch.cuda.is_available()
        )
    
    test_dataset = BinPackingDataset(join(data_dir, "test.h5"))
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_binpacking, 
        pin_memory=torch.cuda.is_available()
        )
    
    model = GATAutoencoder(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        embedding_dim=embedding_dim, 
        n_layers=n_layers, 
        n_heads=n_heads, 
        dropout=dropout
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = torch.nn.MSELoss()

    train(train_loader, test_loader, model, optimizer, loss_fn)
    
