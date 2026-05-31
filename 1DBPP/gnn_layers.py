"""
gnn_layers.py
=============
3 farklı GNN katman implementasyonu — saf PyTorch ile.
Tümü aynı interface'i paylaşır: forward(x, adj) → x'

1. GCNLayer  — Graph Convolutional Network (Kipf & Welling, 2016)
2. GATLayer  — Graph Attention Network (Veličković et al., 2017)
3. GINLayer  — Graph Isomorphism Network (Xu et al., 2018)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network katmanı (Kipf & Welling, 2016).
    
    h_i^(l+1) = σ( Σ_j (1/√(d_i·d_j)) · W · h_j^l + b )
    
    Degree-normalized message passing ile çalışır.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)
        if self.weight.bias is not None:
            nn.init.zeros_(self.weight.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d_in) düğüm özellikleri
            adj: (N, N) adjacency matrix (0/1, köşegen 0)
                 
        Returns:
            out: (N, d_out) güncellenmiş düğüm özellikleri
        """
        # Self-loop ekle
        N = adj.size(0)
        adj_hat = adj + torch.eye(N, device=adj.device)
        
        # Degree matrix
        degree = adj_hat.sum(dim=1).clamp(min=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # Normalized adjacency: D^{-1/2} · A_hat · D^{-1/2}
        adj_norm = D_inv_sqrt @ adj_hat @ D_inv_sqrt
        
        # Message passing: A_norm · X · W
        support = self.weight(x)      # (N, d_out)
        out = adj_norm @ support      # (N, d_out)
        
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network katmanı (Veličković et al., 2017).
    
    Multi-head attention:
        α_ij = softmax_j( LeakyReLU( a^T [Wh_i ‖ Wh_j] ) )
        h_i^(l+1) = σ( ‖_{k=1}^K Σ_j α_ij^k · W^k · h_j^l )
    
    Args:
        in_features: Giriş boyutu
        out_features: Çıkış boyutu (her head için)
        n_heads: Attention kafa sayısı
        concat: True ise head'ler concat edilir, False ise mean alınır
        dropout: Attention dropout oranı
        negative_slope: LeakyReLU negative slope
    """
    
    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, concat: bool = True,
                 dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        
        # Her head için ayrı W matrisi
        self.W = nn.Parameter(torch.Tensor(n_heads, in_features, out_features))
        
        # Attention vektörleri: a = [a_left | a_right] (her head için)
        self.a_left = nn.Parameter(torch.Tensor(n_heads, out_features, 1))
        self.a_right = nn.Parameter(torch.Tensor(n_heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)
        
        if concat:
            self.bias = nn.Parameter(torch.Tensor(n_heads * out_features))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d_in) düğüm özellikleri
            adj: (N, N) adjacency matrix
            
        Returns:
            out: (N, d_out * n_heads) if concat, (N, d_out) if mean
        """
        N = x.size(0)
        
        # Self-loop ekle
        adj_hat = adj + torch.eye(N, device=adj.device)
        
        # Lineer dönüşüm: her head için Wh
        # x: (N, d_in) → (K, N, d_out)
        Wh = torch.einsum('ni,kio->kno', x, self.W)  # (K, N, d_out)
        
        # Attention skorları
        # a_left^T · Wh_i: (K, N, 1)
        e_left = torch.bmm(Wh, self.a_left)    # (K, N, 1)
        # a_right^T · Wh_j: (K, N, 1)  
        e_right = torch.bmm(Wh, self.a_right)  # (K, N, 1)
        
        # e_ij = LeakyReLU(e_left_i + e_right_j)
        # Broadcasting: (K, N, 1) + (K, 1, N) → (K, N, N)
        e = self.leaky_relu(e_left + e_right.transpose(1, 2))
        
        # Maskeleme: kenar olmayan çiftlere -inf
        mask = adj_hat.unsqueeze(0).expand(self.n_heads, -1, -1)  # (K, N, N)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax → attention katsayıları
        alpha = F.softmax(e, dim=2)  # (K, N, N)
        alpha = self.attn_dropout(alpha)
        
        # Ağırlıklı toplam: (K, N, N) @ (K, N, d_out) → (K, N, d_out)
        h_prime = torch.bmm(alpha, Wh)
        
        if self.concat:
            # Head'leri concat et: (N, K*d_out)
            out = h_prime.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            # Head'lerin ortalamasını al: (N, d_out)
            out = h_prime.mean(dim=0)
        
        out = out + self.bias
        return out


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network katmanı (Xu et al., 2018).
    
    h_i^(l+1) = MLP( (1 + ε) · h_i^l  +  Σ_{j ∈ N(i)} h_j^l )
    
    WL testi kadar güçlü olduğu kanıtlanmıştır.
    
    Args:
        in_features: Giriş boyutu
        out_features: Çıkış boyutu
        hidden_features: MLP gizli katman boyutu
        eps_learnable: ε parametresinin öğrenilebilir olup olmadığı
    """
    
    def __init__(self, in_features: int, out_features: int,
                 hidden_features: Optional[int] = None,
                 eps_learnable: bool = True):
        super().__init__()
        
        if hidden_features is None:
            hidden_features = out_features
        
        # MLP: 2 katmanlı
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        
        # Epsilon parametresi
        if eps_learnable:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('eps', torch.zeros(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d_in) düğüm özellikleri
            adj: (N, N) adjacency matrix (self-loop dahil değil)
            
        Returns:
            out: (N, d_out) güncellenmiş düğüm özellikleri
        """
        # Komşu toplamı: A · x
        neighbor_sum = adj @ x  # (N, d_in)
        
        # (1 + ε) · x_i + Σ x_j
        out = (1 + self.eps) * x + neighbor_sum
        
        # MLP
        out = self.mlp(out)
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# TYPING UYUMLULUK
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# İSME GÖRE GNN KATMANI FABRİKASI
# ─────────────────────────────────────────────────────────────────────────────

GNN_REGISTRY = {
    'gcn': GCNLayer,
    'gat': GATLayer,
    'gin': GINLayer,
}


def get_gnn_layer(gnn_type: str, in_features: int, out_features: int,
                  n_heads: int = 4, **kwargs) -> nn.Module:
    """
    İsme göre GNN katmanı oluşturur.
    
    Args:
        gnn_type: 'gcn', 'gat', veya 'gin'
        in_features: Giriş boyutu
        out_features: Çıkış boyutu
        n_heads: GAT için head sayısı
        
    Returns:
        GNN katmanı (nn.Module)
    """
    gnn_type = gnn_type.lower()
    
    if gnn_type == 'gcn':
        return GCNLayer(in_features, out_features)
    elif gnn_type == 'gat':
        # GAT: concat modunda çıkış n_heads * out_features olur
        # Son katmanda concat=False ile mean alınabilir
        concat = kwargs.get('concat', True)
        return GATLayer(in_features, out_features, n_heads=n_heads, 
                       concat=concat, dropout=kwargs.get('dropout', 0.1))
    elif gnn_type == 'gin':
        return GINLayer(in_features, out_features,
                       hidden_features=kwargs.get('hidden_features', out_features))
    else:
        raise ValueError(f"Bilinmeyen GNN tipi: {gnn_type}. "
                        f"Desteklenen: {list(GNN_REGISTRY.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Basit test: 5 düğümlü çizge
    N = 5
    d_in = 2
    d_out = 8
    
    x = torch.randn(N, d_in)
    adj = torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ], dtype=torch.float)
    
    print("Input shape:", x.shape)
    print("Adjacency:\n", adj.int().numpy())
    print()
    
    # GCN testi
    gcn = GCNLayer(d_in, d_out)
    out_gcn = gcn(x, adj)
    print(f"GCN output shape: {out_gcn.shape}")  # (5, 8)
    
    # GAT testi (4 head, concat → çıkış 4*8=32)
    gat = GATLayer(d_in, d_out, n_heads=4, concat=True)
    out_gat = gat(x, adj)
    print(f"GAT output shape (concat): {out_gat.shape}")  # (5, 32)
    
    # GAT mean modu
    gat_mean = GATLayer(d_in, d_out, n_heads=4, concat=False)
    out_gat_mean = gat_mean(x, adj)
    print(f"GAT output shape (mean):   {out_gat_mean.shape}")  # (5, 8)
    
    # GIN testi
    gin = GINLayer(d_in, d_out)
    out_gin = gin(x, adj)
    print(f"GIN output shape: {out_gin.shape}")  # (5, 8)
    
    # Factory testi
    for gnn_type in ['gcn', 'gat', 'gin']:
        layer = get_gnn_layer(gnn_type, d_in, d_out, n_heads=4)
        out = layer(x, adj)
        print(f"Factory {gnn_type}: output shape = {out.shape}")
    
    print("\n✓ Tüm GNN katmanları başarıyla test edildi.")
