"""
rl_model_2d_hybrid.py
======================
Hibrit 2B-KPP için uyarlanmış Aktör-Eleştirmen modeli.

Mevcut GNN kodlayıcıyı (GCN/GAT/GIN) yeniden kullanır.
Fark: Kenar seçimi yerine nesne+oryantasyon seçimi.

Eylem temsili:
  - Her geçerli eylem (item_idx, rotation) bir vektörle temsil edilir
  - Eylem vektörü = [item_embedding ‖ rotation_embedding]
  - Politika ağı bu vektörleri skorlar
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from gnn_layers import get_gnn_layer


class GraphEncoder2D(nn.Module):
    """
    GNN tabanlı çizge kodlayıcı (2B-KPP hibrit için).
    
    1B versiyonla aynı mimari, sadece node_feat_dim=6 (x, y, w, h, rot, placed).
    """
    
    def __init__(self, node_feat_dim: int = 6, embed_dim: int = 128,
                 n_layers: int = 3, gnn_type: str = 'gcn',
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Giriş gömme
        self.input_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, embed_dim),
            nn.ReLU(),
        )
        
        # GNN katmanları
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for l in range(n_layers):
            if gnn_type == 'gat':
                is_last = (l == n_layers - 1)
                if is_last:
                    layer = get_gnn_layer('gat', embed_dim, embed_dim,
                                         n_heads=n_heads, concat=False,
                                         dropout=dropout)
                else:
                    head_dim = embed_dim // n_heads
                    layer = get_gnn_layer('gat', embed_dim, head_dim,
                                         n_heads=n_heads, concat=True,
                                         dropout=dropout)
            else:
                layer = get_gnn_layer(gnn_type, embed_dim, embed_dim,
                                     n_heads=n_heads)
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(embed_dim))
    
    def forward(self, node_features: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """(N, 6) → (N, embed_dim) düğüm gömmeleri."""
        h = self.input_embedding(node_features)
        
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn_layer(h, adj)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = layer_norm(h + h_new)
        
        return h


class ActionScorer2D(nn.Module):
    """
    2B-KPP eylem skorlayıcı.
    
    Her geçerli eylem (item_idx, rotation) için:
      - Item'ın GNN gömmesini al
      - Rotation bilgisini ekle
      - Heightmap bağlamını ekle
      - MLP ile skorla
    """
    
    def __init__(self, embed_dim: int = 128, heightmap_dim: int = 16,
                 hidden_dim: int = 128):
        super().__init__()
        
        # Heightmap kodlayıcı (W boyutlu vektörden küçük gömmeye)
        self.heightmap_encoder = nn.Sequential(
            nn.Linear(100, heightmap_dim),  # bin_width → heightmap_dim
            nn.ReLU(),
        )
        
        # Eylem skorlama: [item_embed ‖ rot_embed ‖ state ‖ heightmap]
        input_dim = embed_dim + 1 + embed_dim + heightmap_dim
        self.score_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, node_embeddings: torch.Tensor,
                state_vector: torch.Tensor,
                heightmap: torch.Tensor,
                valid_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: (N, embed_dim) düğüm gömmeleri
            state_vector: (embed_dim,) durum vektörü
            heightmap: (W,) heightmap normalize edilmiş
            valid_actions: (K, 2) — [item_idx, rotation]
            
        Returns:
            log_probs: (K,) her eylem için log olasılık
        """
        if len(valid_actions) == 0:
            return torch.tensor([])
        
        K = len(valid_actions)
        dev = node_embeddings.device
        
        # Item gömmeleri
        item_indices = valid_actions[:, 0]
        item_embeds = node_embeddings[item_indices]  # (K, embed_dim)
        
        # Rotation bilgisi
        rot_flags = valid_actions[:, 1].float().unsqueeze(1)  # (K, 1)
        
        # Durum vektörü (her eyleme aynı)
        state_expanded = state_vector.unsqueeze(0).expand(K, -1)  # (K, embed_dim)
        
        # Heightmap gömme
        hm_embed = self.heightmap_encoder(heightmap)  # (heightmap_dim,)
        hm_expanded = hm_embed.unsqueeze(0).expand(K, -1)  # (K, heightmap_dim)
        
        # Birleştir: [item_embed ‖ rot ‖ state ‖ heightmap]
        action_vectors = torch.cat([
            item_embeds, rot_flags, state_expanded, hm_expanded
        ], dim=1)
        
        # Skorla
        scores = self.score_network(action_vectors).squeeze(-1)  # (K,)
        log_probs = F.log_softmax(scores, dim=0)
        
        return log_probs


class BPP2DHybridActorCritic(nn.Module):
    """
    Hibrit 2B-KPP Ana Model.
    
    GraphEncoder2D → StateAggregator → ActionScorer2D + ValueNetwork
    """
    
    def __init__(self, node_feat_dim: int = 6, embed_dim: int = 128,
                 n_gnn_layers: int = 3, gnn_type: str = 'gcn',
                 n_heads: int = 4, bin_width: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.bin_width = bin_width
        
        # 1. GNN Kodlayıcı
        self.encoder = GraphEncoder2D(
            node_feat_dim=node_feat_dim,
            embed_dim=embed_dim,
            n_layers=n_gnn_layers,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # 2. Durum Birleştirici
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        
        # 3. Eylem Skorlayıcı (Politika)
        self.action_scorer = ActionScorer2D(
            embed_dim=embed_dim,
            heightmap_dim=16,
            hidden_dim=embed_dim,
        )
        # Heightmap boyutunu bin_width'e göre ayarla
        self.action_scorer.heightmap_encoder = nn.Sequential(
            nn.Linear(bin_width, 16),
            nn.ReLU(),
        )
        
        # 4. Değer Ağı (Eleştirmen)
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, node_features, adj):
        """Çizgeyi kodla → düğüm gömmeleri + durum vektörü."""
        node_embeddings = self.encoder(node_features, adj)
        state_vector = self.aggregator(node_embeddings.mean(dim=0))
        return node_embeddings, state_vector
    
    def select_action(self, state: Dict[str, torch.Tensor],
                      greedy: bool = False
                      ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Verilen durumda bir eylem (nesne+oryantasyon) seçer.
        
        Returns:
            (action_idx, log_prob, value)
        """
        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_actions = state['valid_actions'].to(dev)
        heightmap = state['heightmap'].to(dev)
        
        if len(valid_actions) == 0:
            return -1, torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
        
        # Kodla
        node_embeddings, state_vector = self.encode(node_features, adj)
        
        # Politika → eylem log-olasılıkları
        log_probs = self.action_scorer(
            node_embeddings, state_vector, heightmap, valid_actions
        )
        
        # Değer
        value = self.value_net(state_vector).squeeze(-1)
        
        # Eylem seç
        if greedy:
            action_idx = torch.argmax(log_probs).item()
        else:
            probs = torch.exp(log_probs)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
        
        return action_idx, log_probs[action_idx], value
    
    def evaluate_action(self, state: Dict[str, torch.Tensor],
                        action_idx: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO için: log_prob, value, entropy hesapla."""
        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_actions = state['valid_actions'].to(dev)
        heightmap = state['heightmap'].to(dev)
        
        node_embeddings, state_vector = self.encode(node_features, adj)
        log_probs = self.action_scorer(
            node_embeddings, state_vector, heightmap, valid_actions
        )
        value = self.value_net(state_vector).squeeze(-1)
        
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        
        return log_probs[action_idx], value, entropy


# ---- TEST ----
if __name__ == '__main__':
    print("=" * 60)
    print("Hibrit 2B-KPP Model Testi")
    print("=" * 60)
    
    # Ortamı yükle
    from rl_environment_2d_hybrid import BinPacking2DHybridEnv
    
    env = BinPacking2DHybridEnv(n_items=10, bin_width=50, bin_height=50)
    state = env.reset(seed=42)
    
    # Model oluştur
    model = BPP2DHybridActorCritic(
        node_feat_dim=6, embed_dim=64,
        n_gnn_layers=2, gnn_type='gcn',
        bin_width=50
    )
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Düğüm sayısı: {state['n_nodes']}")
    print(f"Geçerli eylem sayısı: {len(state['valid_actions'])}")
    
    # Eylem seç
    with torch.no_grad():
        action_idx, log_prob, value = model.select_action(state)
    
    valid_actions = state['valid_actions']
    item_idx, rot = valid_actions[action_idx]
    print(f"\nSeçilen eylem: item={item_idx.item()}, rot={rot.item()}")
    print(f"Log prob: {log_prob.item():.4f}")
    print(f"Value: {value.item():.4f}")
    
    # Bir episode çalıştır
    print("\n--- Episode ---")
    state = env.reset(seed=42)
    total_reward = 0
    steps = 0
    
    while not env.done:
        with torch.no_grad():
            action_idx, log_prob, value = model.select_action(state)
        if action_idx == -1:
            break
        state, reward, done, _, info = env.step(action_idx)
        total_reward += reward
        steps += 1
    
    print(f"Adımlar: {steps}")
    print(f"Yerleşen: {env.n_placed}/{len(env.original_items)}")
    print(f"Utilization: {env.get_utilization():.1%}")
    print(f"Toplam ödül: {total_reward}")
    print(f"Çizge kenar sayısı: {int(env.adj.sum().item()) // 2}")
