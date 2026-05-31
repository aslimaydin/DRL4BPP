"""
rl_model.py
===========
Modüler Graph Encoder + Actor (Policy) + Critic (Value) + Q-Network.

Proje önerisine uygun:
- GraphEncoder: GNN katmanları ile düğüm gömmeleri
- StateAggregator: Düğüm vektörlerinden sabit boyutlu durum vektörü
- PolicyNetwork: Eylem vektörlerinden kenar seçim olasılıkları 
- ValueNetwork: Durum vektöründen V(s) tahmini
- QNetwork: Durum + eylem vektöründen Q(s,a) tahmini
- BPPActorCritic: Tüm bileşenleri birleştiren ana model
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from gnn_layers import get_gnn_layer


class GraphEncoder(nn.Module):
    """
    GNN-tabanlı çizge encoder.
    
    Düğüm vektörlerini L katman GNN'den geçirerek güncellenmiş
    düğüm gömmeleri üretir.
    
    h_i^1 = f_θ0(x_i)                          (input embedding)
    h_i^(l+1) = f_θl(h_i^l, {h_j^l : j → i})  (GNN layer)
    
    Args:
        node_feat_dim: Düğüm özellik boyutu (giriş)
        embed_dim: Embedding boyutu
        n_layers: GNN katman sayısı
        gnn_type: 'gcn', 'gat', veya 'gin'
        n_heads: GAT için attention head sayısı
        dropout: Dropout oranı
    """
    
    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 128,
                 n_layers: int = 3, gnn_type: str = 'gat',
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        
        # Input embedding: x → h^0
        self.input_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, embed_dim),
            nn.ReLU(),
        )
        
        # GNN katmanları
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for l in range(n_layers):
            in_dim = embed_dim
            out_dim = embed_dim
            
            if gnn_type == 'gat':
                # GAT: ara katmanlarda concat, son katmanda mean
                is_last = (l == n_layers - 1)
                if is_last:
                    layer = get_gnn_layer('gat', in_dim, out_dim,
                                         n_heads=n_heads, concat=False,
                                         dropout=dropout)
                else:
                    # Concat modunda çıkış n_heads * (embed_dim // n_heads) olmalı
                    head_dim = embed_dim // n_heads
                    layer = get_gnn_layer('gat', in_dim, head_dim,
                                         n_heads=n_heads, concat=True,
                                         dropout=dropout)
                    # concat sonrası: n_heads * head_dim = embed_dim
            else:
                layer = get_gnn_layer(gnn_type, in_dim, out_dim,
                                     n_heads=n_heads)
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(embed_dim))
    
    def forward(self, node_features: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, node_feat_dim) düğüm özellikleri
            adj: (N, N) adjacency matrix
            
        Returns:
            h: (N, embed_dim) güncellenmiş düğüm gömmeleri
        """
        # Input embedding
        h = self.input_embedding(node_features)  # (N, embed_dim)
        
        # GNN katmanları (residual + LayerNorm)
        for l, (gnn_layer, layer_norm) in enumerate(
            zip(self.gnn_layers, self.layer_norms)
        ):
            h_new = gnn_layer(h, adj)           # (N, embed_dim)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = layer_norm(h + h_new)            # Residual + LayerNorm
        
        return h  # (N, embed_dim)


class StateAggregator(nn.Module):
    """
    Düğüm gömmelerinden sabit boyutlu durum vektörü üretir.
    
    Proje önerisinden:
    "Birleşim işlemi için toplama, ortalama, minimum, maximum, 
     MLP katmanı gibi farklı birleşim fonksiyonu bulunmaktadır."
    
    Args:
        embed_dim: Giriş/çıkış embedding boyutu
        agg_type: 'sum', 'mean', 'max', 'mlp'
    """
    
    def __init__(self, embed_dim: int = 128, agg_type: str = 'mean'):
        super().__init__()
        self.agg_type = agg_type
        self.embed_dim = embed_dim
        
        if agg_type == 'mlp':
            # MLP aggregation: önce mean pool, sonra MLP dönüşümü
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: (N, embed_dim) düğüm gömmeleri
            
        Returns:
            state_vector: (embed_dim,) sabit boyutlu durum vektörü
        """
        if self.agg_type == 'sum':
            state = node_embeddings.sum(dim=0)
        elif self.agg_type == 'mean':
            state = node_embeddings.mean(dim=0)
        elif self.agg_type == 'max':
            state, _ = node_embeddings.max(dim=0)
        elif self.agg_type == 'mlp':
            pooled = node_embeddings.mean(dim=0)
            state = self.mlp(pooled)
        else:
            raise ValueError(f"Bilinmeyen aggregation tipi: {self.agg_type}")
        
        return state  # (embed_dim,)


class PolicyNetwork(nn.Module):
    """
    Kenar seçim politikası (Actor).
    
    Her geçerli kenar (i,j) için:
        eylem_vektörü = [h_i ‖ h_j]  (concat)
        skor = MLP(eylem_vektörü)     → skaler
        π(a|s) = softmax(skorlar)     → olasılık dağılımı
    
    Args:
        embed_dim: Düğüm embedding boyutu
        hidden_dim: MLP gizli katman boyutu
    """
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        
        # Eylem vektörü: [h_i ‖ h_j] → 2 * embed_dim
        self.score_network = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, node_embeddings: torch.Tensor,
                valid_edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: (N, embed_dim)
            valid_edges: (E, 2) geçerli kenar listesi
            
        Returns:
            log_probs: (E,) her kenar için log olasılık
        """
        if len(valid_edges) == 0:
            return torch.tensor([])
        
        # Kenar uç düğüm gömmelerini al
        h_i = node_embeddings[valid_edges[:, 0]]  # (E, embed_dim)
        h_j = node_embeddings[valid_edges[:, 1]]  # (E, embed_dim)
        
        # Eylem vektörleri: [h_i ‖ h_j]
        action_vectors = torch.cat([h_i, h_j], dim=1)  # (E, 2*embed_dim)
        
        # Skorlar
        scores = self.score_network(action_vectors).squeeze(-1)  # (E,)
        
        # Log-softmax
        log_probs = F.log_softmax(scores, dim=0)
        
        return log_probs
    
    def get_action_embeddings(self, node_embeddings: torch.Tensor,
                              valid_edges: torch.Tensor) -> torch.Tensor:
        """
        Eylem vektörlerini döndürür (DQN/SAC için).
        
        Returns:
            action_embeddings: (E, 2*embed_dim)
        """
        h_i = node_embeddings[valid_edges[:, 0]]
        h_j = node_embeddings[valid_edges[:, 1]]
        return torch.cat([h_i, h_j], dim=1)


class ValueNetwork(nn.Module):
    """
    Durum değer fonksiyonu (Critic).
    
    V(s) = MLP(durum_vektörü)
    
    REINFORCE, A2C, PPO, SAC için baseline olarak kullanılır.
    
    Args:
        embed_dim: Durum vektörü boyutu
        hidden_dim: MLP gizli katman boyutu
    """
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector: (embed_dim,) durum vektörü
            
        Returns:
            value: () skaler değer tahmini
        """
        return self.value_net(state_vector).squeeze(-1)


class QNetwork(nn.Module):
    """
    State-Action değer fonksiyonu (DQN için).
    
    Q(s, a) = MLP([durum_vektörü ‖ eylem_vektörü])
    
    Args:
        embed_dim: Durum vektörü boyutu
        hidden_dim: MLP gizli katman boyutu
    """
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        
        # Giriş: durum(embed_dim) + eylem(2*embed_dim) = 3*embed_dim
        self.q_net = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state_vector: torch.Tensor,
                action_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector: (embed_dim,) durum vektörü
            action_embedding: (2*embed_dim,) veya (E, 2*embed_dim)
            
        Returns:
            q_value: () veya (E,) Q değeri/değerleri
        """
        if action_embedding.dim() == 1:
            # Tek aksiyon
            sa = torch.cat([state_vector, action_embedding], dim=0)
            return self.q_net(sa).squeeze(-1)
        else:
            # Birden fazla aksiyon
            state_expanded = state_vector.unsqueeze(0).expand(
                action_embedding.size(0), -1)
            sa = torch.cat([state_expanded, action_embedding], dim=1)
            return self.q_net(sa).squeeze(-1)


class BPPActorCritic(nn.Module):
    """
    Tüm bileşenleri birleştiren ana model.
    
    Graph Encoder → State Aggregator → Policy/Value/Q Networks
    
    Konfigürasyon ile GNN tipi, aggregation, ve ağ yapısı değiştirilebilir.
    """
    
    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 128,
                 n_gnn_layers: int = 3, gnn_type: str = 'gat',
                 n_heads: int = 4, agg_type: str = 'mean',
                 policy_hidden: int = 128, value_hidden: int = 128,
                 dropout: float = 0.1, use_q_network: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_q_network = use_q_network
        
        # 1. Graph Encoder
        self.encoder = GraphEncoder(
            node_feat_dim=node_feat_dim,
            embed_dim=embed_dim,
            n_layers=n_gnn_layers,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # 2. State Aggregator
        self.aggregator = StateAggregator(
            embed_dim=embed_dim,
            agg_type=agg_type,
        )
        
        # 3. Policy Network (Actor)
        self.policy = PolicyNetwork(
            embed_dim=embed_dim,
            hidden_dim=policy_hidden,
        )
        
        # 4. Value Network (Critic)
        self.value = ValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=value_hidden,
        )
        
        # 5. Q-Network (DQN/SAC için, opsiyonel)
        if use_q_network:
            self.q_net = QNetwork(
                embed_dim=embed_dim,
                hidden_dim=value_hidden,
            )
    
    @property
    def device(self):
        """Returns the device the model parameters are on."""
        return next(self.parameters()).device

    def encode(self, node_features: torch.Tensor,
               adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Çizgeyi kodla → düğüm gömmeleri ve durum vektörü.
        
        Returns:
            (node_embeddings, state_vector)
        """
        node_embeddings = self.encoder(node_features, adj)
        state_vector = self.aggregator(node_embeddings)
        return node_embeddings, state_vector
    
    def select_action(self, state: Dict[str, torch.Tensor],
                      greedy: bool = False
                      ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Verilen durumda bir kenar seçer.
        
        Args:
            state: Environment'tan gelen state dict
            greedy: True ise argmax, False ise sampling
            
        Returns:
            (edge_idx, log_prob, value): 
                Seçilen kenar indexi, log olasılık, durum değeri
        """
        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_edges = state['valid_edges'].to(dev)
        
        if len(valid_edges) == 0:
            # No valid edges -> episode should end
            return -1, torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
        
        # Encode
        node_embeddings, state_vector = self.encode(node_features, adj)
        
        # Policy → kenar log-olasılıkları
        log_probs = self.policy(node_embeddings, valid_edges)
        
        # Value
        value = self.value(state_vector)
        
        # Aksiyon seç
        if greedy:
            edge_idx = torch.argmax(log_probs).item()
        else:
            probs = torch.exp(log_probs)
            dist = torch.distributions.Categorical(probs)
            edge_idx_tensor = dist.sample()
            edge_idx = edge_idx_tensor.item()
        
        return edge_idx, log_probs[edge_idx], value
    
    def evaluate_action(self, state: Dict[str, torch.Tensor],
                        edge_idx: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Verilen durum ve aksiyon için log_prob, value ve entropy hesaplar.
        PPO için gerekli.
        
        Returns:
            (log_prob, value, entropy)
        """
        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_edges = state['valid_edges'].to(dev)
        
        node_embeddings, state_vector = self.encode(node_features, adj)
        log_probs = self.policy(node_embeddings, valid_edges)
        value = self.value(state_vector)
        
        # Entropy
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        
        return log_probs[edge_idx], value, entropy
    
    def get_q_values(self, state: Dict[str, torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DQN için: tüm geçerli kenarların Q değerlerini hesaplar.
        
        Returns:
            (q_values, valid_edges): Q değerleri ve kenar listesi
        """
        if not self.use_q_network:
            raise RuntimeError("Q-Network aktif değil. use_q_network=True ile oluşturun.")
        
        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_edges = state['valid_edges'].to(dev)
        
        node_embeddings, state_vector = self.encode(node_features, adj)
        action_embeddings = self.policy.get_action_embeddings(
            node_embeddings, valid_edges)
        
        q_values = self.q_net(state_vector, action_embeddings)
        
        return q_values, valid_edges
    
    def solve_greedy(self, env) -> Tuple[int, List]:
        """
        Greedy politika ile tam çözüm üretir.
        
        Returns:
            (num_bins, merge_history): Bin sayısı ve birleştirme geçmişi
        """
        state = env.get_state()
        merge_history = []
        
        with torch.no_grad():
            while not env.done:
                edge_idx, _, _ = self.select_action(state, greedy=True)
                if edge_idx < 0:
                    break
                state, reward, done, _, info = env.step(edge_idx)
                merge_history.append(info)
        
        return env.get_num_bins(), merge_history


# Typing import
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from rl_environment import BinPackingGraphEnv
    
    print("=" * 60)
    print("Model Testi: BPPActorCritic")
    print("=" * 60)
    
    # Küçük örnek
    env = BinPackingGraphEnv(n_items=5, capacity=11, reward_type='step')
    state = env.reset(items=[9, 2, 4, 1, 5])
    
    # Her GNN tipi ile test
    for gnn_type in ['gcn', 'gat', 'gin']:
        print(f"\n--- GNN: {gnn_type.upper()} ---")
        model = BPPActorCritic(
            node_feat_dim=2, embed_dim=32,
            n_gnn_layers=2, gnn_type=gnn_type,
            n_heads=4, agg_type='mean',
        )
        
        # Forward pass
        edge_idx, log_prob, value = model.select_action(state, greedy=False)
        print(f"Seçilen kenar indexi: {edge_idx}")
        print(f"Log-prob: {log_prob.item():.4f}")
        print(f"Value: {value.item():.4f}")
        
        # Evaluate
        log_p, val, ent = model.evaluate_action(state, edge_idx)
        print(f"Entropy: {ent.item():.4f}")
        
        # Parametre sayısı
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parametre sayısı: {n_params:,}")
    
    # Greedy çözüm testi
    print(f"\n--- Greedy Çözüm Testi ---")
    model = BPPActorCritic(node_feat_dim=2, embed_dim=32,
                           n_gnn_layers=2, gnn_type='gat')
    env = BinPackingGraphEnv(n_items=10, capacity=100, reward_type='step')
    state = env.reset(seed=42)
    
    num_bins, history = model.solve_greedy(env)
    print(f"Greedy çözüm: {num_bins} kutu (random ağırlıklarla)")
    
    # DQN modu testi
    print(f"\n--- DQN Modu Testi ---")
    model_dqn = BPPActorCritic(node_feat_dim=2, embed_dim=32,
                                n_gnn_layers=2, gnn_type='gat',
                                use_q_network=True)
    env = BinPackingGraphEnv(n_items=5, capacity=11)
    state = env.reset(items=[9, 2, 4, 1, 5])
    
    q_vals, edges = model_dqn.get_q_values(state)
    print(f"Q değerleri ({len(q_vals)} kenar): {q_vals.detach().numpy()}")
    
    print("\n✓ Tüm model testleri başarılı.")
