"""
rl_environment.py
=================
Graph-tabanlı 1D Bin Packing Problem MDP ortamı.

Proje önerisindeki tanıma uygun:
- State: Yönsüz çizge G_t = (D_t, K_t)
- Action: Kenar seçimi (iki düğümü birleştir)
- Transition: Deterministik (düğüm merge, çizge güncelleme)
- Terminal: Hiç kenar kalmayana kadar
- Reward: R1 (terminal) veya R2 (step)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from rl_utils import build_graph, get_valid_edges, generate_random_instance, generate_perfect_packing_instance


class BinPackingGraphEnv:
    """
    Graph-tabanlı 1D Bin Packing Problem MDP ortamı.
    
    Her episode:
    1. N item ile başla, uyumluluk çizgesi oluştur
    2. Her adımda bir kenar (i,j) seç → düğümleri birleştir
    3. Hiç geçerli kenar kalmayana kadar devam et
    4. Kalan düğüm sayısı = kutu sayısı
    
    Attributes:
        n_items: Instance'daki item sayısı
        capacity: Bin kapasitesi
        reward_type: 'step' (R2) veya 'terminal' (R1)
    """
    
    def __init__(self, n_items: int = 50, capacity: int = 100,
                 reward_type: str = 'step'):
        """
        Args:
            n_items: Item sayısı
            capacity: Bin kapasitesi
            reward_type: 'step' → her birleştirmede +1, terminal'de 0
                        'terminal' → birleştirmelerde 0, terminal'de −kalan_düğüm
        """
        self.n_items = n_items
        self.capacity = capacity
        self.reward_type = reward_type
        
        # Mevcut durum
        self.weights = []          # Her düğümün mevcut ağırlığı
        self.node_features = None  # (N_current, feat_dim) düğüm özellikleri
        self.adj = None            # (N_current, N_current) adjacency matrix
        self.done = False
        self.n_merges = 0          # Toplam birleştirme sayısı
        self.initial_n_items = 0   # Başlangıç item sayısı
        
        # Birleşim geçmişi (debug/analiz için)
        self.merge_history = []
        
        # Her düğümün hangi orijinal itemleri içerdiği
        self.node_contents = []  # Liste of listeler: [[orijinal_idx, ...], ...]
    
    def reset(self, items: Optional[List[int]] = None,
              seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Yeni episode başlat.
        
        Args:
            items: Belirli item ağırlıkları (None ise random üretilir)
            seed: Random seed
            
        Returns:
            state: {
                'node_features': (N, feat_dim),
                'adj': (N, N),
                'valid_edges': (E, 2),
                'weights': list[int]
            }
        """
        if items is None:
            items = generate_random_instance(
                self.n_items, self.capacity, low=1, high=self.capacity, seed=seed
            )
        
        self.weights = list(items)
        self.initial_n_items = len(items)
        self.done = False
        self.n_merges = 0
        self.merge_history = []
        self.node_contents = [[i] for i in range(len(items))]
        
        # Çizgeyi oluştur
        self._rebuild_graph()
        
        # Terminal kontrolü: kenar yoksa zaten done
        valid_edges = get_valid_edges(self.adj)
        if len(valid_edges) == 0:
            self.done = True
        
        return self.get_state()
    
    def step(self, edge_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """
        Bir kenar seçerek iki düğümü birleştir.
        
        Args:
            edge_idx: Geçerli kenar listesindeki index
            
        Returns:
            (state, reward, done, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode zaten bitti. reset() çağırın.")
        
        valid_edges = get_valid_edges(self.adj)
        
        if edge_idx < 0 or edge_idx >= len(valid_edges):
            raise ValueError(f"Geçersiz edge_idx={edge_idx}, "
                           f"geçerli aralık: [0, {len(valid_edges)-1}]")
        
        # Seçilen kenarın düğümlerini al
        node_i = valid_edges[edge_idx][0].item()
        node_j = valid_edges[edge_idx][1].item()
        
        # Birleştirme bilgisini kaydet
        self.merge_history.append({
            'step': self.n_merges,
            'node_i': node_i,
            'node_j': node_j,
            'weight_i': self.weights[node_i],
            'weight_j': self.weights[node_j],
            'merged_weight': self.weights[node_i] + self.weights[node_j],
        })
        
        # Düğümleri birleştir
        self._merge_nodes(node_i, node_j)
        self.n_merges += 1
        
        # Çizgeyi yeniden oluştur
        self._rebuild_graph()
        
        # Terminal kontrolü
        valid_edges_new = get_valid_edges(self.adj)
        if len(valid_edges_new) == 0:
            self.done = True
        
        # Reward hesapla
        reward = self._compute_reward()
        
        # Info
        info = {
            'n_nodes': len(self.weights),
            'n_edges': len(valid_edges_new) if not self.done else 0,
            'n_merges': self.n_merges,
            'n_bins': len(self.weights),  # Mevcut düğüm sayısı = bin sayısı
        }
        
        return self.get_state(), reward, self.done, False, info
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Mevcut durumu döndürür."""
        valid_edges = get_valid_edges(self.adj)
        return {
            'node_features': self.node_features.clone(),
            'adj': self.adj.clone(),
            'valid_edges': valid_edges,
            'weights': list(self.weights),
            'n_nodes': len(self.weights),
        }
    
    def get_num_bins(self) -> int:
        """Mevcut kutu sayısını döndürür (= düğüm sayısı)."""
        return len(self.weights)
    
    def _rebuild_graph(self):
        """
        Mevcut ağırlıklardan çizgeyi yeniden oluşturur.
        """
        self.node_features, self.adj = build_graph(self.weights, self.capacity)
    
    def _merge_nodes(self, node_i: int, node_j: int):
        """
        İki düğümü birleştir.
        
        node_i ve node_j kaldırılır.
        Yeni düğüm (birleşik ağırlık) listenin sonuna eklenir.
        
        Args:
            node_i: İlk düğüm indexi
            node_j: İkinci düğüm indexi (node_i < node_j olmalı)
        """
        # Sıralama garantisi
        if node_i > node_j:
            node_i, node_j = node_j, node_i
        
        # Yeni ağırlık
        new_weight = self.weights[node_i] + self.weights[node_j]
        
        # Yeni içerik
        new_contents = self.node_contents[node_i] + self.node_contents[node_j]
        
        # Büyük indexi önce kaldır (index kayması olmasın)
        del self.weights[node_j]
        del self.node_contents[node_j]
        del self.weights[node_i]
        del self.node_contents[node_i]
        
        # Yeni düğümü ekle
        self.weights.append(new_weight)
        self.node_contents.append(new_contents)
    
    def _compute_reward(self) -> float:
        """
        Reward hesaplar.
        
        R1 (terminal): R = {−r, s_{t+1}=s_T; 0, diğer}
            r = kalan düğüm sayısı (terminal'de)
        R2 (step):     R = {0, s_{t+1}=s_T; 1, diğer}
            Her birleştirmede +1
        """
        if self.reward_type == 'step':
            # R2: Her birleştirmede +1, terminal'de 0
            if self.done:
                return 0.0
            else:
                return 1.0
        
        elif self.reward_type == 'terminal':
            # R1: Terminal'de −kalan_düğüm, ara adımlarda 0
            if self.done:
                return -float(len(self.weights))
            else:
                return 0.0
        
        else:
            raise ValueError(f"Bilinmeyen reward tipi: {self.reward_type}")
    
    def render(self):
        """Mevcut durumu yazdırır (debug için)."""
        print(f"\n{'='*40}")
        print(f"Adım {self.n_merges} | Düğüm sayısı: {len(self.weights)} | Done: {self.done}")
        print(f"Ağırlıklar: {self.weights}")
        
        valid_edges = get_valid_edges(self.adj)
        print(f"Geçerli kenar sayısı: {len(valid_edges)}")
        
        for e_idx, e in enumerate(valid_edges):
            i, j = e[0].item(), e[1].item()
            print(f"  [{e_idx}] ({i},{j}): w={self.weights[i]}+{self.weights[j]}="
                  f"{self.weights[i]+self.weights[j]} ≤ {self.capacity}")
        
        print(f"Düğüm içerikleri:")
        for i, (w, contents) in enumerate(zip(self.weights, self.node_contents)):
            print(f"  Düğüm {i}: w={w}, items={contents}")
        print(f"{'='*40}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Başvuru Formu Örneği: items=[9, 2, 4, 1, 5], C=11")
    print("=" * 60)
    
    env = BinPackingGraphEnv(n_items=5, capacity=11, reward_type='step')
    state = env.reset(items=[9, 2, 4, 1, 5])
    
    env.render()
    
    total_reward = 0.0
    step_count = 0
    
    while not env.done:
        valid_edges = state['valid_edges']
        print(f"\nGeçerli kenarlar: {len(valid_edges)}")
        
        # Random aksiyon seç (ilk geçerli kenarı)
        action = 0
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"→ Reward: {reward}")
        env.render()
    
    print(f"\n{'='*60}")
    print(f"Episode bitti!")
    print(f"Toplam adım: {step_count}")
    print(f"Toplam reward: {total_reward}")
    print(f"Kutu sayısı: {env.get_num_bins()}")
    print(f"{'='*60}")
    
    # Terminal reward testi
    print("\n\nTerminal Reward Testi:")
    env2 = BinPackingGraphEnv(n_items=5, capacity=11, reward_type='terminal')
    state = env2.reset(items=[9, 2, 4, 1, 5])
    
    total_reward = 0.0
    while not env2.done:
        state, reward, done, truncated, info = env2.step(0)
        total_reward += reward
        print(f"  Adım: reward={reward}")
    
    print(f"Terminal toplam reward: {total_reward}")
    print(f"Kutu sayısı: {env2.get_num_bins()}")
