"""
rl_utils.py
===========
Yardımcı fonksiyonlar:
- Random instance üreteci (uniform, triplet)
- BPPLIB veri seti yükleyicileri (Falkenauer, Scholl, Wascher)
- Baseline sezgisel yöntemler (FFD, BFD)
- Çizge oluşturma ve return hesaplama
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE ÜRETECİ
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_instance(n_items: int, capacity: int,
                             low: int = 1, high: Optional[int] = None,
                             seed: Optional[int] = None) -> List[int]:
    """
    Uniform dağılımlı random BPP instance'ı üretir.
    
    Args:
        n_items: Item sayısı
        capacity: Bin kapasitesi
        low: Minimum item boyutu
        high: Maximum item boyutu (varsayılan: capacity)
        seed: Random seed
        
    Returns:
        items: Item ağırlıkları listesi
    """
    if high is None:
        high = capacity
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    items = [rng.randint(low, high) for _ in range(n_items)]
    return items


def generate_perfect_packing_instance(n_items: int, capacity: int,
                                       seed: Optional[int] = None) -> Tuple[List[int], int]:
    """
    Optimum çözümü bilinen bir instance üretir.
    Her bin tam dolacak şekilde itemler üretilir.
    
    Returns:
        (items, optimal_bins): Item listesi ve optimal bin sayısı
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    items = []
    num_bins = 0
    
    while len(items) < n_items:
        items_needed = n_items - len(items)
        
        if items_needed == 1:
            items.append(capacity)
            num_bins += 1
        else:
            num_pieces = min(items_needed, rng.randint(2, 5))
            bin_items = []
            remaining = capacity
            
            for j in range(num_pieces - 1):
                max_allowable = remaining - (num_pieces - 1 - j)
                if max_allowable <= 1:
                    val = 1
                else:
                    val = rng.randint(1, max(1, min(max_allowable, capacity // 2)))
                bin_items.append(val)
                remaining -= val
            
            bin_items.append(remaining)
            items.extend(bin_items)
            num_bins += 1
    
    items = items[:n_items]
    return items, num_bins


# ─────────────────────────────────────────────────────────────────────────────
# BPPLIB VERİ SETİ YÜKLEYİCİLERİ
# ─────────────────────────────────────────────────────────────────────────────

def load_bpplib_instance(filepath: str) -> Tuple[List[int], int, Optional[int]]:
    """
    BPPLIB formatındaki bir BPP instance dosyasını okur.
    
    Format (Falkenauer/Scholl):
        Satır 1: Item sayısı (n)
        Satır 2: Bin kapasitesi (C)  
        Satır 3..n+2: Her satırda bir item boyutu
    
    Returns:
        (items, capacity, known_optimum): Items, kapasite, bilinen optimum (None ise bilinmiyor)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance dosyası bulunamadı: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    n_items = int(lines[0])
    capacity = int(lines[1])
    
    items = []
    for i in range(2, min(2 + n_items, len(lines))):
        items.append(int(lines[i]))
    
    if len(items) != n_items:
        print(f"Uyarı: Beklenen {n_items} item, bulunan {len(items)} item.")
    
    return items, capacity, None


def load_falkenauer_instance(filepath: str) -> Tuple[List[int], int, Optional[int]]:
    """Falkenauer formatındaki instance'ı yükler."""
    return load_bpplib_instance(filepath)


def load_scholl_instance(filepath: str) -> Tuple[List[int], int, Optional[int]]:
    """Scholl/SKJ formatındaki instance'ı yükler."""
    return load_bpplib_instance(filepath)


def load_wascher_instance(filepath: str) -> Tuple[List[int], int, Optional[int]]:
    """Wascher formatındaki instance'ı yükler."""
    return load_bpplib_instance(filepath)


def load_dataset_directory(dir_path: str) -> List[Tuple[List[int], int, Optional[int]]]:
    """
    Bir dizindeki tüm BPP instance dosyalarını yükler.
    
    Returns:
        Liste: [(items, capacity, known_opt), ...]
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Dizin bulunamadı: {dir_path}")
    
    instances = []
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath) and not fname.startswith('.'):
            try:
                instance = load_bpplib_instance(fpath)
                instances.append(instance)
            except (ValueError, IndexError):
                print(f"Uyarı: {fname} okunamadı, atlanıyor.")
    
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE SEZGİSEL YÖNTEMLER
# ─────────────────────────────────────────────────────────────────────────────

def first_fit_decreasing(items: List[int], capacity: int) -> Tuple[int, List[List[int]]]:
    """
    First Fit Decreasing (FFD) sezgisel yöntemi.
    
    Args:
        items: Item ağırlıkları
        capacity: Bin kapasitesi
        
    Returns:
        (num_bins, bins): Bin sayısı ve her bin'deki item indeksleri
    """
    indexed = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins_remaining = []  # Her bin'in kalan kapasitesi
    bins_contents = []   # Her bin'deki item indeksleri
    
    for idx, weight in indexed:
        placed = False
        for b in range(len(bins_remaining)):
            if bins_remaining[b] >= weight:
                bins_remaining[b] -= weight
                bins_contents[b].append(idx)
                placed = True
                break
        if not placed:
            bins_remaining.append(capacity - weight)
            bins_contents.append([idx])
    
    return len(bins_contents), bins_contents


def best_fit_decreasing(items: List[int], capacity: int) -> Tuple[int, List[List[int]]]:
    """
    Best Fit Decreasing (BFD) sezgisel yöntemi.
    
    Args:
        items: Item ağırlıkları
        capacity: Bin kapasitesi
        
    Returns:
        (num_bins, bins): Bin sayısı ve her bin'deki item indeksleri
    """
    indexed = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins_remaining = []
    bins_contents = []
    
    for idx, weight in indexed:
        best_bin = -1
        best_remaining = capacity + 1
        
        for b in range(len(bins_remaining)):
            if bins_remaining[b] >= weight and bins_remaining[b] - weight < best_remaining:
                best_bin = b
                best_remaining = bins_remaining[b] - weight
        
        if best_bin >= 0:
            bins_remaining[best_bin] -= weight
            bins_contents[best_bin].append(idx)
        else:
            bins_remaining.append(capacity - weight)
            bins_contents.append([idx])
    
    return len(bins_contents), bins_contents


def lower_bound(items: List[int], capacity: int) -> int:
    """L1 alt sınır: ceil(sum(items) / capacity)"""
    return -(-sum(items) // capacity)


# ─────────────────────────────────────────────────────────────────────────────
# ÇİZGE OLUŞTURMA
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(weights: List[int], capacity: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Item ağırlıklarından yönsüz uyumluluk çizgesi oluşturur.
    
    Kenar (i,j) var ↔ weights[i] + weights[j] <= capacity
    
    Args:
        weights: Item ağırlıkları
        capacity: Bin kapasitesi
        
    Returns:
        (node_features, adjacency_matrix):
            node_features: (N, 2) — [w_i/C, degree_i/N]
            adjacency_matrix: (N, N) — 0/1 simetrik, köşegen 0
    """
    n = len(weights)
    adj = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i] + weights[j] <= capacity:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    
    # Düğüm özellikleri: [normalized_weight, normalized_degree]
    degrees = adj.sum(dim=1)
    max_degree = max(degrees.max().item(), 1.0)
    
    node_features = torch.zeros(n, 2)
    for i in range(n):
        node_features[i, 0] = weights[i] / capacity       # Doluluk oranı (w_j / C)
        node_features[i, 1] = degrees[i].item() / max_degree  # Normalize derece
    
    return node_features, adj


def get_valid_edges(adj: torch.Tensor) -> torch.Tensor:
    """
    Adjacency matrix'ten geçerli kenar listesini çıkarır.
    Sadece üst üçgeni alır (yönsüz çizge, tekrar yok).
    
    Args:
        adj: (N, N) adjacency matrix
        
    Returns:
        edges: (E, 2) — kenar listesi (i, j) çiftleri, i < j
    """
    indices = torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=False)
    return indices  # (E, 2)


# ─────────────────────────────────────────────────────────────────────────────
# RETURN HESAPLAMA
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    """
    Discounted return hesaplar.
    
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    
    Args:
        rewards: Adım ödülleri listesi
        gamma: Discount factor
        
    Returns:
        returns: Her adım için G_t değerleri
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_advantages(returns: List[float], values: List[float]) -> List[float]:
    """
    Advantage hesaplar: A_t = G_t - V(s_t)
    
    Args:
        returns: Discounted returns
        values: Value function tahminleri
        
    Returns:
        advantages: Normalize edilmiş advantage değerleri
    """
    advantages = [r - v for r, v in zip(returns, values)]
    
    # Normalize (opsiyonel ama stabilite için önemli)
    if len(advantages) > 1:
        mean_adv = sum(advantages) / len(advantages)
        std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]
    
    return advantages


# ─────────────────────────────────────────────────────────────────────────────
# TEST / DOĞRULAMA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Başvuru formu örneği: 5 item, C=11
    items = [9, 2, 4, 1, 5]
    capacity = 11
    
    print("=" * 50)
    print("1D BPP Örnek: items =", items, ", C =", capacity)
    print("=" * 50)
    
    # Çizge oluştur
    node_feats, adj = build_graph(items, capacity)
    print("\nDüğüm özellikleri (w/C, degree/max_degree):")
    for i, (w, feat) in enumerate(zip(items, node_feats)):
        print(f"  d_{i+1}: w={w}, features={feat.tolist()}")
    
    print("\nAdjacency matrix:")
    print(adj.int().numpy())
    
    edges = get_valid_edges(adj)
    print(f"\nGeçerli kenarlar ({len(edges)} adet):")
    for e in edges:
        i, j = e[0].item(), e[1].item()
        print(f"  k: d_{i+1}(w={items[i]}) -- d_{j+1}(w={items[j]})  toplam={items[i]+items[j]}")
    
    # Baseline'lar
    ffd_bins, ffd_sol = first_fit_decreasing(items, capacity)
    bfd_bins, bfd_sol = best_fit_decreasing(items, capacity)
    lb = lower_bound(items, capacity)
    
    print(f"\nAlt sınır (L1): {lb}")
    print(f"FFD: {ffd_bins} bin")
    print(f"BFD: {bfd_bins} bin")
    
    # Return hesaplama testi
    rewards = [1, 1, 0]  # step reward: 2 birleştirme + terminal
    returns = compute_returns(rewards, gamma=1.0)
    print(f"\nRewards: {rewards}")
    print(f"Returns: {returns}")
