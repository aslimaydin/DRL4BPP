"""
rl_test.py
==========
Eğitilmiş modeli test eder ve baseline yöntemlerle karşılaştırır.

Kullanım:
    python rl_test.py --checkpoint checkpoints/gat_reinforce_step/best_model.pth
    python rl_test.py --checkpoint checkpoints/gat_ppo_step/best_model.pth --n_items 100
"""

import argparse
import os
import time
import numpy as np
import torch
import csv

from rl_model import BPPActorCritic
from rl_environment import BinPackingGraphEnv
from rl_utils import (
    first_fit_decreasing, best_fit_decreasing, lower_bound,
    generate_random_instance, generate_perfect_packing_instance,
    load_bpplib_instance, load_dataset_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="1D BPP Graph-DPÖ Test Scripti")
    
    # ── Model ──
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint dosya yolu')
    parser.add_argument('--gnn_type', type=str, default='gat',
                        choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--agg_type', type=str, default='mean',
                        choices=['sum', 'mean', 'max', 'mlp'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_gnn_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=4)
    
    # ── Test ──
    parser.add_argument('--n_items', type=int, default=50)
    parser.add_argument('--capacity', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=100,
                        help='Random test instance sayısı')
    parser.add_argument('--bpplib_dir', type=str, default=None,
                        help='BPPLIB veri seti dizini')
    parser.add_argument('--output_csv', type=str, default='test_results.csv',
                        help='Sonuç CSV dosyası')
    
    # ── Sistem ──
    parser.add_argument('--gpu', action='store_true', default=False)
    
    return parser.parse_args()


def test_instance(model, items, capacity, device):
    """
    Tek bir instance'ı model ile çözer.
    
    Returns:
        (model_bins, elapsed_time)
    """
    env = BinPackingGraphEnv(n_items=len(items), capacity=capacity,
                              reward_type='step')
    state = env.reset(items=items)
    
    t0 = time.time()
    with torch.no_grad():
        while not env.done:
            edge_idx, _, _ = model.select_action(state, greedy=True)
            if edge_idx < 0:
                break
            state, _, _, _, _ = env.step(edge_idx)
    elapsed = time.time() - t0
    
    return env.get_num_bins(), elapsed


def main():
    args = parse_args()
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    # ── Model yükle ──
    model = BPPActorCritic(
        node_feat_dim=2,
        embed_dim=args.embed_dim,
        n_gnn_layers=args.n_gnn_layers,
        gnn_type=args.gnn_type,
        n_heads=args.n_heads,
        agg_type=args.agg_type,
    )
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model yüklendi: {args.checkpoint}")
    else:
        print(f"UYARI: Checkpoint bulunamadı: {args.checkpoint}")
        print("Random ağırlıklarla test ediliyor.")
    
    model.to(device)
    model.eval()
    
    results = []
    
    # ── 1. Random instances ──
    print(f"\n{'='*70}")
    print(f"TEST: Random instances (N={args.n_items}, C={args.capacity})")
    print(f"{'='*70}")
    
    model_bins_list = []
    ffd_bins_list = []
    bfd_bins_list = []
    lb_list = []
    model_times = []
    ffd_times = []
    bfd_times = []
    
    for i in range(args.test_size):
        items = generate_random_instance(args.n_items, args.capacity, seed=20000 + i)
        
        # Model
        model_bins, model_time = test_instance(model, items, args.capacity, device)
        
        # FFD
        t0 = time.time()
        ffd_bins, _ = first_fit_decreasing(items, args.capacity)
        ffd_time = time.time() - t0
        
        # BFD
        t0 = time.time()
        bfd_bins, _ = best_fit_decreasing(items, args.capacity)
        bfd_time = time.time() - t0
        
        # Lower bound
        lb = lower_bound(items, args.capacity)
        
        model_bins_list.append(model_bins)
        ffd_bins_list.append(ffd_bins)
        bfd_bins_list.append(bfd_bins)
        lb_list.append(lb)
        model_times.append(model_time)
        ffd_times.append(ffd_time)
        bfd_times.append(bfd_time)
        
        results.append({
            'instance': f'random_{i}',
            'n': len(items),
            'capacity': args.capacity,
            'lb': lb,
            'model_bins': model_bins,
            'ffd_bins': ffd_bins,
            'bfd_bins': bfd_bins,
            'model_time': model_time,
            'ffd_time': ffd_time,
            'bfd_time': bfd_time,
        })
    
    avg_lb = np.mean(lb_list)
    avg_model = np.mean(model_bins_list)
    avg_ffd = np.mean(ffd_bins_list)
    avg_bfd = np.mean(bfd_bins_list)
    
    def gap_pct(bins, ref):
        return ((bins - ref) / ref * 100) if ref > 0 else 0
    
    print(f"\n{'Yöntem':<12} {'Ort. Bins':>10} {'Gap(LB)%':>10} {'Ort. Süre':>10}")
    print(f"{'-'*44}")
    print(f"{'Alt Sınır':<12} {avg_lb:>10.1f} {'—':>10} {'—':>10}")
    print(f"{'Model':<12} {avg_model:>10.1f} {gap_pct(avg_model, avg_lb):>9.2f}% "
          f"{np.mean(model_times):>9.4f}s")
    print(f"{'FFD':<12} {avg_ffd:>10.1f} {gap_pct(avg_ffd, avg_lb):>9.2f}% "
          f"{np.mean(ffd_times):>9.4f}s")
    print(f"{'BFD':<12} {avg_bfd:>10.1f} {gap_pct(avg_bfd, avg_lb):>9.2f}% "
          f"{np.mean(bfd_times):>9.4f}s")
    
    # Model FFD'den kaç instance'da daha iyi?
    model_better = sum(1 for m, f in zip(model_bins_list, ffd_bins_list) if m < f)
    model_equal = sum(1 for m, f in zip(model_bins_list, ffd_bins_list) if m == f)
    model_worse = sum(1 for m, f in zip(model_bins_list, ffd_bins_list) if m > f)
    print(f"\nModel vs FFD:  Daha iyi: {model_better}, Eşit: {model_equal}, "
          f"Daha kötü: {model_worse}")
    
    # ── 2. BPPLIB instances ──
    if args.bpplib_dir and os.path.isdir(args.bpplib_dir):
        print(f"\n{'='*70}")
        print(f"TEST: BPPLIB instances ({args.bpplib_dir})")
        print(f"{'='*70}")
        
        instances = load_dataset_directory(args.bpplib_dir)
        
        for idx, (items, capacity, known_opt) in enumerate(instances):
            model_bins, model_time = test_instance(model, items, capacity, device)
            
            ffd_bins, _ = first_fit_decreasing(items, capacity)
            bfd_bins, _ = best_fit_decreasing(items, capacity)
            lb = lower_bound(items, capacity)
            
            opt = known_opt if known_opt else lb
            
            print(f"Instance {idx}: N={len(items)}, C={capacity} | "
                  f"LB={lb}, Model={model_bins}, FFD={ffd_bins}, BFD={bfd_bins}")
            
            results.append({
                'instance': f'bpplib_{idx}',
                'n': len(items),
                'capacity': capacity,
                'lb': lb,
                'opt': opt,
                'model_bins': model_bins,
                'ffd_bins': ffd_bins,
                'bfd_bins': bfd_bins,
                'model_time': model_time,
            })
    
    # ── Sonuçları kaydet ──
    if results:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSonuçlar kaydedildi: {args.output_csv}")
    
    # ── Genelleme testi ──
    print(f"\n{'='*70}")
    print(f"GENELLEME TESTİ: Farklı N değerleri")
    print(f"{'='*70}")
    
    for test_n in [20, 50, 100, 200]:
        model_bins_list = []
        ffd_bins_list = []
        
        for i in range(20):
            items = generate_random_instance(test_n, args.capacity, seed=30000 + test_n * 100 + i)
            model_bins, _ = test_instance(model, items, args.capacity, device)
            ffd_bins, _ = first_fit_decreasing(items, args.capacity)
            model_bins_list.append(model_bins)
            ffd_bins_list.append(ffd_bins)
        
        avg_m = np.mean(model_bins_list)
        avg_f = np.mean(ffd_bins_list)
        ratio = avg_m / avg_f if avg_f > 0 else 0
        print(f"N={test_n:4d}: Model={avg_m:.1f}, FFD={avg_f:.1f}, "
              f"Model/FFD={ratio:.3f}")


if __name__ == '__main__':
    main()
