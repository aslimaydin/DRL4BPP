"""
bpplib_test.py
==============
BPPLIB veri setleri uzerinde egitilmis DPO modellerini test eder.
Optimum sonuclarla karsilastirir.

Kullanim:
    python bpplib_test.py --dataset scholl_1 --max_items 50
    python bpplib_test.py --dataset falkenauer_u --max_items 200
    python bpplib_test.py --all --max_items 100
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

from rl_model import BPPActorCritic
from rl_environment import BinPackingGraphEnv
from rl_algorithms import create_algorithm
from rl_utils import first_fit_decreasing, best_fit_decreasing
from bpplib_loader import load_dataset, load_optimal_solutions, load_all_datasets, get_easy_instances


# ---------------------------------------------------------------
# MODEL YUKLEME
# ---------------------------------------------------------------

def load_trained_model(checkpoint_dir: str, gnn_type: str, algorithm: str,
                       embed_dim: int = 128, n_gnn_layers: int = 3,
                       device: str = 'cpu'):
    """Egitilmis modeli yukler."""
    use_q = algorithm in ['dqn', 'sac', 'sarsa']

    model = BPPActorCritic(
        node_feat_dim=2,
        embed_dim=embed_dim,
        n_gnn_layers=n_gnn_layers,
        gnn_type=gnn_type,
        n_heads=4,
        agg_type='mean',
        policy_hidden=embed_dim,
        value_hidden=embed_dim,
        dropout=0.1,
        use_q_network=use_q,
    )

    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, 'final_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # checkpoint icinde model_state_dict var mi kontrol et
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def solve_instance_with_model(model, items, capacity, device='cpu'):
    """Bir BPPLIB ornegini model ile cozer. Kutu sayisini dondurur."""
    env = BinPackingGraphEnv(n_items=len(items), capacity=capacity,
                             reward_type='step')
    state = env.reset(items=items)

    with torch.no_grad():
        while not env.done:
            edge_idx, _, _ = model.select_action(state, greedy=True)
            if edge_idx < 0:
                break
            state, _, _, _, _ = env.step(edge_idx)

    return env.get_num_bins()


# ---------------------------------------------------------------
# TEST
# ---------------------------------------------------------------

MODELS_TO_TEST = [
    ('gcn', 'ppo',       'gcn_ppo_step'),
    ('gat', 'ppo',       'gat_ppo_step'),
    ('gcn', 'reinforce', 'gcn_reinforce_step'),
    ('gat', 'reinforce', 'gat_reinforce_step'),
    ('gin', 'ppo',       'gin_ppo_step'),
    ('gin', 'a2c',       'gin_a2c_step'),
]


def test_on_dataset(dataset_name, instances, solutions, models, max_items=None,
                    device='cpu'):
    """Bir veri seti uzerinde tum modelleri test eder."""

    # max_items filtresi
    if max_items:
        instances = [inst for inst in instances if inst['n_items'] <= max_items]

    if not instances:
        print(f"  {dataset_name}: max_items={max_items} filtresinden sonra ornek kalmadi")
        return None

    print(f"\n{'='*70}")
    print(f"Veri Seti: {dataset_name} ({len(instances)} ornek, "
          f"N={min(i['n_items'] for i in instances)}-{max(i['n_items'] for i in instances)})")
    print(f"{'='*70}")

    results = {
        'dataset': dataset_name,
        'n_instances': len(instances),
        'models': {},
        'heuristics': {},
        'per_instance': [],
    }

    # --- Sezgisel yontemler ---
    ffd_bins_list = []
    bfd_bins_list = []
    opt_bins_list = []

    for inst in instances:
        ffd_bins, _ = first_fit_decreasing(inst['weights'], inst['capacity'])
        bfd_bins, _ = best_fit_decreasing(inst['weights'], inst['capacity'])
        ffd_bins_list.append(ffd_bins)
        bfd_bins_list.append(bfd_bins)

        sol = solutions.get(inst['name'])
        opt = sol['ub'] if sol else None
        opt_bins_list.append(opt)

    results['heuristics']['FFD'] = {
        'avg_bins': np.mean(ffd_bins_list),
        'bins': ffd_bins_list,
    }
    results['heuristics']['BFD'] = {
        'avg_bins': np.mean(bfd_bins_list),
        'bins': bfd_bins_list,
    }

    print(f"\n  {'Yontem':<25} {'Ort. Kutu':>10} {'Opt. Gap':>10} {'Opt Esit':>10} {'Sure':>10}")
    print(f"  {'-'*65}")

    # Optimum
    valid_opts = [o for o in opt_bins_list if o is not None]
    if valid_opts:
        opt_avg = np.mean(valid_opts)
        print(f"  {'Optimum':<25} {opt_avg:>10.2f} {'0.0%':>10} {len(valid_opts):>10} {'-':>10}")

    # FFD/BFD
    for hname, hbins in [('FFD', ffd_bins_list), ('BFD', bfd_bins_list)]:
        avg = np.mean(hbins)
        gap = (avg / np.mean(valid_opts) - 1) * 100 if valid_opts else 0
        n_opt = sum(1 for h, o in zip(hbins, opt_bins_list) if o and h == o)
        print(f"  {hname:<25} {avg:>10.2f} {gap:>9.1f}% {n_opt:>10} {'-':>10}")

    # --- DPO modelleri ---
    for gnn_type, alg_name, ckpt_name in models:
        ckpt_dir = os.path.join('checkpoints', ckpt_name)
        label = f"{gnn_type.upper()}+{alg_name.upper()}"

        if not os.path.exists(ckpt_dir):
            print(f"  {label:<25} {'CHECKPOINT YOK':>10}")
            continue

        try:
            model = load_trained_model(ckpt_dir, gnn_type, alg_name, device=device)
        except Exception as e:
            print(f"  {label:<25} HATA: {e}")
            continue

        model_bins = []
        start_time = time.time()

        for inst in tqdm(instances, desc=f"  {label}", leave=False):
            bins = solve_instance_with_model(
                model, inst['weights'], inst['capacity'], device
            )
            model_bins.append(bins)

        elapsed = time.time() - start_time
        avg = np.mean(model_bins)
        gap = (avg / np.mean(valid_opts) - 1) * 100 if valid_opts else 0
        n_opt = sum(1 for m, o in zip(model_bins, opt_bins_list) if o and m == o)

        results['models'][label] = {
            'avg_bins': avg,
            'gap': gap,
            'n_optimal': n_opt,
            'time': elapsed,
            'bins': model_bins,
        }

        print(f"  {label:<25} {avg:>10.2f} {gap:>9.1f}% {n_opt:>10} {elapsed:>9.1f}s")

    # Per-instance detay
    for i, inst in enumerate(instances):
        entry = {
            'name': inst['name'],
            'n_items': inst['n_items'],
            'capacity': inst['capacity'],
            'optimal': opt_bins_list[i],
            'FFD': ffd_bins_list[i],
            'BFD': bfd_bins_list[i],
        }
        for label, mdata in results['models'].items():
            entry[label] = mdata['bins'][i]
        results['per_instance'].append(entry)

    return results


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BPPLIB Benchmark Test")
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['falkenauer_u', 'falkenauer_t', 'scholl_1',
                                 'scholl_2', 'scholl_3', 'wascher', 'hard28'],
                        help='Test edilecek veri seti')
    parser.add_argument('--all', action='store_true',
                        help='Tum veri setlerini test et')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Max nesne sayisi filtresi (N <= max_items)')
    parser.add_argument('--output', type=str, default='bpplib_results.json',
                        help='Sonuc dosyasi')
    parser.add_argument('--gpu', action='store_true', help='GPU kullan')
    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Optimum sonuclari yukle
    xlsx_path = os.path.join('bpplib_data', 'Instances', 'Solutions', 'Solutions.xlsx')
    solutions = load_optimal_solutions(xlsx_path)
    print(f"Optimum sonuclar yuklendi: {len(solutions)} ornek")

    # Modelleri belirle
    models = []
    for gnn, alg, ckpt in MODELS_TO_TEST:
        if os.path.exists(os.path.join('checkpoints', ckpt)):
            models.append((gnn, alg, ckpt))
    print(f"Test edilecek modeller: {[m[2] for m in models]}")

    all_results = {}

    if args.all:
        datasets_to_test = ['scholl_1', 'falkenauer_u', 'falkenauer_t',
                            'scholl_2', 'wascher', 'hard28', 'scholl_3']
    elif args.dataset:
        datasets_to_test = [args.dataset]
    else:
        # Varsayilan: Scholl_1 (N=50, egitim ile uyumlu)
        datasets_to_test = ['scholl_1']

    for ds_name in datasets_to_test:
        try:
            instances = load_dataset(ds_name, max_items=args.max_items)
            if instances:
                result = test_on_dataset(
                    ds_name, instances, solutions, models,
                    max_items=args.max_items, device=device
                )
                if result:
                    all_results[ds_name] = result
        except Exception as e:
            print(f"\n  {ds_name}: HATA - {e}")

    # --- Genel Ozet ---
    print(f"\n{'='*70}")
    print("GENEL OZET")
    print(f"{'='*70}")
    print(f"\n  {'Veri Seti':<18} {'N':>5} {'Orn':>5} | {'Opt':>6} {'FFD':>6} {'BFD':>6}", end='')
    for gnn, alg, _ in models:
        print(f" {gnn.upper()[:3]+'+'+alg.upper()[:3]:>9}", end='')
    print()
    print(f"  {'-'*75}")

    for ds_name, res in all_results.items():
        n_inst = res['n_instances']
        inst_data = res['per_instance']
        n_range = f"{min(i['n_items'] for i in inst_data)}-{max(i['n_items'] for i in inst_data)}"
        opts = [i['optimal'] for i in inst_data if i['optimal']]
        opt_avg = np.mean(opts) if opts else 0

        ffd_avg = res['heuristics']['FFD']['avg_bins']
        bfd_avg = res['heuristics']['BFD']['avg_bins']

        print(f"  {ds_name:<18} {n_range:>5} {n_inst:>5} | {opt_avg:>6.1f} {ffd_avg:>6.1f} {bfd_avg:>6.1f}", end='')
        for gnn, alg, ckpt in models:
            label = f"{gnn.upper()}+{alg.upper()}"
            if label in res['models']:
                print(f" {res['models'][label]['avg_bins']:>9.1f}", end='')
            else:
                print(f" {'N/A':>9}", end='')
        print()

    # Sonuclari kaydet
    # Numpy/list serialization
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi: {args.output}")


if __name__ == '__main__':
    main()
