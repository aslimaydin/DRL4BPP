"""
bpplib_quick_test.py - Hizli BPPLIB benchmark testi (sadece en iyi 2 model)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json, time, numpy as np, torch
from tqdm import tqdm
from bpplib_loader import load_dataset, load_optimal_solutions
from rl_utils import first_fit_decreasing, best_fit_decreasing
from bpplib_test import load_trained_model, solve_instance_with_model

solutions = load_optimal_solutions(r'bpplib_data\Instances\Solutions\Solutions.xlsx')

# En iyi 2 model
models_cfg = [
    ('gcn', 'ppo', 'gcn_ppo_step'),
    ('gat', 'ppo', 'gat_ppo_step'),
]
loaded = {}
for gnn, alg, ckpt in models_cfg:
    loaded[f'{gnn.upper()}+{alg.upper()}'] = load_trained_model(f'checkpoints/{ckpt}', gnn, alg)

datasets = [
    ('Scholl_1 (N<=50)',   'scholl_1', 50),
    ('Scholl_1 (N<=100)',  'scholl_1', 100),
    ('Scholl_1 (N<=200)',  'scholl_1', 200),
    ('Scholl_2 (N<=100)',  'scholl_2', 100),
    ('Wascher',            'wascher',  None),
    ('Hard28',             'hard28',   None),
    ('Falkenauer_U (N<=250)', 'falkenauer_u', 250),
    ('Falkenauer_T (N<=250)', 'falkenauer_t', 250),
]

all_results = {}
header = f"{'Veri Seti':<28} {'Orn':>5} {'N':>8} {'Opt':>6} {'FFD':>6} {'BFD':>6} {'GCN+PPO':>8} {'GAT+PPO':>8} {'Gap%':>6}"
print(header)
print('-' * len(header))

for label, ds_name, max_n in datasets:
    try:
        instances = load_dataset(ds_name, max_items=max_n)
    except Exception as e:
        print(f"{label:<28} HATA: {e}")
        continue
    if not instances:
        continue

    n_range = f"{min(i['n_items'] for i in instances)}-{max(i['n_items'] for i in instances)}"

    opt_list, ffd_list, bfd_list = [], [], []
    model_results = {k: [] for k in loaded}

    for inst in tqdm(instances, desc=label, leave=False):
        sol = solutions.get(inst['name'])
        opt = sol['ub'] if sol else None
        opt_list.append(opt)

        ffd, _ = first_fit_decreasing(inst['weights'], inst['capacity'])
        bfd, _ = best_fit_decreasing(inst['weights'], inst['capacity'])
        ffd_list.append(ffd)
        bfd_list.append(bfd)

        for mname, model in loaded.items():
            bins = solve_instance_with_model(model, inst['weights'], inst['capacity'])
            model_results[mname].append(bins)

    valid_opts = [o for o in opt_list if o]
    opt_avg = np.mean(valid_opts) if valid_opts else 0
    ffd_avg = np.mean(ffd_list)
    bfd_avg = np.mean(bfd_list)

    gcn_avg = np.mean(model_results['GCN+PPO'])
    gat_avg = np.mean(model_results['GAT+PPO'])
    best_model = min(gcn_avg, gat_avg)
    gap = (best_model / opt_avg - 1) * 100 if opt_avg > 0 else 0

    gcn_opt = sum(1 for m, o in zip(model_results['GCN+PPO'], opt_list) if o and m == o)
    ffd_opt = sum(1 for m, o in zip(ffd_list, opt_list) if o and m == o)

    print(f"{label:<28} {len(instances):>5} {n_range:>8} {opt_avg:>6.1f} {ffd_avg:>6.1f} {bfd_avg:>6.1f} {gcn_avg:>8.1f} {gat_avg:>8.1f} {gap:>5.1f}%")

    all_results[label] = {
        'n_instances': len(instances),
        'n_range': n_range,
        'opt_avg': float(opt_avg),
        'ffd_avg': float(ffd_avg),
        'bfd_avg': float(bfd_avg),
        'gcn_ppo_avg': float(gcn_avg),
        'gat_ppo_avg': float(gat_avg),
        'gap_pct': float(gap),
        'gcn_n_optimal': int(gcn_opt),
        'ffd_n_optimal': int(ffd_opt),
        'total_instances': len(instances),
    }

with open('bpplib_results_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSonuclar kaydedildi: bpplib_results_summary.json")
