"""Greedy vs Sampling karsilastirmasi - FFD ile ayni sonuc cikma sebebini arastirir."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np
from rl_environment import BinPackingGraphEnv
from bpplib_test import load_trained_model
from bpplib_loader import load_dataset, load_optimal_solutions
from rl_utils import first_fit_decreasing

model = load_trained_model('checkpoints/gcn_ppo_step', 'gcn', 'ppo')
solutions = load_optimal_solutions(r'bpplib_data\Instances\Solutions\Solutions.xlsx')
instances = load_dataset('scholl_1', max_items=50)[:30]

N_SAMPLES = 10

print(f"{'Ornek':<15} {'Opt':>4} {'FFD':>4} {'Greedy':>6} {'Best@10':>7} {'FFD=Grdy':>8}")
print('-' * 55)

ffd_eq_greedy = 0
sampling_beats_ffd = 0
sampling_beats_greedy = 0

for inst in instances:
    sol = solutions.get(inst['name'])
    opt = sol['ub'] if sol else None
    ffd_n, _ = first_fit_decreasing(inst['weights'], inst['capacity'])

    # Greedy
    env = BinPackingGraphEnv(n_items=len(inst['weights']),
                             capacity=inst['capacity'], reward_type='step')
    state = env.reset(items=inst['weights'])
    with torch.no_grad():
        while not env.done:
            edge_idx, _, _ = model.select_action(state, greedy=True)
            if edge_idx < 0:
                break
            state, _, _, _, _ = env.step(edge_idx)
    greedy_n = env.get_num_bins()

    # Sampling N deneme
    best_sampling = 999
    for _ in range(N_SAMPLES):
        env2 = BinPackingGraphEnv(n_items=len(inst['weights']),
                                  capacity=inst['capacity'], reward_type='step')
        state2 = env2.reset(items=inst['weights'])
        with torch.no_grad():
            while not env2.done:
                edge_idx, _, _ = model.select_action(state2, greedy=False)
                if edge_idx < 0:
                    break
                state2, _, _, _, _ = env2.step(edge_idx)
        best_sampling = min(best_sampling, env2.get_num_bins())

    eq = 'EVET' if ffd_n == greedy_n else 'HAYIR'
    marker = ''
    if best_sampling < ffd_n:
        marker = ' << FFD\'den iyi!'
        sampling_beats_ffd += 1
    if best_sampling < greedy_n:
        sampling_beats_greedy += 1
    if ffd_n == greedy_n:
        ffd_eq_greedy += 1

    opt_str = str(opt) if opt else '?'
    print(f"{inst['name']:<15} {opt_str:>4} {ffd_n:>4} {greedy_n:>6} {best_sampling:>7} {eq:>8}{marker}")

print(f"\n{'='*55}")
print(f"FFD = Greedy: {ffd_eq_greedy}/{len(instances)} ({100*ffd_eq_greedy/len(instances):.0f}%)")
print(f"Sampling FFD'den iyi: {sampling_beats_ffd}/{len(instances)}")
print(f"Sampling Greedy'den iyi: {sampling_beats_greedy}/{len(instances)}")
