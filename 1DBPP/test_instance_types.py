"""
test_instance_types.py
======================
Farkli ornek turleri uzerinde DPO vs FFD vs GA karsilastirmasi.
Hangi dagilim turlerinde DPO avantajli oldugunu arastirir.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, time
from rl_environment import BinPackingGraphEnv
from bpplib_test import load_trained_model
from rl_utils import first_fit_decreasing, best_fit_decreasing
from genetic_algorithm_bpp import solve_bpp_ga

model = load_trained_model('checkpoints/gcn_ppo_step', 'gcn', 'ppo')

def solve_drl(weights, capacity, n_samples=1, greedy=True):
    """DRL ile coz. n_samples>1 ise sampling yapip en iyisini al."""
    best = 999
    for _ in range(n_samples):
        env = BinPackingGraphEnv(n_items=len(weights), capacity=capacity, reward_type='step')
        state = env.reset(items=weights)
        with torch.no_grad():
            while not env.done:
                edge_idx, _, _ = model.select_action(state, greedy=greedy)
                if edge_idx < 0:
                    break
                state, _, _, _, _ = env.step(edge_idx)
        best = min(best, env.get_num_bins())
    return best

def test_distribution(name, generate_fn, n_instances=50, capacity=100):
    """Bir dagilim turunu test et."""
    ffd_list, drl_greedy_list, drl_sample_list, ga_list, opt_lb_list = [], [], [], [], []
    drl_times, ga_times = [], []

    for seed in range(n_instances):
        items = generate_fn(seed, capacity)
        n = len(items)

        # Alt sinir (L2 bound)
        opt_lb = int(np.ceil(sum(items) / capacity))
        opt_lb_list.append(opt_lb)

        ffd_n, _ = first_fit_decreasing(items, capacity)
        ffd_list.append(ffd_n)

        t0 = time.time()
        drl_g = solve_drl(items, capacity, n_samples=1, greedy=True)
        drl_greedy_list.append(drl_g)

        drl_s = solve_drl(items, capacity, n_samples=10, greedy=False)
        drl_sample_list.append(drl_s)
        drl_times.append(time.time() - t0)

        t0 = time.time()
        ga_n, _ = solve_bpp_ga(items, capacity, time_limit=1.0, seed=42)
        ga_list.append(ga_n)
        ga_times.append(time.time() - t0)

    # Sonuclar
    lb_avg = np.mean(opt_lb_list)
    ffd_avg = np.mean(ffd_list)
    drl_g_avg = np.mean(drl_greedy_list)
    drl_s_avg = np.mean(drl_sample_list)
    ga_avg = np.mean(ga_list)

    ffd_gap = (ffd_avg/lb_avg - 1)*100
    drl_g_gap = (drl_g_avg/lb_avg - 1)*100
    drl_s_gap = (drl_s_avg/lb_avg - 1)*100
    ga_gap = (ga_avg/lb_avg - 1)*100

    # FFD'den iyi olan ornekler
    drl_beats_ffd = sum(1 for d, f in zip(drl_sample_list, ffd_list) if d < f)
    ga_beats_ffd = sum(1 for g, f in zip(ga_list, ffd_list) if g < f)
    drl_eq_ffd = sum(1 for d, f in zip(drl_greedy_list, ffd_list) if d == f)

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {n_instances} ornek, N={len(generate_fn(0, capacity))}, C={capacity}")
    print(f"{'='*70}")
    print(f"  {'Yontem':<20} {'Ort':>7} {'Gap%':>7} {'FFD>':>6} {'Sure':>8}")
    print(f"  {'-'*50}")
    print(f"  {'L2 Alt Sinir':<20} {lb_avg:>7.2f} {'0.0%':>7}")
    print(f"  {'FFD':<20} {ffd_avg:>7.2f} {ffd_gap:>6.1f}%")
    print(f"  {'DPO (greedy)':<20} {drl_g_avg:>7.2f} {drl_g_gap:>6.1f}% {drl_eq_ffd:>5}= {np.mean(drl_times):>7.3f}s")
    print(f"  {'DPO (sample@10)':<20} {drl_s_avg:>7.2f} {drl_s_gap:>6.1f}% {drl_beats_ffd:>5}< {np.mean(drl_times):>7.3f}s")
    print(f"  {'GA (1s)':<20} {ga_avg:>7.2f} {ga_gap:>6.1f}% {ga_beats_ffd:>5}< {np.mean(ga_times):>7.3f}s")

    return {
        'name': name, 'ffd': ffd_avg, 'drl_g': drl_g_avg,
        'drl_s': drl_s_avg, 'ga': ga_avg, 'lb': lb_avg,
        'drl_beats_ffd': drl_beats_ffd, 'ga_beats_ffd': ga_beats_ffd,
        'drl_eq_ffd': drl_eq_ffd
    }


# ---------------------------------------------------------------
# FARKLI DAGILIM TURLERI
# ---------------------------------------------------------------

# 1. Egitim dagilimi: Uniform(1, C)
def gen_uniform_full(seed, C):
    rng = np.random.default_rng(seed)
    return list(rng.integers(1, C+1, size=50))

# 2. Uniform(20, 80) - Orta agirlikli
def gen_uniform_mid(seed, C):
    rng = np.random.default_rng(seed)
    return list(rng.integers(20, 81, size=50))

# 3. Triplet: items ~ C/3 (FFD'nin zayif oldugu durum)
def gen_triplet(seed, C):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(60):
        # Her item ~C/3 +/- %10
        w = int(C/3 + rng.integers(-C//30, C//30+1))
        w = max(1, min(C-1, w))
        items.append(w)
    return items

# 4. Ikili: items ~ C/2 (FFD burada iyi)
def gen_binary(seed, C):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(50):
        w = int(C/2 + rng.integers(-C//10, C//10+1))
        w = max(1, min(C-1, w))
        items.append(w)
    return items

# 5. Cok kucuk itemler (C/10 civari)
def gen_small(seed, C):
    rng = np.random.default_rng(seed)
    return list(rng.integers(1, C//5+1, size=50))

# 6. Karisik: bazi buyuk, bazi kucuk
def gen_bimodal(seed, C):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(25):
        items.append(int(rng.integers(C//2, C)))  # Buyuk
    for _ in range(25):
        items.append(int(rng.integers(1, C//4)))   # Kucuk
    return items

# 7. Perfect packing mumkun (3'lu gruplar)
def gen_perfect_triple(seed, C):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(20):  # 20 kutu, her biri 3 item
        a = rng.integers(1, C//2)
        b = rng.integers(1, C - a)
        c = C - a - b
        if c > 0:
            items.extend([int(a), int(b), int(c)])
        else:
            items.extend([int(a), int(b)])
    return items


print("=" * 70)
print("ORNEK TIPI ANALIZI: DPO vs FFD vs GA")
print("=" * 70)

results = []
results.append(test_distribution("1. Egitim dagilimi: U(1,C)",       gen_uniform_full))
results.append(test_distribution("2. Orta agirlik: U(20,80)",        gen_uniform_mid))
results.append(test_distribution("3. Triplet: ~C/3",                 gen_triplet))
results.append(test_distribution("4. Ikili: ~C/2",                   gen_binary))
results.append(test_distribution("5. Kucuk itemler: U(1,C/5)",       gen_small))
results.append(test_distribution("6. Bimodal: buyuk+kucuk",          gen_bimodal))
results.append(test_distribution("7. Perfect 3'lu gruplama",         gen_perfect_triple))

print(f"\n\n{'='*70}")
print("GENEL OZET")
print(f"{'='*70}")
print(f"{'Dagilim':<30} {'FFD':>6} {'DPO_g':>6} {'DPO_s':>6} {'GA':>6} {'DPO<FFD':>7}")
print(f"{'-'*65}")
for r in results:
    print(f"{r['name'][:30]:<30} {r['ffd']:>6.2f} {r['drl_g']:>6.2f} {r['drl_s']:>6.2f} {r['ga']:>6.2f} {r['drl_beats_ffd']:>5}/50")
