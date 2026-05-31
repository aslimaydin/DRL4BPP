"""Compare DRL model vs baseline heuristics for 2D BPP."""
import sys, os, random, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment_2d import BinPacking2DEnv, generate_random_2d_instance
import torch
from rl_model import BPPActorCritic

W, H, N = 100, 100, 20
N_EVAL = 100
seeds = list(range(90000, 90000 + N_EVAL))


def run_heuristic(name, select_fn):
    groups_list = []
    for seed in seeds:
        env = BinPacking2DEnv(N, W, H, reward_type='step')
        env.reset(seed=seed)
        while not env.done:
            valid = env._get_valid_edges()
            if len(valid) == 0:
                break
            idx = select_fn(env, valid)
            env.step(idx)
        groups_list.append(env.get_num_bins())
    return np.mean(groups_list), np.std(groups_list)


def random_select(env, valid):
    return random.randint(0, len(valid) - 1)

def first_fit(env, valid):
    return 0

def best_fit(env, valid):
    best_idx, best_waste = 0, float('inf')
    for idx in range(len(valid)):
        i, j = valid[idx][0].item(), valid[idx][1].item()
        key = (min(i, j), max(i, j))
        m = env._merge_info[key]
        waste = m[0] * m[1] - (env.widths[i] * env.heights[i] + env.widths[j] * env.heights[j])
        if waste < best_waste:
            best_waste = waste
            best_idx = idx
    return best_idx

def largest_area(env, valid):
    best_idx, best_area = 0, 0
    for idx in range(len(valid)):
        i, j = valid[idx][0].item(), valid[idx][1].item()
        area = env.areas[i] + env.areas[j]
        if area > best_area:
            best_area = area
            best_idx = idx
    return best_idx

def smallest_bb(env, valid):
    best_idx, best_bb = 0, float('inf')
    for idx in range(len(valid)):
        i, j = valid[idx][0].item(), valid[idx][1].item()
        key = (min(i, j), max(i, j))
        m = env._merge_info[key]
        bb = m[0] * m[1]
        if bb < best_bb:
            best_bb = bb
            best_idx = idx
    return best_idx

def best_utilization(env, valid):
    """Pick edge that maximizes area utilization of the merged bounding box."""
    best_idx, best_util = 0, 0
    for idx in range(len(valid)):
        i, j = valid[idx][0].item(), valid[idx][1].item()
        key = (min(i, j), max(i, j))
        m = env._merge_info[key]
        bb_area = m[0] * m[1]
        actual_area = env.areas[i] + env.areas[j]
        util = actual_area / bb_area if bb_area > 0 else 0
        if util > best_util:
            best_util = util
            best_idx = idx
    return best_idx


if __name__ == '__main__':
    print("=" * 65)
    print("  2D BPP BASELINE HEURISTIC COMPARISON")
    print(f"  {N_EVAL} instances | N={N} items | Bin={W}x{H}")
    print("=" * 65)

    heuristics = [
        ("Random",               random_select),
        ("First Fit",            first_fit),
        ("Best Fit (min waste)", best_fit),
        ("Largest Area First",   largest_area),
        ("Smallest BB First",    smallest_bb),
        ("Best Utilization",     best_utilization),
    ]

    results = {}
    for name, fn in heuristics:
        avg, std = run_heuristic(name, fn)
        results[name] = (avg, std)
        print(f"  {name:<25} {avg:.2f} +/- {std:.2f} groups")

    # DRL Models
    print(f"\n  --- DRL Models ---")
    model_configs = [
        ("GCN+PPO",       "gcn", "gcn_ppo_step"),
        ("GAT+PPO",       "gat", "gat_ppo_step"),
        ("GCN+REINFORCE", "gcn", "gcn_reinforce_step"),
        ("GAT+A2C",       "gat", "gat_a2c_step"),
    ]

    best_model_avg = float('inf')
    for label, gnn, exp_name in model_configs:
        model_path = os.path.join("checkpoints_2d", exp_name, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"  {label:<25} NOT FOUND")
            continue

        model = BPPActorCritic(node_feat_dim=2, embed_dim=128,
                               n_gnn_layers=3, gnn_type=gnn)
        model.load_state_dict(torch.load(model_path, map_location='cpu',
                                         weights_only=True))
        model.eval()

        model_groups = []
        for seed in seeds:
            env = BinPacking2DEnv(N, W, H, reward_type='step')
            state = env.reset(seed=seed)
            with torch.no_grad():
                while not env.done:
                    edge_idx, _, _ = model.select_action(state, greedy=True)
                    if edge_idx < 0:
                        break
                    state, _, _, _, _ = env.step(edge_idx)
            model_groups.append(env.get_num_bins())

        avg_m = np.mean(model_groups)
        std_m = np.std(model_groups)
        print(f"  {label:<25} {avg_m:.2f} +/- {std_m:.2f} groups")
        if avg_m < best_model_avg:
            best_model_avg = avg_m

    # Area lower bound
    lbs = []
    for seed in seeds:
        items = generate_random_2d_instance(N, W, H, seed=seed)
        total = sum(w * h for w, h in items)
        lbs.append(max(1, int(np.ceil(total / (W * H)))))

    print(f"\n  --- Reference ---")
    print(f"  {'Area Lower Bound':<25} {np.mean(lbs):.2f} (theoretical optimum)")

    # Summary
    best_h_name = min(results, key=lambda k: results[k][0])
    best_h_avg = results[best_h_name][0]
    if best_model_avg < float('inf'):
        imp = (best_h_avg - best_model_avg) / best_h_avg * 100
        print(f"\n  Best heuristic: {best_h_name} ({best_h_avg:.2f})")
        print(f"  Best DRL model: {best_model_avg:.2f}")
        print(f"  DRL improvement: {imp:+.1f}%")
    print("=" * 65)
