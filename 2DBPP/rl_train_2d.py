"""
rl_train_2d.py
==============
Training script for 2D Bin Packing Problem.
Reuses GNN model and RL algorithms from the 1D project.

Usage:
    python rl_train_2d.py --gnn_type gat --algorithm ppo --epochs 2000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
import argparse
from tqdm import tqdm

from rl_environment_2d import BinPacking2DEnv, generate_random_2d_instance, bottom_left_decreasing_area
from rl_model import BPPActorCritic
from rl_algorithms import create_algorithm


def parse_args():
    parser = argparse.ArgumentParser(description="2D BPP DRL Training")
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--algorithm', type=str, default='reinforce',
                        choices=['reinforce', 'a2c', 'ppo', 'dqn', 'sac', 'sarsa'])
    parser.add_argument('--reward_type', type=str, default='step',
                        choices=['step', 'terminal'])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_items', type=int, default=20)
    parser.add_argument('--bin_width', type=int, default=100)
    parser.add_argument('--bin_height', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_gnn_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--experiment_name', type=str, default='test_2d')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--val_episodes', type=int, default=16)
    parser.add_argument('--val_interval', type=int, default=10)
    return parser.parse_args()


def validate(model, args, device, n_episodes=16, seed_start=90000):
    """Run greedy validation episodes."""
    model.eval()
    model_bins_list = []
    baseline_bins_list = []

    for i in range(n_episodes):
        seed = seed_start + i
        env = BinPacking2DEnv(
            n_items=args.n_items, bin_width=args.bin_width,
            bin_height=args.bin_height, reward_type=args.reward_type
        )
        state = env.reset(seed=seed)

        # Model greedy
        with torch.no_grad():
            while not env.done:
                edge_idx, _, _ = model.select_action(state, greedy=True)
                if edge_idx < 0:
                    break
                state, _, _, _, _ = env.step(edge_idx)
        model_bins_list.append(env.get_num_bins())

        # BLDA baseline
        items = generate_random_2d_instance(
            args.n_items, args.bin_width, args.bin_height, seed=seed
        )
        bl_util = bottom_left_decreasing_area(items, args.bin_width, args.bin_height)
        baseline_bins_list.append(bl_util)

    return {
        'model_avg_groups': np.mean(model_bins_list),
        'model_std_groups': np.std(model_bins_list),
        'baseline_avg_util': np.mean(baseline_bins_list),
    }


def main():
    args = parse_args()

    # Device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # Checkpoint dir
    checkpoint_dir = os.path.join('checkpoints_2d', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print(f"  2D BPP Training: {args.gnn_type.upper()} + {args.algorithm.upper()}")
    print("=" * 60)
    print(f"Device: {device}")

    # Model
    use_q = args.algorithm in ['dqn', 'sac', 'sarsa']
    model = BPPActorCritic(
        node_feat_dim=2,
        embed_dim=args.embed_dim,
        n_gnn_layers=args.n_gnn_layers,
        gnn_type=args.gnn_type,
        use_q_network=use_q,
    ).to(device)

    agg_name = 'mean'
    if hasattr(model, 'aggregator'):
        agg_name = model.aggregator.agg_type
    print(f"Model: {args.gnn_type.upper()} + {agg_name} aggregation")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Reward: {args.reward_type}")
    print(f"Items: N={args.n_items}, Bin={args.bin_width}x{args.bin_height}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 60)

    # Algorithm
    algorithm = create_algorithm(args.algorithm, model, device=device)
    is_off_policy = args.algorithm in ['dqn', 'sac', 'sarsa']

    # Training log
    training_log = []
    best_val = float('inf')

    # Training loop
    for epoch in tqdm(range(args.epochs), desc='Training'):
        model.train()
        epoch_start = time.time()

        epoch_groups = []
        epoch_rewards = []
        episodes = []

        for b in range(args.batch_size):
            env = BinPacking2DEnv(
                n_items=args.n_items, bin_width=args.bin_width,
                bin_height=args.bin_height, reward_type=args.reward_type
            )
            env.reset(seed=epoch * args.batch_size + b)

            episode = algorithm.collect_episode(env, greedy=False)
            episodes.append(episode)
            epoch_groups.append(env.get_num_bins())

            total_r = sum(episode.get('rewards', [])) + episode.get('total_reward', 0)
            epoch_rewards.append(total_r)

        # Update
        if is_off_policy:
            loss_info = algorithm.update()
        else:
            loss_info = algorithm.update(episodes)

        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            'epoch': epoch,
            'avg_groups': float(np.mean(epoch_groups)),
            'avg_reward': float(np.mean(epoch_rewards)),
            'time': epoch_time,
            **loss_info,
        }

        # Validation
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_info = validate(model, args, device, n_episodes=args.val_episodes)
            log_entry.update(val_info)

            avg_groups = val_info['model_avg_groups']
            if avg_groups < best_val:
                best_val = avg_groups
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, 'best_model.pth'))

        training_log.append(log_entry)

        # Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            with open(os.path.join(checkpoint_dir, 'training_log.json'), 'w') as f:
                json.dump(training_log, f, indent=1)

    # Save final
    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, 'final_model.pth'))
    with open(os.path.join(checkpoint_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=1)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation: {best_val:.1f} groups")
    print(f"Log: {os.path.join(checkpoint_dir, 'training_log.json')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
