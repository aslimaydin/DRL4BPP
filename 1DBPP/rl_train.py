"""
rl_train.py
===========
Unified training script. GNN type, algorithm,
reward function and other hyperparameters can be set via command line.

Usage:
    python rl_train.py --gnn_type gat --algorithm reinforce --reward_type step
    python rl_train.py --gnn_type gcn --algorithm ppo --n_items 50 --epochs 3000
    python rl_train.py --gnn_type gin --algorithm dqn --reward_type terminal
"""



import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from rl_model import BPPActorCritic
from rl_environment import BinPackingGraphEnv
from rl_algorithms import create_algorithm
from rl_utils import first_fit_decreasing, best_fit_decreasing, generate_random_instance


def parse_args():
    parser = argparse.ArgumentParser(
        description="1D BPP Graph-RL Training Script"
    )
    
    # ── Model ──
    parser.add_argument('--gnn_type', type=str, default='gat',
                        choices=['gcn', 'gat', 'gin'],
                        help='GNN architecture')
    parser.add_argument('--agg_type', type=str, default='mean',
                        choices=['sum', 'mean', 'max', 'mlp'],
                        help='Aggregation function')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--n_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of GAT attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # ── Algorithm ──
    parser.add_argument('--algorithm', type=str, default='reinforce',
                        choices=['reinforce', 'a2c', 'ppo', 'dqn', 'sac', 'sarsa'],
                        help='RL algorithm')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                        help='Critic learning rate (REINFORCE/SAC)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO inner epochs')
    
    # ── Problem ──
    parser.add_argument('--n_items', type=int, default=50,
                        help='Number of items')
    parser.add_argument('--capacity', type=int, default=100,
                        help='Bin capacity')
    parser.add_argument('--reward_type', type=str, default='step',
                        choices=['step', 'terminal'],
                        help='Reward function type')
    
    # ── Training ──
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Total number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Episode batch size')
    parser.add_argument('--val_size', type=int, default=20,
                        help='Number of validation instances')
    parser.add_argument('--val_interval', type=int, default=50,
                        help='Validation frequency (epochs)')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Checkpoint save frequency (epochs)')
    
    # ── System ──
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (checkpoint directory)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def validate(model, env, n_instances, capacity, n_items, device):
    """
    Validation: measures bin count with greedy decoding, compares against FFD/BFD.
    """
    model.eval()
    
    model_bins_list = []
    ffd_bins_list = []
    bfd_bins_list = []
    
    for i in range(n_instances):
        # Test with fixed seed for reproducibility
        items = generate_random_instance(n_items, capacity, seed=10000 + i)
        
        # FFD/BFD baseline
        ffd_bins, _ = first_fit_decreasing(items, capacity)
        bfd_bins, _ = best_fit_decreasing(items, capacity)
        
        # Model (greedy)
        env_val = BinPackingGraphEnv(n_items=len(items), capacity=capacity,
                                     reward_type='step')
        state = env_val.reset(items=items)
        
        with torch.no_grad():
            while not env_val.done:
                edge_idx, _, _ = model.select_action(state, greedy=True)
                if edge_idx < 0:
                    break
                state, _, _, _, _ = env_val.step(edge_idx)
        
        model_bins = env_val.get_num_bins()
        
        model_bins_list.append(model_bins)
        ffd_bins_list.append(ffd_bins)
        bfd_bins_list.append(bfd_bins)
    
    return {
        'model_avg_bins': np.mean(model_bins_list),
        'ffd_avg_bins': np.mean(ffd_bins_list),
        'bfd_avg_bins': np.mean(bfd_bins_list),
        'model_bins': model_bins_list,
        'ffd_bins': ffd_bins_list,
        'bfd_bins': bfd_bins_list,
    }


def main():
    args = parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.gnn_type}_{args.algorithm}_{args.reward_type}"
    
    checkpoint_dir = os.path.join('checkpoints', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ── Build model ──
    use_q = args.algorithm in ['dqn', 'sac', 'sarsa']
    
    model = BPPActorCritic(
        node_feat_dim=2,
        embed_dim=args.embed_dim,
        n_gnn_layers=args.n_gnn_layers,
        gnn_type=args.gnn_type,
        n_heads=args.n_heads,
        agg_type=args.agg_type,
        policy_hidden=args.embed_dim,
        value_hidden=args.embed_dim,
        dropout=args.dropout,
        use_q_network=use_q,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.gnn_type.upper()} + {args.agg_type} aggregation")
    print(f"Parameters: {n_params:,}")
    
    # ── Build algorithm ──
    alg_kwargs = {'gamma': args.gamma}
    
    if args.algorithm == 'reinforce':
        alg_kwargs.update({
            'lr_actor': args.lr,
            'lr_critic': args.lr_critic,
            'entropy_coef': args.entropy_coef,
        })
    elif args.algorithm == 'a2c':
        alg_kwargs.update({
            'lr': args.lr,
            'entropy_coef': args.entropy_coef,
        })
    elif args.algorithm == 'ppo':
        alg_kwargs.update({
            'lr': args.lr,
            'clip_epsilon': args.clip_epsilon,
            'entropy_coef': args.entropy_coef,
            'ppo_epochs': args.ppo_epochs,
        })
    elif args.algorithm == 'dqn':
        alg_kwargs.update({
            'lr': args.lr,
            'batch_size': args.batch_size,
        })
    elif args.algorithm == 'sac':
        alg_kwargs.update({
            'lr_actor': args.lr,
            'lr_critic': args.lr_critic,
            'batch_size': args.batch_size,
        })
    elif args.algorithm == 'sarsa':
        alg_kwargs.update({
            'lr': args.lr,
            'batch_size': args.batch_size,
        })
    
    algorithm = create_algorithm(args.algorithm, model, device=device, **alg_kwargs)
    
    print(f"Algoritma: {args.algorithm.upper()}")
    print(f"Reward: {args.reward_type}")
    print(f"Items: N={args.n_items}, C={args.capacity}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 60)
    
    # ── Environment ──
    env = BinPackingGraphEnv(
        n_items=args.n_items,
        capacity=args.capacity,
        reward_type=args.reward_type,
    )
    
    # ── Training loop ──
    best_val_bins = float('inf')
    training_log = []
    
    for epoch in tqdm(range(args.epochs), desc='Training'):
        epoch_start = time.time()
        
        # Collect batch episodes
        episodes = []
        epoch_bins = []
        epoch_rewards = []
        
        for b in range(args.batch_size):
            state = env.reset(seed=epoch * args.batch_size + b)
            episode = algorithm.collect_episode(env)
            episodes.append(episode)
            epoch_bins.append(episode.get('n_bins', 0))
            epoch_rewards.append(episode.get('total_reward', 
                                            sum(episode.get('rewards', []))))
        
        # Update model
        if args.algorithm in ['dqn', 'sac', 'sarsa']:
            # Off-policy: sample from replay buffer
            loss_info = algorithm.update()
        else:
            # On-policy: update with collected episodes
            loss_info = algorithm.update(episodes)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            'epoch': epoch,
            'avg_bins': np.mean(epoch_bins),
            'avg_reward': np.mean(epoch_rewards),
            'time': epoch_time,
            **loss_info,
        }
        training_log.append(log_entry)
        
        # Update progress bar
        if (epoch + 1) % 10 == 0:
            tqdm.write(
                f"Epoch {epoch+1:4d} | "
                f"Bins: {log_entry['avg_bins']:.1f} | "
                f"Reward: {log_entry['avg_reward']:.2f} | "
                f"Loss: {loss_info.get('loss', loss_info.get('actor_loss', 0)):.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
        
        # ── Validation ──
        if (epoch + 1) % args.val_interval == 0:
            val_results = validate(
                model, env, args.val_size, args.capacity, args.n_items, device
            )
            
            tqdm.write(
                f"\n  [VAL] Model: {val_results['model_avg_bins']:.1f} bins | "
                f"FFD: {val_results['ffd_avg_bins']:.1f} | "
                f"BFD: {val_results['bfd_avg_bins']:.1f}"
            )
            
            # Save best model
            if val_results['model_avg_bins'] < best_val_bins:
                best_val_bins = val_results['model_avg_bins']
                algorithm.save(os.path.join(checkpoint_dir, 'best_model.pth'))
                tqdm.write(f"  * New best model: {best_val_bins:.1f} bins")
        
        # ── Checkpoint ──
        if (epoch + 1) % args.save_interval == 0:
            algorithm.save(
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    # ── Final save ──
    algorithm.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Save training log
    import json
    log_path = os.path.join(checkpoint_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation: {best_val_bins:.1f} bins")
    print(f"Log: {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
