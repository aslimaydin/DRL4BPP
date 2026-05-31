"""
rl_train_2d_hybrid.py
=====================
Hibrit 2B-KPP PPO Eğitim Betiği.

Koordinat tabanlı yerleştirme + çizge gösterimi ile
2B kutu paketleme problemi için PPO eğitimi.
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
from typing import List, Dict

from rl_environment_2d_hybrid import BinPacking2DHybridEnv
from rl_model_2d_hybrid import BPP2DHybridActorCritic


def collect_episode(env, model, greedy=False):
    """Bir episode toplama."""
    state = env.reset()
    
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    
    while not env.done:
        with torch.no_grad():
            action_idx, log_prob, value = model.select_action(state, greedy=greedy)
        
        if action_idx == -1:
            break
        
        states.append({k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in state.items()})
        actions.append(action_idx)
        
        state, reward, done, _, info = env.step(action_idx)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
    
    return states, actions, rewards, log_probs, values


def compute_returns(rewards, gamma=0.99):
    """Diskontlu getirileri hesapla."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def ppo_update(model, optimizer, states, actions, old_log_probs, returns,
               advantages, clip_epsilon=0.2, entropy_coef=0.01,
               value_coef=0.5, n_epochs=4):
    """PPO güncelleme adımı."""
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0
    
    for _ in range(n_epochs):
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            old_lp = old_log_probs[i]
            ret = returns[i]
            adv = advantages[i]
            
            # Yeniden değerlendir
            log_prob, value, entropy = model.evaluate_action(state, action)
            
            # Oran
            ratio = torch.exp(log_prob - old_lp.detach())
            
            # Kırpılmış güvenli bölge
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2)
            
            # Değer kaybı
            value_loss = F.mse_loss(value, torch.tensor(ret, device=value.device))
            
            # Toplam kayıp
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1
    
    return (total_policy_loss / max(n_updates, 1),
            total_value_loss / max(n_updates, 1),
            total_entropy / max(n_updates, 1))


def evaluate(env, model, n_episodes=16):
    """Modeli değerlendir."""
    placements = []
    utilizations = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=1000 + ep)
        while not env.done:
            with torch.no_grad():
                action_idx, _, _ = model.select_action(state, greedy=True)
            if action_idx == -1:
                break
            state, _, _, _, _ = env.step(action_idx)
        
        placements.append(env.n_placed)
        utilizations.append(env.get_utilization())
    
    return {
        'mean_placed': np.mean(placements),
        'mean_utilization': np.mean(utilizations),
        'min_placed': np.min(placements),
        'max_placed': np.max(placements),
    }


def train(args):
    """Ana eğitim döngüsü."""
    print(f"{'='*60}")
    print(f"Hibrit 2B-KPP PPO Eğitimi")
    print(f"{'='*60}")
    print(f"GNN: {args.gnn_type} | N: {args.n_items} | "
          f"Kutu: {args.bin_width}x{args.bin_height}")
    print(f"Epochs: {args.n_epochs} | Batch: {args.batch_size} | "
          f"LR: {args.lr}")
    print(f"{'='*60}")
    
    # Ortam
    env = BinPacking2DHybridEnv(
        n_items=args.n_items,
        bin_width=args.bin_width,
        bin_height=args.bin_height,
        reward_type=args.reward_type,
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BPP2DHybridActorCritic(
        node_feat_dim=6,
        embed_dim=args.embed_dim,
        n_gnn_layers=args.n_gnn_layers,
        gnn_type=args.gnn_type,
        bin_width=args.bin_width,
        dropout=0.1,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Cihaz: {device}")
    
    # Checkpoint dizini
    ckpt_dir = os.path.join(os.path.dirname(__file__),
                            'checkpoints_2d_hybrid',
                            f'{args.gnn_type}_ppo')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Eğitim geçmişi
    history = {
        'epoch': [], 'mean_placed': [], 'mean_utilization': [],
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'best_placed': [], 'best_utilization': [],
    }
    best_placed = 0
    best_utilization = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        
        # Batch toplama
        all_states = []
        all_actions = []
        all_returns = []
        all_old_log_probs = []
        all_advantages = []
        
        for _ in range(args.batch_size):
            states, actions, rewards, log_probs, values = collect_episode(env, model)
            
            if len(rewards) == 0:
                continue
            
            returns = compute_returns(rewards, gamma=args.gamma)
            
            # Advantage hesapla
            advantages = []
            for r, v in zip(returns, values):
                adv = r - v.item()
                advantages.append(adv)
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)
            all_old_log_probs.extend(log_probs)
            all_advantages.extend(advantages)
        
        if len(all_states) == 0:
            continue
        
        # Advantage normalize
        adv_tensor = torch.tensor(all_advantages)
        if adv_tensor.std() > 1e-8:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        all_advantages = adv_tensor.tolist()
        
        # PPO güncelleme
        p_loss, v_loss, ent = ppo_update(
            model, optimizer, all_states, all_actions,
            all_old_log_probs, all_returns, all_advantages,
            clip_epsilon=0.2, entropy_coef=0.01,
            n_epochs=args.ppo_epochs,
        )
        
        # Değerlendirme
        if epoch % args.eval_interval == 0 or epoch == 1:
            model.eval()
            eval_results = evaluate(env, model, n_episodes=16)
            
            mean_placed = eval_results['mean_placed']
            mean_util = eval_results['mean_utilization']
            
            if mean_placed > best_placed:
                best_placed = mean_placed
                best_utilization = mean_util
                # Checkpoint kaydet
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_placed': best_placed,
                    'best_utilization': best_utilization,
                    'args': vars(args),
                }, os.path.join(ckpt_dir, 'best_model.pt'))
            
            history['epoch'].append(epoch)
            history['mean_placed'].append(mean_placed)
            history['mean_utilization'].append(mean_util)
            history['policy_loss'].append(p_loss)
            history['value_loss'].append(v_loss)
            history['entropy'].append(ent)
            history['best_placed'].append(best_placed)
            history['best_utilization'].append(best_utilization)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}/{args.n_epochs} | "
                  f"Placed: {mean_placed:.1f}/{args.n_items} | "
                  f"Util: {mean_util:.1%} | "
                  f"Best: {best_placed:.1f} ({best_utilization:.1%}) | "
                  f"PLoss: {p_loss:.4f} | Ent: {ent:.3f} | "
                  f"T: {elapsed:.0f}s")
    
    # Final kaydet
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.n_epochs,
        'best_placed': best_placed,
        'best_utilization': best_utilization,
    }, os.path.join(ckpt_dir, 'final_model.pt'))
    
    # Geçmişi kaydet
    with open(os.path.join(ckpt_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Eğitim tamamlandı!")
    print(f"Süre: {total_time/60:.1f} dakika")
    print(f"En iyi: {best_placed:.1f} yerleşim, {best_utilization:.1%} utilization")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")
    
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hibrit 2B-KPP PPO Eğitimi')
    
    # Problem parametreleri
    parser.add_argument('--n_items', type=int, default=20)
    parser.add_argument('--bin_width', type=int, default=100)
    parser.add_argument('--bin_height', type=int, default=100)
    parser.add_argument('--reward_type', type=str, default='step')
    
    # Model parametreleri
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_gnn_layers', type=int, default=3)
    
    # Eğitim parametreleri
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=50)
    
    args = parser.parse_args()
    train(args)
