"""
rl_algorithms.py
================
5 Derin Pekiştirmeli Öğrenme algoritması implementasyonu.

1. REINFORCE    — Monte Carlo Policy Gradient + Critic Baseline
2. A2C          — Advantage Actor-Critic (n-step return)
3. PPO          — Proximal Policy Optimization (clipped objective)
4. DQN          — Deep Q-Network (experience replay + target network)
5. SAC          — Soft Actor-Critic (maximum entropy RL)

Tümü ortak BaseAlgorithm interface'ini paylaşır.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from rl_model import BPPActorCritic
from rl_environment import BinPackingGraphEnv
from rl_utils import compute_returns, compute_advantages


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE BUFFER
# ─────────────────────────────────────────────────────────────────────────────

Transition = namedtuple('Transition', 
                        ['state', 'edge_idx', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """DQN ve SAC için experience replay buffer."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# BASE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

class BaseAlgorithm:
    """Tüm DPÖ algoritmaları için ortak interface."""
    
    def __init__(self, model: BPPActorCritic, lr: float = 3e-4,
                 gamma: float = 1.0, device: str = 'cpu'):
        self.model = model.to(device)
        self.gamma = gamma
        self.device = device
        self.training_step = 0
    
    def collect_episode(self, env: BinPackingGraphEnv) -> Dict:
        """Bir episode çalıştırır ve deneyimleri toplar."""
        raise NotImplementedError
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """Toplanan deneyimlerle model günceller."""
        raise NotImplementedError
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_step': self.training_step,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REINFORCE
# ─────────────────────────────────────────────────────────────────────────────

class REINFORCE(BaseAlgorithm):
    """
    Monte Carlo Policy Gradient + Critic Baseline.
    
    Loss = -E[A_t · log π(a_t|s_t)] + c_v · MSE(V(s_t), G_t)
    
    A_t = G_t - V(s_t)  (advantage)
    G_t = Σ γ^k r_{t+k} (monte carlo return)
    """
    
    def __init__(self, model: BPPActorCritic, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, gamma: float = 1.0,
                 entropy_coef: float = 0.01, device: str = 'cpu'):
        super().__init__(model, lr_actor, gamma, device)
        
        self.entropy_coef = entropy_coef
        
        # Ayrı optimizer'lar
        self.actor_optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': lr_actor},
            {'params': model.aggregator.parameters(), 'lr': lr_actor},
            {'params': model.policy.parameters(), 'lr': lr_actor},
        ])
        
        self.critic_optimizer = optim.Adam(
            model.value.parameters(), lr=lr_critic
        )
    
    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """Bir episode çalıştır ve log_probs, rewards, values topla."""
        state = env.get_state()
        
        log_probs = []
        rewards = []
        values = []
        entropies = []
        states = []
        actions = []
        
        while not env.done:
            self.model.eval()
            edge_idx, log_prob, value = self.model.select_action(
                state, greedy=greedy)
            
            if edge_idx < 0:
                break
            
            states.append(state)
            actions.append(edge_idx)
            
            next_state, reward, done, _, info = env.step(edge_idx)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            
            state = next_state
        
        return {
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'states': states,
            'actions': actions,
            'n_bins': env.get_num_bins(),
        }
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """Batch episode ile model güncelle."""
        self.model.train()
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_steps = 0
        
        for episode in episodes:
            if len(episode['rewards']) == 0:
                continue
            
            # Returns hesapla
            returns = compute_returns(episode['rewards'], self.gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)
            
            # Values tensor
            values_tensor = torch.stack(episode['values'])
            
            # Advantages
            advantages = returns_tensor - values_tensor.detach()
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Log probs tensor
            log_probs_tensor = torch.stack(episode['log_probs'])
            
            # Actor loss: -E[A_t · log π(a_t|s_t)]
            actor_loss = -(advantages * log_probs_tensor).mean()
            
            # Critic loss: MSE(V(s_t), G_t)
            critic_loss = F.mse_loss(values_tensor, returns_tensor)
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            n_steps += len(episode['rewards'])
        
        if n_steps == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Birleşik loss (tek backward — encoder paylaşıldığı için gerekli)
        avg_actor_loss = total_actor_loss / len(episodes)
        avg_critic_loss = total_critic_loss / len(episodes)
        combined_loss = avg_actor_loss + 0.5 * avg_critic_loss
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        combined_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        self.training_step += 1
        
        return {
            'actor_loss': avg_actor_loss.item(),
            'critic_loss': avg_critic_loss.item(),
            'n_steps': n_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. A2C (Advantage Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────

class A2C(BaseAlgorithm):
    """
    Advantage Actor-Critic.
    
    REINFORCE'a benzer ama TD(n) veya GAE ile advantage hesaplar.
    Actor ve Critic eşzamanlı güncellenir.
    
    Loss = actor_loss + c_v · critic_loss - c_e · entropy
    """
    
    def __init__(self, model: BPPActorCritic, lr: float = 3e-4,
                 gamma: float = 1.0, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, n_steps: int = 5,
                 device: str = 'cpu'):
        super().__init__(model, lr, gamma, device)
        
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """REINFORCE ile aynı toplama mekanizması."""
        state = env.get_state()
        
        log_probs = []
        rewards = []
        values = []
        entropies = []
        states = []
        actions = []
        
        while not env.done:
            self.model.eval()
            edge_idx, log_prob, value = self.model.select_action(
                state, greedy=greedy)
            
            if edge_idx < 0:
                break
            
            # Entropy hesapla
            _, _, entropy = self.model.evaluate_action(state, edge_idx)
            
            states.append(state)
            actions.append(edge_idx)
            
            next_state, reward, done, _, info = env.step(edge_idx)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            entropies.append(entropy)
            
            state = next_state
        
        return {
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'entropies': entropies,
            'states': states,
            'actions': actions,
            'n_bins': env.get_num_bins(),
        }
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """A2C güncelleme."""
        self.model.train()
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        n_episodes = 0
        
        for episode in episodes:
            if len(episode['rewards']) == 0:
                continue
            
            returns = compute_returns(episode['rewards'], self.gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)
            
            values_tensor = torch.stack(episode['values'])
            log_probs_tensor = torch.stack(episode['log_probs'])
            entropies_tensor = torch.stack(episode['entropies'])
            
            advantages = returns_tensor - values_tensor.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            actor_loss = -(advantages * log_probs_tensor).mean()
            critic_loss = F.mse_loss(values_tensor, returns_tensor)
            entropy_loss = -entropies_tensor.mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            total_loss = total_loss + loss
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()
            n_episodes += 1
        
        if n_episodes == 0:
            return {'loss': 0.0}
        
        avg_loss = total_loss / n_episodes
        self.optimizer.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return {
            'loss': avg_loss.item(),
            'actor_loss': total_actor_loss / n_episodes,
            'critic_loss': total_critic_loss / n_episodes,
            'entropy': -total_entropy_loss / n_episodes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. PPO (Proximal Policy Optimization)
# ─────────────────────────────────────────────────────────────────────────────

class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization (Schulman et al., 2017).
    
    Clipped surrogate objective:
    L = min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)
    r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """
    
    def __init__(self, model: BPPActorCritic, lr: float = 3e-4,
                 gamma: float = 1.0, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 ppo_epochs: int = 4, device: str = 'cpu'):
        super().__init__(model, lr, gamma, device)
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """Episode topla + old log_probs kaydet."""
        state = env.get_state()
        
        log_probs = []
        rewards = []
        values = []
        states = []
        actions = []
        
        while not env.done:
            self.model.eval()
            edge_idx, log_prob, value = self.model.select_action(
                state, greedy=greedy)
            
            if edge_idx < 0:
                break
            
            states.append(state)
            actions.append(edge_idx)
            
            next_state, reward, done, _, info = env.step(edge_idx)
            
            log_probs.append(log_prob.detach())  # Old policy log probs
            rewards.append(reward)
            values.append(value.detach())
            
            state = next_state
        
        return {
            'old_log_probs': log_probs,
            'rewards': rewards,
            'old_values': values,
            'states': states,
            'actions': actions,
            'n_bins': env.get_num_bins(),
        }
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """PPO multi-epoch güncelleme."""
        self.model.train()
        
        # Tüm episode'ları düzleştir
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        
        for episode in episodes:
            if len(episode['rewards']) == 0:
                continue
            
            returns = compute_returns(episode['rewards'], self.gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)
            
            old_values = torch.stack(episode['old_values'])
            advantages = returns_tensor - old_values
            
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_old_log_probs.extend(episode['old_log_probs'])
            all_returns.append(returns_tensor)
            all_advantages.append(advantages)
        
        if len(all_states) == 0:
            return {'loss': 0.0}
        
        all_returns = torch.cat(all_returns)
        all_advantages = torch.cat(all_advantages)
        old_log_probs = torch.stack(all_old_log_probs)
        
        total_loss = 0.0
        
        # PPO multi-epoch
        for _ in range(self.ppo_epochs):
            epoch_loss = torch.tensor(0.0, device=self.device)
            
            for idx in range(len(all_states)):
                state = all_states[idx]
                action = all_actions[idx]
                
                new_log_prob, value, entropy = self.model.evaluate_action(
                    state, action)
                
                # Ratio
                ratio = torch.exp(new_log_prob - old_log_probs[idx])
                
                # Clipped surrogate
                adv = all_advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * adv
                
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = F.mse_loss(value, all_returns[idx])
                entropy_loss = -entropy
                
                step_loss = actor_loss + self.value_coef * critic_loss + \
                           self.entropy_coef * entropy_loss
                epoch_loss = epoch_loss + step_loss
            
            avg_epoch_loss = epoch_loss / len(all_states)
            self.optimizer.zero_grad()
            avg_epoch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += avg_epoch_loss.item()
        
        self.training_step += 1
        
        return {
            'loss': total_loss / self.ppo_epochs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. DQN (Deep Q-Network)
# ─────────────────────────────────────────────────────────────────────────────

class DQN(BaseAlgorithm):
    """
    Deep Q-Network (Mnih et al., 2015).
    
    Graph-MDP'ye uyarlanmış:
    - Q(s, a) = QNet(state_vector, action_embedding)
    - ε-greedy aksiyon seçimi
    - Experience replay
    - Target network (periyodik güncelleme)
    """
    
    def __init__(self, model: BPPActorCritic, lr: float = 1e-3,
                 gamma: float = 1.0, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: int = 5000,
                 buffer_size: int = 50000, batch_size: int = 32,
                 target_update_freq: int = 100, device: str = 'cpu'):
        # DQN modeli Q-network ile oluşturulmalı
        if not model.use_q_network:
            raise ValueError("DQN için model use_q_network=True ile oluşturulmalı.")
        
        super().__init__(model, lr, gamma, device)
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Target network
        self.target_model = deepcopy(model).to(device)
        self.target_model.eval()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def get_epsilon(self) -> float:
        """Mevcut ε değeri (decaying)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.training_step / self.epsilon_decay)
    
    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """ε-greedy ile episode çalıştır, buffer'a ekle."""
        state = env.get_state()
        total_reward = 0.0
        
        while not env.done:
            epsilon = self.get_epsilon() if not greedy else 0.0
            
            with torch.no_grad():
                q_values, valid_edges = self.model.get_q_values(state)
            
            if len(valid_edges) == 0:
                break
            
            # ε-greedy
            if random.random() < epsilon:
                edge_idx = random.randrange(len(valid_edges))
            else:
                edge_idx = torch.argmax(q_values).item()
            
            next_state, reward, done, _, info = env.step(edge_idx)
            total_reward += reward
            
            # Buffer'a ekle
            self.replay_buffer.push(state, edge_idx, reward, next_state, done)
            
            state = next_state
        
        return {
            'total_reward': total_reward,
            'n_bins': env.get_num_bins(),
        }
    
    def update(self, episodes: List[Dict] = None) -> Dict[str, float]:
        """Replay buffer'dan sample alıp güncelle."""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}
        
        self.model.train()
        
        transitions = self.replay_buffer.sample(self.batch_size)
        
        total_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        for trans in transitions:
            state, edge_idx, reward, next_state, done = trans
            
            # Current Q value
            q_values, _ = self.model.get_q_values(state)
            if edge_idx >= len(q_values):
                continue
            current_q = q_values[edge_idx]
            
            # Target Q value
            with torch.no_grad():
                if done:
                    target_q = torch.tensor(reward, dtype=torch.float, device=self.device)
                else:
                    next_q_values, next_edges = self.target_model.get_q_values(next_state)
                    if len(next_q_values) > 0:
                        max_next_q = next_q_values.max()
                    else:
                        max_next_q = torch.tensor(0.0, device=self.device)
                    target_q = reward + self.gamma * max_next_q
            
            loss = F.mse_loss(current_q, target_q)
            total_loss = total_loss + loss
            valid_count += 1
        
        if valid_count == 0:
            return {'loss': 0.0}
        
        avg_loss = total_loss / valid_count
        self.optimizer.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Target network güncelle
        if self.training_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return {
            'loss': avg_loss.item(),
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAC (Soft Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────

class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic (Haarnoja et al., 2018).
    
    Maximum entropy RL:
    J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
    
    Actor: π_θ parametrize edilmiş politika
    Critic: Q_φ(s, a) değer fonksiyonu
    Entropy: α otomatik ayarlı (opsiyonel)
    """
    
    def __init__(self, model: BPPActorCritic, lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3, gamma: float = 1.0,
                 alpha: float = 0.2, auto_alpha: bool = True,
                 buffer_size: int = 50000, batch_size: int = 32,
                 tau: float = 0.005, device: str = 'cpu'):
        if not model.use_q_network:
            raise ValueError("SAC için model use_q_network=True ile oluşturulmalı.")
        
        super().__init__(model, lr_actor, gamma, device)
        
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.tau = tau  # Soft target update coefficient
        self.batch_size = batch_size
        
        # Target network (soft update)
        self.target_model = deepcopy(model).to(device)
        self.target_model.eval()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Optimizers
        self.actor_optimizer = optim.Adam([
            {'params': model.encoder.parameters()},
            {'params': model.aggregator.parameters()},
            {'params': model.policy.parameters()},
        ], lr=lr_actor)
        
        self.critic_optimizer = optim.Adam([
            {'params': model.q_net.parameters()},
            {'params': model.value.parameters()},
        ], lr=lr_critic)
        
        # Auto-tuning alpha
        if auto_alpha:
            self.target_entropy = -1.0  # Heuristic
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
    
    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """Stochastic policy ile episode çalıştır."""
        state = env.get_state()
        total_reward = 0.0
        
        while not env.done:
            with torch.no_grad():
                if greedy:
                    edge_idx, _, _ = self.model.select_action(state, greedy=True)
                else:
                    edge_idx, _, _ = self.model.select_action(state, greedy=False)
            
            if edge_idx < 0:
                break
            
            next_state, reward, done, _, info = env.step(edge_idx)
            total_reward += reward
            
            self.replay_buffer.push(state, edge_idx, reward, next_state, done)
            state = next_state
        
        return {
            'total_reward': total_reward,
            'n_bins': env.get_num_bins(),
        }
    
    def update(self, episodes: List[Dict] = None) -> Dict[str, float]:
        """SAC güncelleme: actor, critic, ve alpha."""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}
        
        self.model.train()
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # ── Critic güncelleme ──
        critic_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0
        
        for trans in transitions:
            state, edge_idx, reward, next_state, done = trans
            
            q_values, _ = self.model.get_q_values(state)
            if edge_idx >= len(q_values):
                continue
            current_q = q_values[edge_idx]
            
            with torch.no_grad():
                if done:
                    target_q = torch.tensor(reward, dtype=torch.float, device=self.device)
                else:
                    # Next state value with entropy
                    next_node_feats = next_state['node_features'].to(self.device)
                    next_adj = next_state['adj'].to(self.device)
                    next_valid = next_state['valid_edges'].to(self.device)
                    
                    if len(next_valid) > 0:
                        next_emb, next_sv = self.target_model.encode(next_node_feats, next_adj)
                        next_log_probs = self.target_model.policy(next_emb, next_valid)
                        next_probs = torch.exp(next_log_probs)
                        
                        next_q_values, _ = self.target_model.get_q_values(next_state)
                        next_v = (next_probs * (next_q_values - self.alpha * next_log_probs)).sum()
                    else:
                        next_v = torch.tensor(0.0, device=self.device)
                    
                    target_q = reward + self.gamma * next_v
            
            critic_loss = critic_loss + F.mse_loss(current_q, target_q)
            valid_count += 1
        
        if valid_count > 0:
            avg_critic_loss = critic_loss / valid_count
            self.critic_optimizer.zero_grad()
            avg_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
        
        # ── Actor güncelleme ──
        actor_loss = torch.tensor(0.0, device=self.device)
        alpha_loss_val = 0.0
        actor_count = 0
        
        for trans in transitions:
            state = trans.state
            valid_edges = state['valid_edges'].to(self.device)
            
            if len(valid_edges) == 0:
                continue
            
            node_emb, state_vec = self.model.encode(
                state['node_features'].to(self.device), state['adj'].to(self.device))
            log_probs = self.model.policy(node_emb, valid_edges)
            probs = torch.exp(log_probs)
            
            q_values, _ = self.model.get_q_values(state)
            
            # Actor loss: E[α·log π - Q]
            policy_loss = (probs * (self.alpha * log_probs - q_values.detach())).sum()
            actor_loss = actor_loss + policy_loss
            
            # Alpha loss (auto-tuning)
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * 
                              (log_probs.detach() + self.target_entropy).mean())
                alpha_loss_val += alpha_loss.item()
            
            actor_count += 1
        
        if actor_count > 0:
            avg_actor_loss = actor_loss / actor_count
            self.actor_optimizer.zero_grad()
            avg_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                alpha_loss_tensor = torch.tensor(alpha_loss_val / actor_count,
                                                  requires_grad=True, device=self.device)
                alpha_loss_tensor.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        
        # ── Soft target güncelleme ──
        for target_param, param in zip(self.target_model.parameters(),
                                        self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_step += 1
        
        return {
            'critic_loss': avg_critic_loss.item() if valid_count > 0 else 0.0,
            'actor_loss': avg_actor_loss.item() if actor_count > 0 else 0.0,
            'alpha': self.alpha,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. SARSA (State-Action-Reward-State-Action)
# ─────────────────────────────────────────────────────────────────────────────

class SARSABuffer:
    """Replay buffer for SARSA storing (s, a, r, s', a', done) tuples."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class SARSA(BaseAlgorithm):
    """
    SARSA (Rummery and Niranjan, 1994).

    On-policy TD control:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

    Key difference from DQN: uses Q(s', a') where a' is the ACTUAL
    next action taken by the policy, not max_a Q(s', a).

    Adapted for graph-MDP with Q-network and epsilon-greedy.
    """

    def __init__(self, model: BPPActorCritic, lr: float = 1e-3,
                 gamma: float = 1.0, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: int = 5000,
                 buffer_size: int = 50000, batch_size: int = 32,
                 target_update_freq: int = 100, device: str = 'cpu'):
        if not model.use_q_network:
            raise ValueError("SARSA requires model with use_q_network=True.")

        super().__init__(model, lr, gamma, device)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Target network
        self.target_model = deepcopy(model).to(device)
        self.target_model.eval()

        # SARSA replay buffer
        self.replay_buffer = SARSABuffer(buffer_size)

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def get_epsilon(self) -> float:
        """Current epsilon value (decaying)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.training_step / self.epsilon_decay)

    def _select_action(self, state, greedy=False):
        """Epsilon-greedy action selection. Returns (edge_idx, q_values)."""
        epsilon = self.get_epsilon() if not greedy else 0.0

        with torch.no_grad():
            q_values, valid_edges = self.model.get_q_values(state)

        if len(valid_edges) == 0:
            return -1, q_values

        if random.random() < epsilon:
            edge_idx = random.randrange(len(valid_edges))
        else:
            edge_idx = torch.argmax(q_values).item()

        return edge_idx, q_values

    def collect_episode(self, env: BinPackingGraphEnv,
                        greedy: bool = False) -> Dict:
        """Run episode with epsilon-greedy, store (s,a,r,s',a') tuples."""
        state = env.get_state()
        edge_idx, _ = self._select_action(state, greedy)
        total_reward = 0.0

        while not env.done:
            if edge_idx < 0:
                break

            next_state, reward, done, _, info = env.step(edge_idx)
            total_reward += reward

            # Select NEXT action a' (key SARSA difference)
            if done:
                next_edge_idx = -1
            else:
                next_edge_idx, _ = self._select_action(next_state, greedy)

            # Store (s, a, r, s', a', done) in buffer
            self.replay_buffer.push(state, edge_idx, reward,
                                     next_state, next_edge_idx, done)

            state = next_state
            edge_idx = next_edge_idx

        return {
            'total_reward': total_reward,
            'n_bins': env.get_num_bins(),
        }

    def update(self, episodes: List[Dict] = None) -> Dict[str, float]:
        """Sample from buffer and update using SARSA target."""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}

        self.model.train()

        transitions = self.replay_buffer.sample(self.batch_size)

        total_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0

        for trans in transitions:
            state, edge_idx, reward, next_state, next_edge_idx, done = trans

            # Current Q(s, a)
            q_values, _ = self.model.get_q_values(state)
            if edge_idx >= len(q_values):
                continue
            current_q = q_values[edge_idx]

            # SARSA target: r + gamma * Q(s', a')
            with torch.no_grad():
                if done or next_edge_idx < 0:
                    target_q = torch.tensor(reward, dtype=torch.float,
                                            device=self.device)
                else:
                    next_q_values, next_edges = self.target_model.get_q_values(
                        next_state)
                    if next_edge_idx < len(next_q_values):
                        next_q = next_q_values[next_edge_idx]
                    else:
                        next_q = torch.tensor(0.0, device=self.device)
                    target_q = reward + self.gamma * next_q

            loss = F.mse_loss(current_q, target_q)
            total_loss = total_loss + loss
            valid_count += 1

        if valid_count == 0:
            return {'loss': 0.0}

        avg_loss = total_loss / valid_count
        self.optimizer.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_step += 1

        # Target network update
        if self.training_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return {
            'loss': avg_loss.item(),
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ALGORİTMA FABRİKASI
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_REGISTRY = {
    'reinforce': REINFORCE,
    'a2c': A2C,
    'ppo': PPO,
    'dqn': DQN,
    'sac': SAC,
    'sarsa': SARSA,
}


def create_algorithm(name: str, model: BPPActorCritic,
                     device: str = 'cpu', **kwargs) -> BaseAlgorithm:
    """
    İsme göre DPÖ algoritması oluşturur.
    
    Args:
        name: 'reinforce', 'a2c', 'ppo', 'dqn', 'sac'
        model: BPPActorCritic instance
        device: 'cpu' veya 'cuda'
        **kwargs: Algoritmaya özgü parametreler
    """
    name = name.lower()
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Bilinmeyen algoritma: {name}. "
                        f"Desteklenen: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[name](model=model, device=device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# F import
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as F


if __name__ == '__main__':
    from rl_environment import BinPackingGraphEnv
    
    print("=" * 60)
    print("Algoritma Testleri")
    print("=" * 60)
    
    env = BinPackingGraphEnv(n_items=10, capacity=100, reward_type='step')
    
    # Policy gradient algoritmaları (REINFORCE, A2C, PPO)
    for alg_name in ['reinforce', 'a2c', 'ppo']:
        print(f"\n--- {alg_name.upper()} ---")
        model = BPPActorCritic(node_feat_dim=2, embed_dim=32,
                               n_gnn_layers=2, gnn_type='gat')
        
        alg = create_algorithm(alg_name, model)
        
        # Bir episode topla
        state = env.reset(seed=42)
        episode = alg.collect_episode(env)
        print(f"Episode: {len(episode.get('rewards', []))} adım, "
              f"{episode['n_bins']} kutu")
        
        # Güncelle
        losses = alg.update([episode])
        print(f"Loss: {losses}")
    
    # DQN
    print(f"\n--- DQN ---")
    model_dqn = BPPActorCritic(node_feat_dim=2, embed_dim=32,
                                n_gnn_layers=2, gnn_type='gat',
                                use_q_network=True)
    dqn = create_algorithm('dqn', model_dqn)
    state = env.reset(seed=42)
    episode = dqn.collect_episode(env)
    print(f"Episode: {episode['n_bins']} kutu")
    losses = dqn.update()
    print(f"Loss: {losses}")
    
    # SAC
    print(f"\n--- SAC ---")
    model_sac = BPPActorCritic(node_feat_dim=2, embed_dim=32,
                                n_gnn_layers=2, gnn_type='gat',
                                use_q_network=True)
    sac = create_algorithm('sac', model_sac)
    state = env.reset(seed=42)
    episode = sac.collect_episode(env)
    print(f"Episode: {episode['n_bins']} kutu")
    losses = sac.update()
    print(f"Loss: {losses}")
    
    print("\n✓ Tüm algoritma testleri başarılı.")
