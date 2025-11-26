from copy import deepcopy
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
from actor_critic import ActorCritic
import random

class PPO:
    def __init__(
        self,
        eps_clip,
        gamma,
        k_epochs,
        batch_size,
        lr,
        decay_step_size,
        decay_ratio,
        policy_loss_coeff,
        value_loss_coeff,
        entropy_loss_coeff,
        n_mlp_layers_feature_extractor,
        n_layers_feature_extractor,
        input_dim_feature_extractor,
        hidden_dim_feature_extractor,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
        device="cpu",
    ):

        self.policy = ActorCritic(
            n_mlp_layers_feature_extractor,
            n_layers_feature_extractor,
            input_dim_feature_extractor,
            hidden_dim_feature_extractor,
            n_mlp_layers_actor,
            hidden_dim_actor,
            n_mlp_layers_critic,
            hidden_dim_critic,
            device,
        )
        #self.policy_old = deepcopy(self.policy)
        #self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_step_size, gamma=decay_ratio
        )
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        # The value part of the Loss is a classic MSE Loss
        self.value_loss_function = nn.MSELoss()

        self.value_loss_coeff = value_loss_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.device=device
    
    def state2tensor(self,state):
        '''state = [bins,G,capacity]'''
        bins,G,capacity = state
        
        edges = np.array(G.edges()).T
        degrees = np.array([G.degree(n) for n in G.nodes()])
        batch = (degrees==0).astype(int)
        degrees = degrees / degrees.max()
        bins = np.array(bins) / capacity
        features = np.stack((bins,degrees),axis=1)

        x = torch.tensor(features,dtype=torch.float,device=self.device)
        edge_index = torch.tensor(edges,dtype=torch.long,device=self.device)
        batch = torch.tensor(batch,dtype=torch.long,device=self.device)
        #tensor state
        state = Data(x=x,edge_index=edge_index,batch=batch).to(self.device)

        
        return state

    def select_action(self, state,memory=None):
        _,G,_ = state
        edges = list(G.edges())
        state = self.state2tensor(state)
        log_prob = 0.0
        action_index = 0
        done = True

        if len(edges)!=1:
            pi,v = self.policy(state)
            dist = Categorical(pi.squeeze())
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
            action_index = int(action_index.item())
            done = False

        if(memory!=None):
            memory.graphs.append(state)
            memory.actions.append(action_index)
            memory.done.append(done)
            memory.log_probability.append(log_prob)

        return  edges[action_index], action_index, log_prob

    def eval_action(self, pi, actions):
        softmax_dist = Categorical(pi)
        log_prob = softmax_dist.log_prob(actions)
        entropy = softmax_dist.entropy()
        return log_prob, entropy
    
    def greedy_select_action(self, state):
        _,G,_ = state
        edges = list(G.edges())
        action_index = 0
        if len(edges)>1:
            state = self.state2tensor(state)
            pi,v = self.policy(state)
            _,action_index = pi.max(0)
            action_index=action_index.item()
        return edges[action_index], action_index

    def update(self, memories):

        for epoch in range(self.k_epochs):
            vLoss=0
            for episode  in memories:
                current_log_probabilities = []
                current_values = []
                entropies = []
                for i in range(len(episode.graphs)):    
                    state = episode.graphs[i].to(self.device)
                    current_pi, current_value = self.policy(state=state)
                    if len(current_pi.size())==0:
                        current_pi=current_pi.unsqueeze(0)

                    current_log_probabilitiy, entropy = self.eval_action(
                        current_pi,torch.tensor(episode.actions[i],device=self.device)
                        )
                    current_log_probabilities.append(current_log_probabilitiy)
                    current_values.append(current_value)
                    entropies.append(entropy)
                
                current_log_probabilities = torch.stack(current_log_probabilities).to(self.device)
                current_values = torch.cat(current_values).to(self.device)
                entropies = torch.stack(entropies).to(self.device)
                # Here ratio = pi(a|s)/pi_old(a|s). We compute it with exponential
                # since what we have are the log of the probabilities
                ratios = torch.exp(
                    current_log_probabilities - torch.tensor(episode.log_probability,device=self.device).detach()
                )

                # The advantage function is a MC estimate of the state-value function
                # i.e. At = Gt - V(st, w)
                MC_values = torch.tensor(episode.compute_returns(self.gamma),dtype=torch.float,device=self.device)
                #MC_values = (MC_values - MC_values.mean()) / (MC_values.std() + 1e-5)
                advantages = MC_values - current_values.detach()
                #advantages = (advantages - advantages.mean()) - (advantages.std() + 1e-5)
                # We compute unclipped and clipped objectives, as specified in PPO
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)*advantages

                # And then we compute the 3 losses : policy, value, and entropy.
                # Since policy objectives and entropy are to be maximized, we add a
                # minus sign before, since loss is going to be minimized.
                policy_loss = -torch.min(surr1, surr2)
                value_loss = self.value_loss_function(current_values, MC_values)
                entropy_loss = -entropies.mean()
                vLoss+=value_loss.mean()
                loss = (
                    self.policy_loss_coeff * policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_loss_coeff * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        #print("Loss:", loss.mean().item())
        # Old policy is updated with the new policy
        #self.policy_old.load_state_dict(self.policy.state_dict())
        # And we can decrease learning rate according to lr_scheduler
        self.scheduler.step()

        return loss.mean().item(),vLoss.mean().item(),current_values.mean().item()

    def update_old(self, memories):

        for epoch in range(self.k_epochs):
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0
            for batch  in memories:
                # We use a dataloader structure to compute the pi and values of the
                # actor_critic agent, to be able to use batches to speed up computation
                #dataloader = DataLoader(memories[i].graphs, batch_size=self.batch_size)
                current_log_probabilities = []
                current_values = []
                current_pis = []
                entropies = []
                for i in range(len(batch.graphs)):
                    state = batch.graphs[i].to(self.device)
                    current_pi, current_value = self.policy(state=state)
                    current_values.append(current_value)
                    current_pis.append(current_pi)
                
                actions = torch.tensor(batch.actions, device=self.device)
                current_pis = torch.stack(current_pis).to(self.device)
                current_values = torch.cat(current_values).to(self.device)

                current_log_probabilities, entropies = self.eval_action(current_pis,actions)

                # Here ratio = pi(a|s)/pi_old(a|s). We compute it with exponential
                # since what we have are the log of the probabilities
                ratios = torch.exp(
                    current_log_probabilities - torch.tensor(batch.log_probability)
                )

                # The advantage function is a MC estimate of the state-value function
                # i.e. At = Gt - V(st, w)
                MC_values = torch.tensor(batch.compute_returns(self.gamma),dtype=torch.float)
                MC_values = (MC_values - MC_values.mean()) / (MC_values.std() + 1e-5)
                advantages = MC_values - current_values.detach()

                # We compute unclipped and clipped objectives, as specified in PPO
                unclipped_objective = ratios * advantages
                clipped_objective = torch.clamp(
                    ratios, 1 - self.eps_clip, 1 + self.eps_clip
                )*advantages

                # And then we compute the 3 losses : policy, value, and entropy.
                # Since policy objectives and entropy are to be maximized, we add a
                # minus sign before, since loss is going to be minimized.
                current_policy_loss = -torch.min(unclipped_objective, clipped_objective)
                current_value_loss = self.value_loss_function(current_values, MC_values)
                current_entropy_loss = -entropies.clone()

                policy_loss += current_policy_loss
                value_loss += current_value_loss
                entropy_loss += current_entropy_loss.clone()

            loss = (
                self.policy_loss_coeff * policy_loss
                + self.value_loss_coeff * value_loss
                + self.entropy_loss_coeff * entropy_loss
            )

            # Optimization step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            print("Loss:", loss.mean().item())
        # Old policy is updated with the new policy
        #self.policy_old.load_state_dict(self.policy.state_dict())
        # And we can decrease learning rate according to lr_scheduler
        self.scheduler.step()

        return loss.mean().item()


if __name__ == "__main__":

    from Memory import Memory

    memory = Memory()



    for i in range(9):
        memory.graphs.append(
            Data(
                x=torch.rand(4, 2),
                edge_index=torch.tensor(
                    [[0, 0, 1, 1, 2], [1, 3, 2, 3, 3]],
                    dtype=torch.long,
                ),
            )
        )
        memory.actions.append(random.randint(0,4))
        memory.reward.append(1)
        memory.done.append(False if i < 8 else True)
        memory.log_probability.append(random.random())

    memory.graphs.append(
        Data(
            x=torch.rand(4, 2),
            edge_index=torch.tensor(
                [[0], [1]],
                dtype=torch.long,
            ),
        )
    )
    memory.actions.append(0)
    memory.reward.append(1)
    memory.done.append(True)
    memory.log_probability.append(0)

    ppo = PPO(
        eps_clip=0.2,
        gamma=0.99,
        k_epochs=100,
        batch_size=1,
        lr=2e-5,
        decay_step_size=2000,
        decay_ratio=0.9,
        policy_loss_coeff=2,
        value_loss_coeff=1,
        entropy_loss_coeff=0.01,
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=8,
        n_mlp_layers_actor=3,
        hidden_dim_actor=32,
        n_mlp_layers_critic=3,
        hidden_dim_critic=32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ppo.update([memory])
