from copy import deepcopy
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx
from actor_critic import ActorCritic


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
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

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
        '''state has [x,G]'''
        x,G = state
        edges = list(G.edges())
        degrees = [G.degree(n) for n in G.nodes()]
        dG = max(degrees)
        x = torch.tensor([x,degrees],dtype=torch.float,device=self.device).permute([1,0])
        edge_index = torch.tensor(edges,dtype=torch.long,device=self.device).permute([1,0])
        #tensor state
        state = Data(x=x,edge_index=edge_index)

        
        return state

    def select_action(self, state):
        '''
        state is list [x,G]
        '''
        x,G = state
        edges = list(G.edges())
        
        if len(edges)==1:
            return edges[0],1
        degrees = [G.degree(n) for n in G.nodes()]
        dG = max(degrees)
        x = torch.tensor([x,degrees],dtype=torch.float,device=self.device).permute([1,0])
        edge_index = torch.tensor(edges,dtype=torch.long,device=self.device).permute([1,0])
        #tensor state
        state = Data(x=x,edge_index=edge_index)
        pi,v = self.policy(state)
        dist = Categorical(pi.squeeze())
        action_index = dist.sample()
        action = edges[action_index]

        return action,action_index[0].item()

    def eval_action(self, pi, actions):
        pass

    def update(self, memories):

        for epoch in range(self.k_epochs):
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0
            for i in range(len(memories)):
                # We use a dataloader structure to compute the pi and values of the
                # actor_critic agent, to be able to use batches to speed up computation
                dataloader = DataLoader(memories[i].graphs, batch_size=self.batch_size)
                current_pi = []
                current_values = []
                for batch in dataloader:
                    batch_pi, batch_values = self.policy(batch=batch)
                    current_pi.append(batch_pi)
                    current_values.append(batch_values)

                current_pi = torch.cat(current_pi)
                current_values = torch.cat(current_values)

                current_log_probabilities, entropy = self.eval_action(
                    current_pi, memories[i].action
                )

                # Here ratio = pi(a|s)/pi_old(a|s). We compute it with exponential
                # since what we have are the log of the probabilities
                ratios = torch.exp(
                    current_log_probabilities - memories[i].log_probability
                )

                # The advantage function is a MC estimate of the state-value function
                # i.e. At = Gt - V(st, w)
                MC_values = memories[i].compute_returns(self.gamma)
                advantages = MC_values - current_values

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
                current_entropy_loss = -entropy.clone()

                policy_loss += current_policy_loss
                value_loss += current_value_loss
                entropy_loss += current_entropy_loss

            loss = (
                self.policy_loss_coeff * policy_loss
                + self.value_loss_coeff * value_loss
                + self.entropy_loss_coeff * entropy_loss
            )

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Old policy is updated with the new policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # And we can decrease learning rate according to lr_scheduler
        self.scheduler.step()

        return loss.mean().item()


if __name__ == "__main__":

    from Memory import Memory

    memory = Memory()
    for i in range(10):
        memory.graphs.append(
            Data(
                x=torch.rand(4, 2),
                edge_index=torch.tensor(
                    [[0, 0, 1, 1, 1, 1, 2, 2, 2, 3], [0, 3, 0, 1, 2, 3, 0, 2, 3, 3]],
                    dtype=torch.long,
                ),
            )
        )
        memory.action.append((0, 1))
        memory.reward.append(-1)
        memory.done.append(False if i < 9 else True)
        memory.log_probability.append(torch.rand(1, 1))

    ppo = PPO(
        eps_clip=0.2,
        gamma=1,
        k_epochs=3,
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
