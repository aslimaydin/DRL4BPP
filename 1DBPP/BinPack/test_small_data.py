from math import ceil
import torch
from matplotlib import colors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym.utils import EzPickle
import random
import time
from Dataset import Dataset
from BinPack_Env import BinPack
from ppo import PPO



if __name__ == "__main__":
    modelPATH = './model/policy.pt'
    items = [13,10,10,6,5,4,2,1]
    #items = [10,5,11,13,12,12,6,12,13,2,3,2,8]
    #items = [10,5,2,14,3]
    capacity = 18

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
        hidden_dim_actor=16,
        n_mlp_layers_critic=3,
        hidden_dim_critic=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    #ppo.policy.load_state_dict(torch.load(modelPATH))
    #ppo.policy.eval()
    for i in range(1):
        env = BinPack(capacity,items.copy())
        observation_ = env.reset()
        done = False
        epRewards = 0
        torch.no_grad()
        plt.close('all')
    
        while not done:
            env.render()
            #action,_ = ppo.greedy_select_action(observation_)
            action = env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            #print('Rew:',reward, 'EpRew:',epRewards)
            time.sleep(2) 
            #input()

        #print('LB:',ceil(sum(items)/capacity)," # of bins:",env.solG.number_of_nodes()- env.solG.number_of_edges(),"Ep:", epRewards)
        print(env.solG.number_of_nodes()- env.solG.number_of_edges(),",", epRewards)
    env.render()
    plt.show() 