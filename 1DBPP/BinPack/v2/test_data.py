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
    dataPATH = './data/Falkenauer/Falkenauer_T'
    dataset = Dataset(dataPATH)
    capacity,items = dataset[0]
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
    ppo.policy.load_state_dict(torch.load(modelPATH))
    #ppo.policy.eval()
    env = BinPack(capacity,items)
    observation_ = env.reset()
    done = False
    epRewards = 0
    torch.no_grad()
    while not done:
        action,_ = ppo.greedy_select_action(observation_)
        #action = env.actionSpaceSample()
        if epRewards%100==0:
            print(epRewards,":",action, ":",env.G.number_of_edges())
        observation_, reward, done, info = env.step(action)
        epRewards += reward
     
    print("# of bins:",env.solG.number_of_nodes()- env.solG.number_of_edges())
