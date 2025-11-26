from math import ceil, floor
from Memory import Memory
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

    memories = []
    modelPATH = './model/policy.pt'
    dataPATH = './data/Randomly_Generated'
    ppo = PPO(
            eps_clip=0.2,
            gamma=0.9,
            k_epochs=6,
            batch_size=1,
            lr=2e-5,
            decay_step_size=2000,
            decay_ratio=0.9,
            policy_loss_coeff=1,
            value_loss_coeff=0.5,
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
    dataset = Dataset(dataPATH)
   
    episodes = []

    for i in range(len(dataset)):
        for j in range(100):
            episode = Memory()
            capacity,items = dataset[i]
            if(capacity>50):
                continue
            #items = [10,5,11,13,12,12,6,12,13,2,3,2,8]
            #capacity = 18
        
            env = BinPack(capacity,items)
            observation_ = env.reset()
            done = False
            epRewards = 0
            with torch.no_grad():
                while not done:
                    action,_,_ = ppo.select_action(observation_,episode)
                    #action = env.actionSpaceSample()
                    #    print(epRewards,":",action, ":",env.G.number_of_edges())
                    observation_, reward, done, info = env.step(action)
                    episode.rewards.append(reward)
                    epRewards += reward
            nbin = env.nBins
            #memories.append(episode)
            #print("# of bins:",env.solG.number_of_nodes()- env.solG.number_of_edges())
            loss,vloss,val = ppo.update([episode])
            if(j%10==0):
                print('LB: {} - Bins: {} Loss: {:.8f} Val: {:.8f}'.format(ceil(sum(items)/capacity),nbin,vloss,val))
                torch.save(ppo.policy.state_dict(),modelPATH)