import gym
from matplotlib import colors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym.utils import EzPickle
import random
import time
from Dataset import Dataset

mcolors = list(colors.TABLEAU_COLORS)

class BinPack(gym.Env, EzPickle):
    def __init__(self,capacity, items):
        EzPickle.__init__(self)
        self.capacity = capacity
        self.items = items
        self.bins = items
        self.num_nodes = len(items)
        self.lbl = {index : item for index, item in enumerate(items)}
        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])\
                            for i in range(self.num_nodes)]
        self.n_colors = '#AAAAAA'
        
        self.edges = []
        self.actions = []
        for i in range(self.num_nodes-1):
            for j in range(i+1,self.num_nodes):
                if(items[i]+ items[j] <= capacity):
                    self.edges+= [(i,j)]      
        self.reset()


    def done(self):
        return self.G.number_of_edges()==0
    
    def actionSpaceSample(self):
        edge = random.choice(list(self.G.edges))
        return edge

    def override(fn):
        """
        override decorator
        """
        return fn
    def updateGraph(self, action):
        (u,v) = action
        
        #weigth = self.bins[u] + self.bins[v]
        self.bins[u] = self.bins[u] + self.bins[v]
        self.bins[v] = 0
        ebunch =[]
        for n in self.G.neighbors(v):
            self.G.add_edge(u,n)
            ebunch.append((n,v))
        self.G.remove_edges_from(ebunch)
        
        ebunch =[]
        for n in self.G.neighbors(u):
            if(self.bins[u] + self.bins[n]>self.capacity):
                 ebunch.append((n,u))
        
        self.G.remove_edges_from(ebunch)

        return

    @override
    def step(self, action):
        if(self.G.has_edge(*action)):
            self.G.remove_edge(*action)
            self.solG.add_edge(*action)
            self.updateGraph(action) 
        
        reward = 1 if not self.done() else 1      
        state = [self.bins, self.G]
        return state, reward, self.done(), None

    @override
    def reset(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from(self.edges)
        self.bins = self.items

        self.solG = nx.Graph()
        self.solG.add_nodes_from(range(self.num_nodes))
        self.g_pos= nx.spring_layout(self.solG)

        state = [self.bins, self.G]
        return state

    def render(self):
        plt.clf()

        plt.subplot(121)
        nx.draw_networkx(self.solG, self.g_pos,with_labels=False,node_color=self.n_colors)
        nx.draw_networkx_labels(self.solG, self.g_pos,self.lbl)
        i=0
        for g in nx.connected_components(self.solG):
            nx.draw_networkx_edges(self.solG,self.g_pos,edgelist=self.solG.subgraph(g).edges(),edge_color=mcolors[i%len(mcolors)])
            i+=1
        plt.subplot(122)
        nx.draw_networkx(self.G, self.g_pos,with_labels=False,node_color=self.n_colors)
        nx.draw_networkx_labels(self.G, self.g_pos,self.lbl)

 
        plt.show(block=False)   
        plt.pause(.1)  
        return

if __name__ == "__main__":
    items = [10,5,2,13,4,1,6,9]
    capacity = 17
    env = BinPack(capacity,items)
    done = False
    epRewards = 0
    plt.close('all')
    while not done:
        env.render()
        action = env.actionSpaceSample()
        observation_, reward, done, info = env.step(action)
        epRewards += reward
        time.sleep(.5) 
     
    print("# of bins :",env.solG.number_of_nodes()- env.solG.number_of_edges())
    print("Rewards:",epRewards)
    #env.render()
    #plt.show() 


