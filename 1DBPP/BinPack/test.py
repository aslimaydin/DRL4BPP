import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import numpy as np

G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edges_from([(0,1),(0,2),(1,3),(2,4),(3,4),(3,4)])
nx.draw_networkx(G)
plt.show()