import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool
from mlp import MLP


class ActorCriticBatch(nn.Module):
    def __init__(
        self,
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=3,
        n_mlp_layers_actor=3,
        hidden_dim_actor=64,
        n_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device="cpu",
    ):
        super(ActorCriticBatch, self).__init__()
        self.feature_extractors = []
        self.n_layers_feature_extractor = n_layers_feature_extractor
        self.input_dim_feature_extractor=input_dim_feature_extractor
        for layer in range(n_layers_feature_extractor):
            self.feature_extractors.append(
                GINConv(
                    MLP(
                        n_layers=n_mlp_layers_feature_extractor,
                        input_dim=input_dim_feature_extractor,
                        hidden_dim=hidden_dim_feature_extractor,
                        output_dim=input_dim_feature_extractor,
                        batch_norm=True,
                        device=device,
                    )
                )
            )

        self.actor = MLP(
            n_layers=n_mlp_layers_actor,
            input_dim=input_dim_feature_extractor * 3,
            hidden_dim=hidden_dim_actor,
            output_dim=1,
            batch_norm=False,
            device=device,
        )
        self.critic = MLP(
            n_layers=n_mlp_layers_critic,
            input_dim=input_dim_feature_extractor,
            hidden_dim=hidden_dim_critic,
            output_dim=1,
            batch_norm=False,
            device=device,
        )

    def forward(self, batch):
        """
        The batch input is an object of type torch_geometric.data.batch.Batch
        It represents a batch of graphs as a giant graph, to speed up computation
        (see https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        for details)
        It has at least 3 attributes :
        - x : the feature tensor (n_nodes*batch_size,input_dim_feature_extractor) shaped
        - edge_index : the edge tensor (2, n_edges_giant_graph) shaped
        - batch : the id of the node in the batch (n_nodes*batch_size,) shaped
        """
        batch_size = batch.batch[-1].item() + 1 
        n_nodes = self._compute_n_nodes(batch.batch[batch.edge_index[0]])
        # Feature extraction
        features = batch.x
        edge_index = torch.cat([batch.edge_index , batch.edge_index[[1,0]]],dim=1)
        for layer in range(self.n_layers_feature_extractor):
            features = self.feature_extractors[layer](features, edge_index)
       
        graph_embedding = global_mean_pool(features,batch.batch)
        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_s_a_pairs(
            graph_embedding,features, batch.edge_index,batch.batch
        )
        probabilities = self.actor(possible_s_a_pairs)
        probabilities = torch.split(probabilities,n_nodes)
        pi =[]
        for p in probabilities:
            pi.append(F.softmax(p, dim=0))
        return pi, value

    def compute_s_a_pairs(self, graph_embedding, features,edge_index,indexes):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        states = graph_embedding[indexes[edge_index[0]]]
        nodes1 = features[edge_index[0]]
        nodes2 = features[edge_index[1]]
        s_a_pairs = torch.cat([states, nodes1, nodes2], dim=1)
  
        return s_a_pairs

    def _compute_n_nodes(self,batch):
        cnt =1
        n_nodes=[]
        for i in range(1,batch.size(-1)):
            if(batch[i-1]!=batch[i]):
                n_nodes.append(cnt)
                cnt=1
            else:
                cnt = cnt+1
        n_nodes.append(cnt)
        return n_nodes

# This is used for unit testing
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCriticBatch(
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=3,
        n_mlp_layers_actor=3,
        hidden_dim_actor=64,
        n_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device=device,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2], 
         [1, 2, 2, 3, 3]],
        dtype=torch.long,
    )
    edge_index2 = torch.tensor(
        [[0, 0, 1, 1,2], 
         [1, 2, 2, 3,3]],
        dtype=torch.long,
    )
    features = torch.rand(4, 2)
    features2 = torch.rand(5, 2)
    graph = Data(x=features, edge_index=edge_index)
    graph2 = Data(x=features2, edge_index=edge_index2)

    dataloader = DataLoader([graph, graph2], batch_size=2)
    for batch in dataloader:
        print(actor_critic(batch))
