import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool
from mlp import MLP


class ActorCritic(nn.Module):
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
        device="cuda:0",
    ):
        super(ActorCritic, self).__init__()
        self.device=device

        self.feature_extractors = []

        self.feature_extractors.append(
                GINConv(
                    MLP(
                        n_layers=n_mlp_layers_feature_extractor,
                        input_dim=input_dim_feature_extractor,
                        hidden_dim=hidden_dim_feature_extractor,
                        output_dim=hidden_dim_feature_extractor,
                        batch_norm=True,
                        device=device,
                    )
                    ,eps=torch.tensor(0.1) 
                )
            )
        self.feature_extractors[0].to(device=device)

        for layer in range(1,n_layers_feature_extractor):
            self.feature_extractors.append(
                GINConv(
                    MLP(
                        n_layers=n_mlp_layers_feature_extractor,
                        input_dim=hidden_dim_feature_extractor,
                        hidden_dim=hidden_dim_feature_extractor,
                        output_dim=hidden_dim_feature_extractor,
                        batch_norm=True,
                        device=device,
                    )
                    ,eps=torch.tensor(0.1) 
                )
            )
            self.feature_extractors[layer].to(device=device)
        self.actor = MLP(
            n_layers=n_mlp_layers_actor,
            input_dim=hidden_dim_feature_extractor * 3,
            hidden_dim=hidden_dim_actor,
            output_dim=1,
            batch_norm=False,
            device=device,
        )
        self.critic = MLP(
            n_layers=n_mlp_layers_critic,
            input_dim=hidden_dim_feature_extractor,
            hidden_dim=hidden_dim_critic,
            output_dim=1,
            batch_norm=False,
            device=device,
        )

    def forward(self, state):
        """
        state has at least 2 attributes :
        - x : the feature tensor (n_nodes*input_dim_feature_extractor) shaped
        - edge_index : the edge tensor (2, n_edges) shaped
        """

        # Feature extraction
        features = state.x
        n_nodes = features.size(0)
        batch = state.batch
        #dummy batch for global_mean_pool
        #batch = torch.zeros(n_nodes,1,dtype=torch.int64,device=self.device)
        edge_index = torch.cat([state.edge_index , state.edge_index[[1,0]]],dim=1)
        for layer in range(len(self.feature_extractors)):
            features = self.feature_extractors[layer](features, edge_index)
       
        graph_embedding = global_mean_pool(features,batch)[0]
        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_s_a_pairs(
            graph_embedding,features, state.edge_index
        )
        probabilities = self.actor(possible_s_a_pairs)
        probabilities = probabilities.squeeze()
        pi = F.softmax(probabilities,dim=0)
        return pi, value

    def compute_s_a_pairs(self, graph_embedding, features,edge_index):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        states = graph_embedding.repeat(edge_index.size(-1),1)
        nodes1 = features[edge_index[0]]
        nodes2 = features[edge_index[1]]
        s_a_pairs = torch.cat([states, nodes1,nodes2], dim=1)
  
        return s_a_pairs


# This is used for unit testing
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=8,
        n_mlp_layers_actor=3,
        hidden_dim_actor=64,
        n_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device=device,
    )
    batch = torch.tensor([0,1,0,1],dtype=torch.long,device=device)
    batch2 = torch.tensor([0,0,0,0,0],dtype=torch.long,device=device)
    edge_index = torch.tensor(
        [[0, 0, 1 ], 
         [1, 2, 2]],
        dtype=torch.long,
        device=device
    )
    edge_index2 = torch.tensor(
        [[0, 0, 1, 1,2], 
         [1, 2, 2, 3,3]],
        dtype=torch.long,
        device=device
    )
    features = torch.tensor([[1,2],[1,3],[.5,1],[0,0]],device=device)
    features2 = torch.rand(5, 2).to(device=device)
    graph = Data(x=features, edge_index=edge_index,batch=batch).to(device=device)
    graph2 = Data(x=features2, edge_index=edge_index2,batch=batch2).to(device=device)

    #dataloader = DataLoader([graph, graph2], batch_size=1)
    #for batch in dataloader:
    print(actor_critic(graph))
    print(actor_critic(graph2))
