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
        device="cpu",
    ):
        super(ActorCritic, self).__init__()
        self.n_mlp_layers_feature_extractor=n_mlp_layers_feature_extractor
        self.n_layers_feature_extractor=n_layers_feature_extractor
        self.input_dim_feature_extractor=input_dim_feature_extractor
        self.hidden_dim_feature_extractor=hidden_dim_feature_extractor
        self.n_mlp_layers_actor=n_mlp_layers_actor
        self.hidden_dim_actor=hidden_dim_actor
        self.n_mlp_layers_critic=n_mlp_layers_critic
        self.hidden_dim_critic=hidden_dim_critic
        self.device=device

        self.feature_extractors = []
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

    def forward(self, state):
        """
        state has at least 2 attributes :
        - x : the feature tensor (n_nodes*input_dim_feature_extractor) shaped
        - edge_index : the edge tensor (2, n_edges) shaped
        """

        # Feature extraction
        features = state.x
        n_nodes = features.size(0)
        #dummy batch for global_mean_pool
        batch = torch.zeros(n_nodes,1,dtype=torch.int64)
        edge_index = torch.cat([state.edge_index , state.edge_index[[1,0]]],dim=1)
        for layer in range(self.n_layers_feature_extractor):
            features = self.feature_extractors[layer](features, edge_index)
       
        graph_embedding = global_mean_pool(features,batch)
        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_s_a_pairs(
            graph_embedding,features, state.edge_index
        )
        probabilities = self.actor(possible_s_a_pairs)
        #probabilities = probabilities.view(1,-1)
        pi = F.softmax(probabilities, dim=0)
        return pi, value

    def compute_s_a_pairs(self, graph_embedding, features,edge_index):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        states = graph_embedding.repeat(edge_index.size(-1),1)
        nodes1 = features[edge_index[0]]
        nodes2 = features[edge_index[1]]
        s_a_pairs = torch.cat([states, nodes1, nodes2], dim=1)
  
        return s_a_pairs


# This is used for unit testing
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(
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

    dataloader = DataLoader([graph, graph2], batch_size=1)
    for batch in dataloader:
        print(actor_critic(batch))
