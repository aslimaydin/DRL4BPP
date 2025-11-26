class Memory:
    def __init__(self):
        self.graphs = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.log_probability = []
    def __len__(self):
        return len(self.graphs)

    def compute_returns(self, gamma):
        n = len(self.rewards)
        returns = [0]*n
        discounted_return = 0
        for i in reversed(range(n)):
            if self.done[i]:
                discounted_return = 0
            discounted_return = self.rewards[i] + gamma * discounted_return
            returns[i] = discounted_return
        return returns


# This is used for unit testing
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    memory = Memory()
    for i in range(10):
        memory.graphs.append(
            Data(
                x=torch.rand(2, 4),
                edge_index=torch.tensor(
                    [[0, 0, 1, 1, 1, 1, 2, 2, 2, 3], [0, 3, 0, 1, 2, 3, 0, 2, 3, 3]],
                    dtype=torch.long,
                ),
            )
        )
        memory.action.append((0, 1))
        memory.rewards.append(-1)
        memory.done.append(False if i < 9 else True)
        memory.log_probability.append(torch.rand(1, 1))

    returns = memory.compute_returns(0.91)
    print(list(returns))
