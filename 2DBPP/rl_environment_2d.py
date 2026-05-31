"""
rl_environment_2d.py
====================
Graph-based 2D Bin Packing Problem (Single Bin) MDP Environment.

Problem: Pack n rectangular items into a single W x H bin,
         maximizing area utilization.

Approach (analogous to 1D edge-selection):
  - Nodes = items or merged groups, each with bounding box (w, h) and actual area
  - Edge (i,j) = items i,j can be merged (placed side by side)
      Horizontal: width = w_i + w_j, height = max(h_i, h_j), fits in bin
      Vertical:   width = max(w_i, w_j), height = h_i + h_j, fits in bin
  - Action: select an edge -> merge nodes (best direction chosen automatically)
  - Terminal: no more valid edges
  - Reward: step (+1 per merge) or utilization-based

Same interface as 1D BinPackingGraphEnv for model/algorithm reuse.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


def generate_random_2d_instance(n_items: int, bin_w: int, bin_h: int,
                                 low_ratio: float = 0.1, high_ratio: float = 0.5,
                                 seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Generate random rectangular items.

    Each item has width in [low_ratio*W, high_ratio*W]
    and height in [low_ratio*H, high_ratio*H].

    Returns:
        List of (width, height) tuples.
    """
    rng = np.random.RandomState(seed)
    items = []
    for _ in range(n_items):
        w = rng.randint(max(1, int(low_ratio * bin_w)), max(2, int(high_ratio * bin_w)) + 1)
        h = rng.randint(max(1, int(low_ratio * bin_h)), max(2, int(high_ratio * bin_h)) + 1)
        items.append((w, h))
    return items


def bottom_left_decreasing_area(items: List[Tuple[int, int]],
                                 bin_w: int, bin_h: int) -> float:
    """
    Bottom-Left Decreasing Area heuristic baseline.

    Sorts items by area (descending), places each at the bottom-left
    most available position using a skyline approach.

    Returns:
        utilization: fraction of bin area used (0 to 1).
    """
    sorted_items = sorted(items, key=lambda x: x[0] * x[1], reverse=True)
    # Skyline: list of (x_start, x_end, height)
    skyline = [(0, bin_w, 0)]
    placed_area = 0

    for w, h in sorted_items:
        best_pos = None
        best_y = bin_h + 1

        # Try each skyline segment
        for seg_idx, (sx, ex, sy) in enumerate(skyline):
            if w <= ex - sx and sy + h <= bin_h:
                if sy < best_y:
                    best_y = sy
                    best_pos = (sx, sy, seg_idx)

        if best_pos is None:
            continue  # Item doesn't fit

        px, py, seg_idx = best_pos
        placed_area += w * h

        # Update skyline
        new_skyline = []
        for i, (sx, ex, sy) in enumerate(skyline):
            if i == seg_idx:
                # Split segment
                if px > sx:
                    new_skyline.append((sx, px, sy))
                new_skyline.append((px, px + w, py + h))
                if px + w < ex:
                    new_skyline.append((px + w, ex, sy))
            else:
                new_skyline.append((sx, ex, sy))

        # Merge adjacent segments with same height
        merged = [new_skyline[0]]
        for seg in new_skyline[1:]:
            if seg[2] == merged[-1][2] and seg[0] == merged[-1][1]:
                merged[-1] = (merged[-1][0], seg[1], seg[2])
            else:
                merged.append(seg)
        skyline = merged

    return placed_area / (bin_w * bin_h)


def _compute_merge_info(w_i, h_i, w_j, h_j, bin_w, bin_h):
    """
    Compute best merge direction for two items.

    Returns:
        (new_w, new_h, merge_type, waste) or None if no merge possible.
        merge_type: 'h' (horizontal) or 'v' (vertical)
    """
    results = []

    # Horizontal: [i | j]
    hw = w_i + w_j
    hh = max(h_i, h_j)
    if hw <= bin_w and hh <= bin_h:
        waste = hw * hh - (w_i * h_i + w_j * h_j)
        results.append((hw, hh, 'h', waste))

    # Vertical: [i / j]
    vw = max(w_i, w_j)
    vh = h_i + h_j
    if vw <= bin_w and vh <= bin_h:
        waste = vw * vh - (w_i * h_i + w_j * h_j)
        results.append((vw, vh, 'v', waste))

    if not results:
        return None

    # Pick the merge direction with smaller bounding box area (less waste)
    best = min(results, key=lambda x: x[0] * x[1])
    return best


class BinPacking2DEnv:
    """
    Graph-based 2D Bin Packing (Single Bin) MDP Environment.

    Each episode:
    1. Start with n rectangular items
    2. Build compatibility graph (edges = feasible merges)
    3. At each step, select an edge -> merge two items
    4. Continue until no valid merges remain
    5. Utilization = total item area / bin area
    """

    def __init__(self, n_items: int = 20, bin_width: int = 100,
                 bin_height: int = 100, reward_type: str = 'step'):
        self.n_items = n_items
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.reward_type = reward_type

        # Current state
        self.widths = []           # Current width of each node
        self.heights = []          # Current height of each node
        self.areas = []            # Actual area (sum of component items)
        self.node_features = None  # (N, feat_dim) node features
        self.adj = None            # (N, N) adjacency matrix
        self.done = False
        self.n_merges = 0
        self.initial_n_items = 0
        self.total_item_area = 0   # Sum of all original item areas

        # Merge direction for each edge
        self._merge_info = {}  # (i,j) -> (new_w, new_h, type)

        # Merge history
        self.merge_history = []
        self.node_contents = []

    def reset(self, items: Optional[List[Tuple[int, int]]] = None,
              seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Start a new episode.

        Args:
            items: List of (width, height) tuples. None = random.
            seed: Random seed.

        Returns:
            state dict with node_features, adj, valid_edges
        """
        if items is None:
            items = generate_random_2d_instance(
                self.n_items, self.bin_width, self.bin_height, seed=seed
            )

        self.widths = [w for w, h in items]
        self.heights = [h for w, h in items]
        self.areas = [w * h for w, h in items]
        self.total_item_area = sum(self.areas)
        self.initial_n_items = len(items)
        self.done = False
        self.n_merges = 0
        self.merge_history = []
        self.node_contents = [[i] for i in range(len(items))]

        self._rebuild_graph()

        valid_edges = self._get_valid_edges()
        if len(valid_edges) == 0:
            self.done = True

        return self.get_state()

    def step(self, edge_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """
        Select an edge to merge two items/groups.

        Args:
            edge_idx: Index into valid_edges list.

        Returns:
            (state, reward, done, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        valid_edges = self._get_valid_edges()

        if edge_idx < 0 or edge_idx >= len(valid_edges):
            raise ValueError(f"Invalid edge_idx={edge_idx}, "
                           f"valid range: [0, {len(valid_edges)-1}]")

        node_i = valid_edges[edge_idx][0].item()
        node_j = valid_edges[edge_idx][1].item()

        # Get merge info
        info_key = (min(node_i, node_j), max(node_i, node_j))
        merge = self._merge_info[info_key]
        new_w, new_h, merge_type = merge[0], merge[1], merge[2]

        # Record merge
        self.merge_history.append({
            'step': self.n_merges,
            'node_i': node_i,
            'node_j': node_j,
            'size_i': (self.widths[node_i], self.heights[node_i]),
            'size_j': (self.widths[node_j], self.heights[node_j]),
            'merged_size': (new_w, new_h),
            'merge_type': merge_type,
        })

        # Merge nodes
        self._merge_nodes(node_i, node_j, new_w, new_h)
        self.n_merges += 1

        # Rebuild graph
        self._rebuild_graph()

        # Terminal check
        valid_edges_new = self._get_valid_edges()
        if len(valid_edges_new) == 0:
            self.done = True

        # Reward
        reward = self._compute_reward()

        info = {
            'n_nodes': len(self.widths),
            'n_edges': len(valid_edges_new) if not self.done else 0,
            'n_merges': self.n_merges,
            'utilization': self.get_utilization(),
        }

        return self.get_state(), reward, self.done, False, info

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Return current state as tensors."""
        valid_edges = self._get_valid_edges()
        return {
            'node_features': self.node_features.clone(),
            'adj': self.adj.clone(),
            'valid_edges': valid_edges,
            'n_nodes': len(self.widths),
        }

    def get_num_bins(self) -> int:
        """Return current number of groups (analogous to bins in 1D)."""
        return len(self.widths)

    def get_utilization(self) -> float:
        """Return area utilization = total_item_area / bin_area."""
        bin_area = self.bin_width * self.bin_height
        if bin_area == 0:
            return 0.0
        return sum(self.areas) / bin_area

    def _rebuild_graph(self):
        """Build node features, adjacency matrix, and merge info."""
        n = len(self.widths)
        W = self.bin_width
        H = self.bin_height

        # Node features: [w/W, h/H] — same dim as 1D for model reuse
        feat = torch.zeros(n, 2)
        for i in range(n):
            feat[i, 0] = self.widths[i] / W
            feat[i, 1] = self.heights[i] / H
        self.node_features = feat

        # Adjacency matrix and merge info
        adj = torch.zeros(n, n)
        self._merge_info = {}

        for i in range(n):
            for j in range(i + 1, n):
                result = _compute_merge_info(
                    self.widths[i], self.heights[i],
                    self.widths[j], self.heights[j],
                    W, H
                )
                if result is not None:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
                    self._merge_info[(i, j)] = result

        self.adj = adj

    def _get_valid_edges(self) -> torch.Tensor:
        """Extract valid edges from adjacency matrix."""
        if self.adj is None:
            return torch.zeros(0, 2, dtype=torch.long)
        edges = (self.adj > 0).nonzero(as_tuple=False)
        if len(edges) == 0:
            return torch.zeros(0, 2, dtype=torch.long)
        # Keep only upper triangle (i < j)
        mask = edges[:, 0] < edges[:, 1]
        return edges[mask]

    def _merge_nodes(self, node_i: int, node_j: int,
                     new_w: int, new_h: int):
        """Merge two nodes into one with the given bounding box."""
        if node_i > node_j:
            node_i, node_j = node_j, node_i

        # New area = sum of component areas
        new_area = self.areas[node_i] + self.areas[node_j]
        new_contents = self.node_contents[node_i] + self.node_contents[node_j]

        # Remove old nodes (larger index first)
        del self.widths[node_j]
        del self.heights[node_j]
        del self.areas[node_j]
        del self.node_contents[node_j]
        del self.widths[node_i]
        del self.heights[node_i]
        del self.areas[node_i]
        del self.node_contents[node_i]

        # Add merged node
        self.widths.append(new_w)
        self.heights.append(new_h)
        self.areas.append(new_area)
        self.node_contents.append(new_contents)

    def _compute_reward(self) -> float:
        """
        Compute reward.

        step:     +1 for each merge, 0 at terminal
        terminal: 0 during merges, utilization at terminal
        """
        if self.reward_type == 'step':
            return 0.0 if self.done else 1.0
        elif self.reward_type == 'terminal':
            if self.done:
                return self.get_utilization()
            return 0.0
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def render(self):
        """Print current state for debugging."""
        print(f"\n{'='*50}")
        print(f"Step {self.n_merges} | Nodes: {len(self.widths)} | "
              f"Done: {self.done} | Util: {self.get_utilization():.1%}")
        print(f"Bin: {self.bin_width} x {self.bin_height}")
        for i in range(len(self.widths)):
            print(f"  Node {i}: {self.widths[i]}x{self.heights[i]} "
                  f"(area={self.areas[i]}, items={self.node_contents[i]})")
        valid = self._get_valid_edges()
        print(f"Valid edges: {len(valid)}")
        for idx, e in enumerate(valid):
            i, j = e[0].item(), e[1].item()
            key = (min(i, j), max(i, j))
            m = self._merge_info[key]
            print(f"  [{idx}] ({i},{j}): {self.widths[i]}x{self.heights[i]} + "
                  f"{self.widths[j]}x{self.heights[j]} -> "
                  f"{m[0]}x{m[1]} ({m[2]})")
        print(f"{'='*50}")


# ---- TEST ----
if __name__ == '__main__':
    print("=" * 60)
    print("2D Bin Packing Environment Test")
    print("=" * 60)

    # Small test: 5 items in 10x10 bin
    items = [(3, 4), (5, 2), (2, 3), (4, 5), (3, 3)]
    env = BinPacking2DEnv(n_items=5, bin_width=10, bin_height=10, reward_type='step')
    state = env.reset(items=items)
    env.render()

    total_reward = 0.0
    while not env.done:
        state, reward, done, _, info = env.step(0)
        total_reward += reward
        env.render()

    print(f"\nEpisode finished!")
    print(f"Total merges: {env.n_merges}")
    print(f"Total reward: {total_reward}")
    print(f"Final groups: {env.get_num_bins()}")
    print(f"Utilization: {env.get_utilization():.1%}")

    # Random instance test
    print(f"\n{'='*60}")
    print("Random Instance Test (20 items, 100x100 bin)")
    env2 = BinPacking2DEnv(n_items=20, bin_width=100, bin_height=100)
    state = env2.reset(seed=42)
    print(f"Items: {env2.initial_n_items}, Edges: {len(state['valid_edges'])}")

    step = 0
    while not env2.done:
        state, reward, done, _, info = env2.step(0)
        step += 1

    print(f"Steps: {step}, Groups: {env2.get_num_bins()}, "
          f"Utilization: {env2.get_utilization():.1%}")

    # Baseline comparison
    items_for_baseline = generate_random_2d_instance(20, 100, 100, seed=42)
    bl_util = bottom_left_decreasing_area(items_for_baseline, 100, 100)
    print(f"BLDA baseline utilization: {bl_util:.1%}")
