This repository accompanies our work on project "Solution Approach to One and Two-Dimensional Bin Packing Problem With Deep Reinforcement Learning" 
supported by TUBITAK-1001 under contract no 124M974

## Abstract
The bin packing problem (BPP) is one of the well-known combinatorial optimization problem. Briefly, it can be defined as the problem of placing objects of known dimensions into boxes of known capacity in such a way as to minimize the number of boxes used. This problem has attracted great interest from researchers, as it finds frequent use in a wide range of industrial applications such as logistics, manufacturing, scheduling, and cloud computing. There are numerous variations in the literature based on factors such as the size and shape of the packed objects and whether they are known before packing. In this project, the main focus is on the One and Two-Dimensional Bin Packing Problem (1D-BPP and 2D-BPP). 
1B-CPP applications are mostly found in areas such as resource management and cloud storage. On the other hand, 2D-BPP is widely applied in the manufacturing industry where materials such as steel, glass, paper, leather, fabric, etc. are used as raw materials. The problem is placing the pieces to be cut from the raw material rolls in a way to minimize the residual material. 2D-BPP also has applications in areas such as layout design and logistics.

Combinatorial optimization problems mostly belong to the NP-Hard problem class. For this reason, instead of exact solution methods, approximation algorithms, heuristics and metaheuristic are more widely used in as solution approaches. Current work shows there is still room improvement for the solutions. Machine learning methods are a promising approach for solving combinatorial optimization problems. However, the difficulty of obtaining exact solutions prevents the use of them as class labels in supervised learning. Hence, Deep Reinforcement Learning (DRL), which does not rely on class labels, is a viable machine learning method for addressing these issues.

Using DRL as a solution approach usually requires Markov Decision Process (MDP)formulation of the problem. For combinatorial optimization problems such as the Travelling Salesman Problem the MDP formulation is straightforward since this problem by definition have a graphical structure. Since KPP problems lack this feature, the graphical representation of the problem must be defined first. A MDP model will be constructed utilizing this representation, whereby the graph elements incorporated in the MDP formulation will be transformed into vectors by means of Graph Neural Networks (GNN). Then the designed value and policy networks will be learnt by using different DRL approaches such as REINFORCE, PPO, Deep-Q Learning (DQN), SARSA and so on. The methods will be tested on open access datasets and the results will be compared with existing heuristic, metaheuristic and the learning based methods in the literature.

The proposed DRL approach in the project differs from existing deep learning approaches in two aspects. Firstly, it applies the DPL approach specifically to the graph representation of problems. Secondly, it excels in scalability. Deep learning approaches in previous literature tend to be dependent on the problem size. So it requires training for each problem of different sizes. However, the proposed DRL approach graphically represents states in the MDP formulation, independent of the problem size. This allows for generalisation of the reinforcement learning network, which can be trained to handle problems of varying sizes, including those previously unseen.

## Keywords
Combinatorial Optimization, Deep Reinforcement Learning, Bin Packing Problem, Cutting Stok Problem
## Test Instances
List of resources for test instances:
### BBPLIB - A Bin Packing Problem Library
    https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library
 2DPackLib 
    https://site.unibo.it/operations-research/en/research/2dpacklib
 ### OR-Library
    1D BPP: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html
    2D BPP: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpacktwoinfo.html
 ### ESICUP - Cutting and PAcking
    https://www.euro-online.org/websites/esicup/data-sets/

 ### SCIP Optimization Suite
    https://www.scipopt.org/doc/html/probdata__binpacking_8c.php










