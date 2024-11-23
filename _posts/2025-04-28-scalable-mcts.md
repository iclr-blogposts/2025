---
layout: distill
title: Understanding Methods for Scalable MCTS
description: Monte Carlo Tree Search (MCTS) is pivotal for decision-making in environments with large state spaces, such as strategic games and real-world optimization. This blog post explores scalable methods to enhance MCTS through various forms of parallelism and distribution strategies, shedding light on techniques that maintain performance while maximizing computational efficiency. Our exploration includes detailed analyses of lock-based, lock-free, and distributed MCTS approaches, demonstrating their application in high-performance settings.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
    - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-scalable-mcts.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
    - name: Intro
    - name: MCTS Background
      subsections:
          - name: Vanilla MCTS
          - name: The UCT Selection Policy
          - name: MCTS Extensions
    - name: How can we scale MCTS?
      subsections:
          - name: Leaf Parallelism
          - name: Root Parallelism
          - name: Tree Parallelism
    - name: Lock-Based Tree Parallelism
    - name: Lock-Free Tree Parallelism and Virtual Loss
    - name: Transposition-Table Driven Scheduling (TDS)
    - name: Distributed MCTS
    - name: Distributed Depth-First MCTS
    - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
    .fake-img {
      background: #bbb;
      border: 1px solid rgba(0, 0, 0, 0.1);
      box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
      margin-bottom: 12px;
    }
    .fake-img p {
      font-family: monospace;
      color: white;
      text-align: left;
      margin: 12px 0;
      text-align: center;
      font-size: 16px;
    }
---

## Intro

Recently, there has been a large focus on training Large Language Models (LLMs) with up to trillions of parameters, using vast amounts of compute for training. However, expecting these models to produce perfect answers instantaneously -- especially for complex queries -- can seem impractical. Naturally, the AI industry is shifting from simply scaling model size to optimizing test-time compute. This shift involves finding ways to harness computational resources effectively during inference. One promising approach is leveraging search algorithms, which enable models to plan, reason, and iteratively refine their outputs. If we can use search in a scalable way, we can leverage vast compute resources to tackle complex challenges such as proving the Riemann Hypothesis or discovering drugs for rare diseases. In this blog post, we will focus our discussion on one such method for leveraging test-time compute: **Monte Carlo Tree Search (MCTS)**.

Rich Sutton’s [_bitter lesson_](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) encapsulates a key insight that is highly relevant to this shift in focus:

> “One thing that should be learned from the bitter lesson is the great power of general-purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are **search** and **learning**.”

This lesson underscores the importance of scalable methods like search, which can capitalize on increased computational power to deliver more robust results. But how effective can search really be? The power of search-based planning becomes strikingly evident when examining MuZero's performance in the game of Go. As computational resources increase, MuZero demonstrates remarkable scaling properties:

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/2025-04-28-scalable-mcts/mu_zero_scaling.png" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">
  Impact of additional search time given to MuZero on performance in Go<d-cite key="schrittwieser2020mastering"></d-cite>
</div>

The empirical results reveal a characteristic relationship between search depth and playing strength typically seen in MCTS applications -- as the number of simulations increases, performance typically improves logarithmically, where each doubling of computational resources adds a relatively constant increment to playing strength<d-cite key="camacho2017mcts"></d-cite>. This pattern is clearly demonstrated in MuZero's Go performance, where increased search depth consistently yields improved results. It's worth noting that MuZero uses a form of tree parallelism called virtual loss to evaluate the N most promising positions in parallel, an approach that we will describe in more detail later.

MCTS algorithms have proven successful in many domains having large state spaces (e.g. Go, chess, protein folding, molecular design). However, it is difficult to parallelize MCTS without degrading its performance, since each iteration requires information from all previous iterations to provide an effective exploration-exploitation tradeoff<d-cite key="liu2020watch"></d-cite>. In this blogpost, we will explain and analyze different methods for effectively scaling MCTS.

## MCTS Background

Monte Carlo Tree Search (MCTS) is a powerful algorithm for decision-making in large state spaces, commonly used in games, optimization problems, and real-world domains such as protein folding and molecular design. MCTS stands out for its ability to search complex spaces without the need for additional heuristic knowledge, making it adaptable across a variety of problems. Before MCTS became prominent, techniques like minimax with alpha-beta pruning were the standard in game AI. While alpha-beta pruning could efficiently reduce the search space, its effectiveness often depended on the quality of evaluation functions and move ordering. MCTS offered a different approach that could work without domain-specific knowledge, though both methods can benefit from incorporating heuristics<d-cite key="swiechowski2022mcts"></d-cite>.

### Vanilla MCTS

MCTS operates within an environment, which defines the possible states and actions, along with the rules for transitioning between states and assigning rewards. The algorithm iteratively builds a search tree, collecting statistics for each node, including how many times it has been visited and the average reward obtained from simulations passing through it. These statistics guide the decision-making process, enabling the algorithm to intelligently balance exploration and exploitation. The algorithm operates in four main phases:

{% include figure.html path="assets/img/2025-04-28-scalable-mcts/mcts_phases.png" class="img-fluid rounded" %}

<div class="caption">
    Diagram taken from Wikipedia<d-cite key="wiki:mcts"></d-cite>
</div>

1. **Selection**: Starting from the root node, MCTS selects child nodes according to a policy that balances exploration and exploitation. This continues until either a node with unexplored children is reached (where expansion is possible) or a terminal node is reached.
2. **Expansion**: If the selected node is not terminal, the algorithm chooses an unexplored child node randomly and adds it to the tree.
3. **Simulation**: From the newly expanded node, a simulation (or "rollout") is conducted by sampling actions within the environment until a terminal state is reached, using a default policy (often random). The cumulative reward obtained during this phase serves as feedback for evaluating the node.
4. **Backpropagation**: Once a terminal state is reached, the simulation's result is propagated back up the tree. At each node along the path, visit counts are incremented, and average rewards are updated to reflect the outcomes.

MCTS is an **anytime** algorithm, meaning it can be stopped at any point during its execution and still return the best decision found up to that point. This is particularly useful when dealing with problems where the state space (e.g., a game tree) is too large to be fully explored. In practical applications, MCTS operates within a computational budget, which could be defined by a fixed number of iterations or a set amount of time.

The decision-making process in MCTS can be represented mathematically. At any point during the search, MCTS recommends the best action $a^*$ based on the current statistics:

$$
a^* = \underset{a \in A(s)}{\operatorname{argmax}} Q(s, a) \tag{1}
$$

where:

-   $A(s)$ represents the set of actions available in state $s$.
-   $Q(s, a)$ represents the average reward from playing action $a$ in state $s$ based on simulations performed so far.

This equation implies that as more iterations are completed, the statistics for each action become more robust, increasing confidence that the suggested action $a^*$ approaches the optimal solution.

### The UCT Selection Policy

A key component of MCTS is the **Upper Confidence Bounds for Trees (UCT)** algorithm, introduced by Kocsis and Szepesvári<d-cite key="kocsis2006bandit"></d-cite>. UCT adapts the UCB1 algorithm to determine how child nodes are selected during the selection phase. There are two cases. (1) If a given node has not expanded all of its leaf nodes, then we expand them randomly. (2) Otherwise we select the node with the highest UCT value. The aim of the selection policy is to maintain a proper balance between the exploration (of not well-tested actions) and exploitation (of the best actions identifed so far)<d-cite key="swiechowski2022mcts"></d-cite>. The optimal action at any point in the search is selected by maximizing the UCT score:

$$
a^* = \underset{a \in A(s)}{\operatorname{argmax}} \left\{ Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}} \right\} \tag{2}
$$

where:

-   $a^*$ is the action selected from state $s$.
-   $A(s)$ is the set of actions available in state $s$.
-   $Q(s, a)$ represents the average result of playing action $a$ in state $s$ based on simulations performed so far.
-   $N(s)$ is the number of times state $s$ has been visited.
-   $N(s, a)$ is the number of times action $a$ has been played from state $s$.
-   $C$ is a constant controlling the balance between exploration and exploitation. In general, it is set differently depending on the domain.

We will now move on to explaining an extension to vanilla MCTS and then dive in to the different ways to implement a scalable distributed MCTS solution. If you would like to learn more about Vanilla MCTS, I would highly recommend [this blogpost](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/ "Introduction to Monte Carlo Tree Search") on MCTS by Jeff Bradberry.

### MCTS Extensions

While we have discussed the vanilla Monte Carlo Tree Search (MCTS) algorithm, there are numerous modifications which enhance its flexibility and applicability across different domains. These modifications adapt MCTS for a variety of complex scenarios including games with both perfect and imperfect information, and extend its utility to real-world applications such as planning, security, and chemical synthesis. For those interested in a deeper exploration of these extensions, a detailed overview can be found in the comprehensive survey by Swiechowski et al.<d-cite key="swiechowski2022mcts"></d-cite>

## How can we scale MCTS?

To find better solutions without increasing [wall-clock time](https://en.wikipedia.org/wiki/Elapsed_real_time), we can consider how the four different phases of MCTS can be distributed. There are three broad types of parallelism used to scale MCTS, and we will discuss each one below.

{% include figure.html path="assets/img/2025-04-28-scalable-mcts/mcts_parallelism.png" class="img-fluid" %}

<div class="caption">
    Diagram taken from Chaslot et al.<d-cite key="chaslot2008parallel"></d-cite>
</div>

### Leaf Parallelism

Leaf parallelism focuses on the simulation phase of MCTS. By distributing the rollout simulations across multiple workers, it achieves a high degree of parallelism with minimal coordination overhead. Once we expand a new leaf node, we execute rollouts on multiple different workers, getting more rollout results in the same amount of wall-clock time. Leaf parallelism reduces the variance of our rollouts, giving us more precise estimates of the true value of a given node. Similar to root parallelism, leaf parallelism requires no communication between worker machines until the results of the rollouts are aggregated. Although leaf parallelism increases the speed of statistical accumulation, its effectiveness can be limited by the quality of rollouts. If early simulations indicate a low value, running more simulations can be a complete waste of compute. Chaslot et al. find that the strength of leaf parallelism is rather low, and suggest that it is not a good way to parallelize MCTS<d-cite key="chaslot2008parallel"></d-cite>.

### Root Parallelism

Root parallelism takes the approach of distributing all phases onto separate worker machines. Each worker uses MCTS to build a **separate** tree concurrently. This approach is straightforward and avoids the complexities associated with managing shared memory. It is analogous to ensemble methods in machine learning where multiple models (trees, in this case) vote on the best action. Each tree explores different trajectories independently, which can provide a broader exploration of the state space but may result in redundant computations. The independence of each tree allows for diverse explorations of the search space, increasing the likelihood of escaping local optima that a single search might get trapped in<d-cite key="chaslot2008parallel"></d-cite>. When the available time is spent, we aggregate the results of the different trees and select the most promising action based on a majority vote or averaging method. Workers do not share information with each other until the final aggregation step, so this method requires minimal communication. While root parallelism is easy to implement and requires little communication, it often lacks efficiency because all instances explore the same or similar states without leveraging the knowledge gained by others.

### Tree Parallelism

Finally, we describe tree parallelism, which we will focus on for the remainder of the blogpost. This is the method of choice for the AlphaGo line of work<d-cite key="schrittwieser2024questions"></d-cite> and is very popular in modern distributed implementations of MCTS. Tree parallelism, involves multiple threads or processes working on the same tree. It aims to expand the tree more quickly by allowing concurrent operations on different parts of the tree. The main challenge with tree parallelism is in balancing maximum utilization of compute resources with evaluating the most promising parts of the tree. There are many reasons that we may want to parallelize:

-   It will enable us to explore more of the game tree, making it more likely that we find the optimal trajectory (sequence of actions)
-   It makes full use of accelerators (GPUs, TPUs, etc.)

And at the same time there are many reasons to work sequentially:

-   Previous iterations yield statistics indicating the most promising actions in the tree
-   Evaluating actions that are far away the optimal trajectory may actively hurt us, as approximation errors and variability in rollouts can make a bad move seem promising in the short term<d-cite key="schrittwieser2024questions"></d-cite>
-   Parallelism introduces challenges since we need to manage concurrent access to the tree and ensure data integrity<d-cite key="steinmetz2020more"></d-cite>

Techniques such as lock-free programming, virtual loss, and transposition-table driven scheduling have been developed to manage these issues. Below, we will explain these different tree parallelism techniques in more detail.

## Lock-Based Tree Parallelism

As mentioned, one of the main challenges with tree parallelism is dealing with resource contention. There are two main approaches used in the literature: **global mutex** and **local mutex** methods. Global mutex methods lock the entire tree such that only one thread can access the tree at a time. However, rollouts (phase 3) can still be completed in parallel, since they don't modify the tree. Locking the entire tree creates a bottleneck that prevents speedups when scaling to a large number of threads. So, while this approach is easy to implement due to its straightforward synchronization, the limited scalability is a huge issue that cannot be ignored.

Local mutex methods make locking much more fine-grained, allowing for mutliple workers to work in parallel on the same tree. In local mutex methods, threads only lock the node they are currently working on. Since this allows multiple workers to operate on different parts of the tree, local mutex methods have much greater scalability compared to global mutex methods. However, while this method increases parallelism, it also increases overhead since we have to manage multiple locks. To mitigate this, Chaslot et al. recommend using fast-access mutexes like spinlocks, which are designed for situations where threads are expected to hold locks for very short durations<d-cite key="chaslot2008parallel"></d-cite>.

## Lock-Free Tree Parallelism and Virtual Loss

Since lock-based methods add aditional overhead and require a complex implementation, there are also methods that don't use locks and accept a small probability of data overwrites as an optimization tradeoff<d-cite key="steinmetz2020more"></d-cite>. These methods rely on atomic operations and memory models to enable lock-free concurrency with low probability of overwrites.

One method used frequently in lock-free implementations to reduce probability of overwrites and encourage exploration of other nodes is **virtual loss**. If we were to naively parallelize the rollouts by having K workers run rollouts based on the same values and visit counts, they will all use the same path within the known tree and we will effectively have no exploration. Virtual loss solves this by assuming that each ongoing rollout will give 0 reward, reducing the UCT score of the current node and discouraging other threads from visiting it until its results are backpropagated. This adjustment ensures that parallel workers are less likely to redundantly explore the same subtree. This diagram from Yang et al. shows what this looks like:

{% include figure.html path="assets/img/2025-04-28-scalable-mcts/virtual_loss_traversal.png" class="img-fluid" %}

<div class="caption">
  (a) parallel UCT using UCB1 (failed) (b) parallel UCT with virtual loss, and the search paths of three parallel workers shown in solid circles, (green, red, and blue, from left to right)<d-cite key="yang2021practical"></d-cite>.
</div>

Virtual loss modifies the usual UCT selection policy to be the following:

$$
a^* = \underset{a \in A(s)}{\operatorname{argmax}} \left( \frac{W(s, a)}{N(s, a) + T(s, a)} + C \sqrt{\frac{\ln (N(s) + T(s))}{N(s, a) + T(s, a)}} \right) \tag{3}
$$

where:

-   $a^*$ is the action selected from state $s$.
-   $A(s)$ is the set of actions available in state $s$.
-   $W(s, a)$ represents the sum of rewards from playing action $a$ in state $s$ based on simulations performed so far.
-   $N(s)$ is the number of times state $s$ has been visited.
-   $N(s, a)$ is the number of times action $a$ has been played from state $s$.
-   $T(s)$ is the number of workers currently exploring action $a$ in state $s$.
-   $T(s, a)$ is the number of workers currently exploring action $a$ in state $s$.
-   $C$ is a constant controlling the balance between exploration and exploitation. In general, it is set differently depending on the domain.

The effectiveness of virtual losses depends on careful tuning of their magnitude. If the virtual loss is too small, it might fail to deter other threads, leading to redundant exploration and potential overwriting of data. Conversely, excessively large virtual losses may overly penalize promising nodes, hindering exploration and reducing the algorithm's overall performance.

Liu et al. change the standard virtual loss formula by assuming that the reward remains unchanged, yet use the same modification for the exploration term. Thus, the optimal action is selected by maximizing the modified UCT score:

$$
a^* = \underset{a \in A(s)}{\operatorname{argmax}} \left(Q(s, a) + C \sqrt{\frac{\ln (N(s) + T(s))}{N(s, a) + T(s, a)}} \right) \tag{4}
$$

## Distributed MCTS

While the previous sections primarily focus on storing the tree on a single machine, scaling MCTS effectively often requires distributing the tree and associated statistics across multiple machines. This shift is essential for tackling even larger state spaces or executing more complex simulations within the same wall-clock time.

In addition to modifying the virtual loss formula, Liu et al. propose a system for distributing different phases of MCTS. The system has many moving parts but is straightforward to understand. The authors define three types of workers: the master worker, expansion workers, and simulation workers. The master worker runs the main MCTS logic and stores the truth version of all global data structures (i.e. nodes in the tree, tree structure, and values and visit counts for each node). It runs until it has successfully completed a predetermined number of rollouts. Crucially, each rollout can be broken down into two distinct phases: (1) the expansion phase in which the known tree is traversed using UCT until a leaf is reached and (2) the simulation phase, which starts at a leaf in the known tree and continues to a terminal state making random traversal decisions. In Liu et al's framework, these steps are performed independently on different types of workers.

Expansion workers take in a tree state from the master and run the UCT selection policy until a leaf is reached. They then report the path taken to the master and the new leaf is placed in a simulation buffer. At this point, master makes an _incomplete update_ in which it increments the _active rollout counter_ $O_s$ for each node s on the path received from the expansion node.

Meanwhile, simulation nodes take in a state (represented by a leaf chosen from the simulation buffer) and randomly play until a terminal state is reached. The simulation nodes then return the reward achieved during the random simulation to the master.

The master node finally back propagates this return and updates the visit counts $N(s)$ (increment) and active rollout counts $T(s)$ (decrement) for corresponding tree nodes. This algorithm can be best understood by tracing the below block diagram, which was included in the original paper:

{% include figure.html path="assets/img/2025-04-28-scalable-mcts/wu_mcts_diagram.png" class="img-fluid" %}

While there are many rounds of communication, the authors showed that the expansion and simulation steps took longer than the communication steps in their system, suggesting communication is not a bottleneck.

To test their procedure, the authors benchmarked against sequential MCTS and found only slight performance degradation on an RL benchmark suite of 15 Atari games. This method also outperformed existing parallel MCTS algorithms on the benchmark suite, beating standard leaf and root parallelization implementations on 15/15 games and standard tree parallelization on 13/15.

In short, Liu et al break down the MCTS process into two distinct phases (expansion and simulation) which can be parallelized by implementing an adjusted UCT selection policy that tracks the number of rollouts that are active in addition to the number of times each node in the tree has been visited. The empirical results of the study demonstrate that this is an effective MCTS parallelization method, with only limited performance setback vs sequential MCTS and a strong showing against previous parallelized MCTS methods.

## Transposition-Table Driven Scheduling (TDS)

In 2011, Yoshizoe et al<d-cite key="yoshizoe2011scalable"></d-cite> introduced an efficient scheme for parallelized MCTS with two core ideas: transposition-table driven scheduling (TDS) parallelism and depth-first MCTS. First, we will discuss these topics separately and then see how the the two concepts combine for efficient parallelization.

At a high level, TDS is a mechanism for evenly distributing data across W worker nodes and efficiently distributing work to be done on said data. In TDS, each record of data is passed through a hash function that maps it to one of the worker nodes. This is the record’s “home” -- the record is stored in memory on its home worker and on no other workers. With a good hash function, this scheme ensures that the data is partitioned evenly across a network of workers. The key idea behind TDS is that data doesn’t move, requests do. If worker A receives a request to run some function on a partition of data that resides on worker B, the request is forwarded to worker B and the response is computed locally on B before being sent back to worker A. This design is very efficient, as moving data across a network is generally far slower than encoding requests to process that data and sending the requests where the data resides.

An important observation to make is that TDS communication is asynchronous. When worker A receives a request to process data that actually resides on worker B, it siimply forwards the request to B. Worker A does not need to pause local computation and wait for worker B’s response before handling other requests. Thus, even though communication is frequent in this scheme, its cost is offset by its inherent asynchronicity.

Before thinking about how TDS could support distributed MCTS, let’s understand how it could support sequential MCTS. Yoshizoe et al build a distributed search tree by assigning each node to a unique worker using a hash function. Each time a new node is discovered via the MCTS search process, its home worker is computed and memory is allocated for the new node's metadata: its parent, its children, its value, and its visit count.

The authors describe a distributed MCTS process where tasks are delegated by the root node to workers using a hash-based mapping. Workers either expand the tree by exploring new nodes or conduct rollouts on child nodes selected according to the UCT policy. Once a terminal state is reached, results are propagated back up the tree, incrementing visit counts and adjusting values at each node along the path. Although asynchrony may lead to occasional use of outdated node statistics, the authors demonstrate that this distributed approach closely approximates sequential MCTS while achieving significant speedups through parallelism.

Before moving forward, we must address a major flaw in this paradigm. As the number of nodes grows, the number of requests sent across the system grows accordingly. However, even though the nodes are roughly evenly distributed across the workers via the TDS hashing protocol, the work done on each worker node is not evenly distributed. Workers that house nodes “higher up” in the search tree which are visited more frequently process far more jobs than workers that do not have such nodes. This problem is especially clear for the worker that houses the root node, which must process requests for every single rollout handled by the system.

## Distributed Depth-First MCTS

In an attempt to mitigate the uneven distribution of job requests, the authors propose using a depth-first version of MCTS. The core innovation found in df-MCTS is that the frequency of backpropagation steps (i.e., the frequency of Report() messages) is greatly reduced compared to normal MCTS. df-MCTS acts like normal MCTS when it receives a Search() job, but it doesn’t always report results to its parent when a Report() message is received. Instead, it either sends a new Search() message to one of its children if more exploration is needed or, if the node has been thoroughly explored, it sends a Report() message to its parent. The exact threshold for when to send a Search() or a Report() in response to a child’s Report() is not fleshed out by the authors and is implementation dependent.

The authors tested their system against standard, sequential MCTS on a 2-player benchmark game. They found that if both algorithms trained for the same amount of rollouts, sequential MCTS slightly outperforms df-MCTS. This is expected, as the delayed propagation of rewards under df-MCTS means that some exploration decisions are made based on stale UCT statistics. However, when both algorithms were allowed to train for the same amount of wall-clock time, the distributed system significantly outperformed sequential MCTS as it was able to perform substantially more rollouts in the same time period.

The final system presented by the authors enables parallelized MCTS by (1) evenly distributing the search tree and its metadata across W workers under a partitioning based on TDS, (2) performing all work on the partitioned tree locally without sending data across the network, and (3) better balancing the work done on each worker node by running a depth-first version of MCTS which takes backpropagation steps less frequently and puts less strain on the worker housing the root node. The distributed MCTS system significantly outperformed sequential MCTS given the same amount of wall-clock train time in the author's experiments, suggesting it is a viable approach to parallelizing MCTS.

## Conclusion

MCTS is a powerful algorithom for that serves as the backbone behind pivotal technologies such as AlphaZero with the potential to be applied in any domain requiring efficient tree search or sampling. MCTS in its original form cannot be parallelized as it is inherently sequential; current iterations rely on statistics aggregated over all previous iterations. Although the exact MCTS algorithm cannot be parallelized across mulitple workers, there is a wide body of existing literature presenting distrbuted schemes that approximate vanilla MCTS and empirically outperform it when controlling for wall-clock time. We hope that in reading this report our audience has become familiar with the basic techniques behind distributed MCTS in root, leaf and tree parallelization.

Ultimately, no single distibuted MCTS algorithm outperforms all others. Distributed MCTS algorithms differ in how often they send messages, the size of messages sent, and the roles of nodes in the system. The choice of which distributed MCTS algorithm to use requires informed tradeoffs based on sound knowledge of your network and compute capabilities. But, emprirical evidence shows that taking the time to parallelize MCTS is well-worth the effort, as learning speed (as measured by wall-clock time) greatly increases in a distributed setting.
