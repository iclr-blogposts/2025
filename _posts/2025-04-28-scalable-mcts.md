---
layout: distill
title: Understanding Methods for Scalable MCTS
description: Your blog post's abstract.
    Please add your abstract or summary here and not in the main body of your text.
    Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
    - name: Will Knipe
      url: ""
      affiliations:
          name: CMU
    - name: Nate Rising
      url: ""
      affiliations:
          name: CMU

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
    - name: Intro
    - name: MCTS Background
    - name: Root Parallelism
    - name: Tree Parallelism
    - name: Leaf Parallelism
    - name: Transposition Driven Scheduling (TDS)
    - name: Equations

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

Recently, there has been a large focus on training Large Language Models (LLMs) with up to trillions of parameters, using vast amounts of compute for training. However, expecting these models to produce perfect answers instantaneously -- especially for complex queries -- can seem impractical. Naturally, the AI industry is shifting from simply scaling model size to optimizing test-time compute. This shift involves finding ways to harness computational resources effectively during inference. One promising approach is leveraging search algorithms, which enable models to plan, reason, and iteratively refine their outputs. If we can use search in a scalable way, we can allocate compute to tackle complex challenges such as proving the Riemann Hypothesis or discovering drugs for rare diseases. In this blog post, we will focus our discussion on one such method for leveraging test-time compute: **Monte Carlo Tree Search (MCTS)**.

Rich Sutton’s _bitter lesson_ encapsulates a key insight that is highly relevant to this shift in focus:

> “One thing that should be learned from the bitter lesson is the great power of general-purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.”

This lesson underscores the importance of scalable methods like search, which can capitalize on increased computational power to deliver more robust results. MCTS algorithms have proven successful in many domains having large state spaces (e.g. Chess, Go, protein folding, molecular design). However, it is difficult to parallelize MCTS without degrading its performance, since each iteration requires information from all previous iterations to provide an effective exploration-exploitation tradeoff (Cite: WU-MCTS). In this blogpost, we will explain and analyze different methods for effectively scaling MCTS.

## MCTS Background

Monte Carlo Tree Search (MCTS) is a powerful algorithm for decision-making in large state spaces, commonly used in games, optimization problems, and real-world domains such as protein folding and molecular design. MCTS stands out for its ability to search complex spaces without the need for additional heuristic knowledge, making it adaptable across a variety of problems. Before the advent of MCTS, techniques like minimax with alpha-beta pruning were the standard in game AI, but they were limited by their reliance on complete tree exploration and domain-specific heuristics.

### Classic MCTS

MCTS iteratively builds a search tree, collecting statistics on potential actions to make intelligent decisions. The algorithm operates in four main phases:

1. **Selection**: Starting from the root node, MCTS selects child nodes according to a policy that balances exploration and exploitation. This continues until a leaf node is reached, where expansion is possible.
2. **Expansion**: If the selected node is not a terminal state, the algorithm adds one or more child nodes to the tree, representing potential future states.
3. **Simulation**: From the newly expanded node, a simulation (or "rollout") is conducted. This is typically done by playing out the game or continuing through the state space until a terminal state is reached, using a default policy (often random).
4. **Backpropagation**: Once a terminal state is reached, the result of the simulation is propagated back up the tree, updating the statistics of the visited nodes to reflect the outcome.

MCTS is an **anytime** algorithm, meaning it can be stopped at any point during its execution and still return the best decision found up to that point. This is particularly useful when dealing with problems where the state space (e.g., a game tree) is too large to be fully explored. In practical applications, MCTS operates within a computational budget, which could be defined by a fixed number of iterations or a set amount of time.

The decision-making process in MCTS can be represented mathematically. At any point during the search, MCTS recommends the best action \( a^\* \) based on the current statistics, as shown in Equation 1:

$$
a^* = \arg \max_{a \in A(s)} Q(s, a)
$$

where:

-   $A(s)$ represents the set of actions available in state $s$.
-   $Q(s, a)$ denotes the empirical average result of selecting action $a$ in state $s$.

This equation implies that as more iterations are completed, the statistics for each action become more robust, increasing confidence that the suggested action \( a^\* \) approaches the optimal solution.

### The UCT Selection Policy

A key component of MCTS is the **Upper Confidence Bounds for Trees (UCT)** algorithm, which determines how child nodes are selected during the selection phase. The UCT formula is given by:

$$
UCT(v) = \frac{w(v)}{n(v)} + c \sqrt{\frac{\ln N(v_p)}{n(v)}}
$$

where:

-   $w(v)$ is the total reward of node $v$.
-   $n(v)$ is the number of times node $v$ has been visited.
-   $N(v_p)$ is the number of times the parent node has been visited.
-   $c$ is an exploration parameter that controls the tradeoff between exploration and exploitation.

The UCT policy ensures that nodes with higher potential (even if they have been visited fewer times) are explored, balancing between revisiting promising paths and exploring new ones.

## Root Parallelism

## Tree Parallelism

## Leaf Parallelism

## Transposition Driven Scheduling (TDS)
