---
layout: distill
title: "Inside SMACv2: Building and Extending MARL Algorithms"
description: 
  The StarCraft Multi Agent Challenge v2 (SMACv2) environment provides a robust platform for testing and developing Multi-Agent Reinforcement Learning (MARL) algorithms, but navigating its codebase can be challenging. This blog post serves as a comprehensive guide to understanding the structure of the MARL algorithms within the SMACv2 framework. 
  By breaking down the key components—such as agent networks, mixer networks, and transformation networks—we clarify how these algorithms are implemented and highlight the critical locations in the codebase for editing or extending functionality. We also provide a step-by-step walkthrough for adding new algorithms to the framework, lowering the barrier to entry for researchers and developers. Whether you're seeking to adapt existing algorithms or contribute novel approaches, this guide provides the necessary knowledge to navigate and enhance the SMACv2 ecosystem.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-inside-smacv2-building-and-extending-marl-algorithms.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.

toc:
  - name: Introduction
  - name: Main Components of Any Algorithm
  - name: QPLEX Algorithm
    subsections:
    - name: Agent Network
    - name: Transformation Network
    - name: Mixing Network
  - name : How to Add a New Algorithm
  - name : Common Pitfalls
  - name : Future Work
  - name : Conclusion


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

## Introduction

The StarCraft Multi Agent Challenge v2 (SMACv2) environment is a commonly used testbed for multi-agent reinforcement learning algorithms. The authors share the implementation of the environment in the [SMACv2 repository](https://github.com/oxwhirl/smacv2/tree/main). Moreover, to overcome the [reproducibility crisis](https://neuripsconf.medium.com/designing-the-reproducibility-program-for-neurips-2020-7fcccaa5c6ad), the authors share the instructions to replicate some of their experiments in [RUNNING_EXPERIMENTS.md](https://github.com/oxwhirl/smacv2/blob/main/RUNNING_EXPERIMENTS.md). This takes us to the [pymarl2 repository](https://github.com/benellis3/pymarl2) where many value factorisation algorithms have been implemented including VDN<d-cite key="VDN"></d-cite>, QMIX<d-cite key="QMIX"></d-cite>, QPLEX<d-cite key="QPLEX"></d-cite>, etc.

The [pymarl2 repository](https://github.com/benellis3/pymarl2) has a variety of value factorisation algorithms that are implemented efficiently. However, like many respositories, the repository is quite complex to understand and edit. Moreover, there are some implementation details that do not match the theory in the respective papers of said algorithms. Our aim with this blog is to reduce the entry barrier to implementing custom algorithms to complex environments like SMACv2 and foster better algorithms than the ones in the existing literature. 

In this blog, we explain where specific parts of an algorithm are located to facilitate the easy editing/omiting of those parts for new algorithms. We also match the theory to the code where we can while also trying to explain where the theory and implementation don't match.

Note: If there are any errors in this explanation or if you think this blog would benefit from adding more details, please feel free to reach out to us! Moreover, as the repo is updated, we will try to keep this blog updated as well (but there may be a delay).

## Main Components of Any Algorithm
1. Agent network
- These are found in src -> modules -> agents.
- First refer to src -> modules -> agents -> \__init__.py. Here, all the different types of agent networks are registered.
2. Mixer network
- These are found in src -> modules -> mixers.
3. Hyperparameters
- These are found in src -> config -> algs
- For any algorithm, first refer to default.yaml where the default parameters are specified. Then refer to the [algorithm].yaml file (e.g., qplex.yaml) where some parameters/hyperparameters would be added/overwritten.
4. The algorithm
- These are found in src -> learners.
- Refer to src -> learners -> \__init__.py to see how all the algorithm learners are registered. The names of the files here may not be intuitive, so refer to the yaml file of that specific algorithm (found in src -> config -> algs) to see which learner file is being used.
5. Shapes of Some Key Elements:  
- state = (B, T, state_dim)  
- obs = (B, T, A, obs_dim)  
- actions = (B, T, A, 1) # chosen actions  
- avail_actions = (B, T, A, n_actions)  
- probs = (B, T, A, 1) # probabilities of the chosen action  
- reward = (B, T, 1)  
- terminated = (B, T, 1)  
- actions_onehot = (B, T, A, n_actions)   

where, B = batch size, T = total number of timesteps in an episode, A = number of agents, state_dim = length of the state vector (found in [SMACv2 repository](https://github.com/oxwhirl/smacv2/tree/main)), obs_dim = length of the observation vector (found in [SMACv2 repository](https://github.com/oxwhirl/smacv2/tree/main)), n_actions = total number of actions that an agents can take.  

In the remainder of the blog, we go deeper into one algorithm. Majority of the algorithms follow a similar structure. Therefore, if the basic flow of one algorithm is clear, the others will follow.

## QPLEX Algorithm

Now let's take an example algorithm and see where the parts of it are located and try to compare the theory to the code. We chose to start with the QPLEX algorithm. The architecture for the QPLEX algorithm <d-cite key="QPLEX"></d-cite> is shown below.
<div style="max-width: 600px; margin: 0 auto;">
{% include figure.html path="assets/img/2025-04-28-inside-smacv2-building-and-extending-marl-algorithms/QPLEX_architecture.png" class="img-fluid" %}
</div>

### 1. Agent Network
  Let's start bottom up from the agent network, i.e., the bottom-green part of Figure (c) in the architecture.
  <div style="max-width: 140px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-inside-smacv2-building-and-extending-marl-algorithms/QPLEX_agent_network.png" class="img-fluid" %}
  </div>
This agent network is found in src -> modules -> agents -> rnn_agent.py. There are two outputs to the `forward` function: 

```
return q.view(b, a, -1), h.view(b, a, -1)
```

- q : The q is $Q_i(\tau_i, a_i)$ in the diagram above. 
- h : The h represents $$h_i^t$$, i.e., the next hidden state of the GRU in the diagram above.


### 2. Transformation Network

  <div style="max-width: 160px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-inside-smacv2-building-and-extending-marl-algorithms/QPLEX_transformation_network.png" class="img-fluid" %}
  </div>

The transformation network is the top brown part of Figure (c) in the architecture.
In the repository, this is found in src -> modules -> mixers -> dmaq_general.py. There are two main operations happening here: calculating the output of the MLP and calculating the Weighted Sum. Let's see what is happening in both of these operations.


**(a) Calculating the output of the MLP**

There are 2 outputs of the MLP. One is the w and the other is b. In the `forward` function of dmaq_general.py, 

```
w_final = self.hyper_w_final(states)
w_final = th.abs(w_final)
```

The w\_final in the code is the w in the diagram above. The hyper\_w\_final is the MLP in the diagram above.


To calculate the b, once again, in the `forward` function, we have:

```
v = self.V(states)
```

The v in the code represents the b in the diagram above.

Note that in the figure, we just see 1 MLP. However, in the code there are 2 MLPs. One is hyper\_w\_final and another is V. This is because the following equation in the paper (the first part of equation 7 from the QPLEX paper <d-cite key="QPLEX"></d-cite>):

$$V_i(\mathbf{\tau}) = w_i(\mathbf{\tau})V_i(\tau_i) + b_i(\mathbf{\tau}) $$

Therefore, hyper\_w\_final represents the $$w_i(\mathbf{\tau})$$ and V represents the $$b_i(\mathbf{\tau})$$ in the above equation, respectively.



**(b) Calculating the Weighted Sum**

In the code, the advantage values are not calculated directly. The Q values and V values are calculated. Note that advantage can be calculated using the Q and V values, however, the authors find it easier to just use the Q and V values. Therefore, in the code, the output to the Transformation Network are $$V_i(\mathbf{\tau})$$ and $$Q_i(\mathbf{\tau}, a_i)$$ rather than $$V_i(\mathbf{\tau})$$ and $$A_i(\mathbf{\tau}, a_i)$$.


These Q and V values are calculated in the `forward` function as follows:

```
agent_qs = w_final * agent_qs + v
```

Here agent_qs are the Q values.

```
max_q_i = w_final * max_q_i + v
```

Here max\_q\_i are the V values because $$V_i(\tau) = max_{a_i} Q(\tau, a_i)$$


### 3. Mixing Network
  <div style="max-width: 180px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-inside-smacv2-building-and-extending-marl-algorithms/QPLEX_mixing_network.png" class="img-fluid" %}
  </div>

The mixing network is Figure (a) in the architecture. In the repository, the mixing network for the QPLEX algorithm is in the same file as the transformation network. Found in src -> modules -> mixers -> dmaq_general.py. There are 
four main operations happening here: calculating the output of the MLP, calculating the Dot Product, summing over the $$V_i$$ values and summing the $$V_{tot}$$ and the $$A_{tot}$$. Below we explain these four operations:

**(a) Calculating the output of the MLP**

In the `calc_adv` function of dmaq_general.py, 
```
adv_w_final = self.si_weight(states, actions)
```
The `adv_w_final` are the $$\lambda$$ weights. The function `si_weight` implements the multi-head attention mechanism and is implemented in src -> modules -> mixers -> dmaq_si_weight.py.

**(b) Calculating the Dot Product**

In the `forward` function of dmaq_general.py, 
```
y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
```
The function `calc` is used to calculate either $$V_{tot}$$ or $$A_{tot}$$. If the parameter `is_v` is set to True, it returns $$V_{tot}$$ and if it is set to False, the `calc` function returns $$A_{tot}$$. Since the $$A_{tot}$$ is being calculated here, the `is_v` parameter is set to False. The dot product is implemented in the `calc_adv` function in the same file as follows:

```
adv_tot = th.sum(adv_q * adv_w_final, dim=1)
```
Here `adv_q` are the $$A_i(\mathbf{\tau}, a_i)$$s and `adv_w_final` are the $$\lambda$$ weights. 

NOTE: In src -> modules -> mixers -> dmaq_general.py, in the `calc_adv` function when calculating the dot product, there is the following line of code:
```
if self.args.is_minus_one:
    adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
```
This resembles the rightmost term in equation 11 in the QPLEX paper<d-cite key="QPLEX"></d-cite>. Therefore, in the code, they are implementing $$Q_{tot}(\mathbf{\tau}, \mathbf{a}) =$$ $$ \sum_{i=1}^n Q_i(\mathbf{\tau}, a_i) +$$ $$ \sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$, In the code, the `calc_v` function actually returns $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$ and the `calc_adv` function returns $$\sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$. However, this is fine because ultimately the sum $$Q_{tot}$$ turns out to be correct, but this is an important detail if you wish to edit the agorithm.



**(c) Summing over the $$V_i$$ Values**

In the `calc_v` function of dmaq_general.py, 
```
v_tot = th.sum(agent_qs, dim=-1)
```

As mentioned in the previous point, this `v_tot` is actually $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$.


**(d) Summing the $$V_{tot}$$ and the $$A_{tot}$$**

In src -> modules -> learners -> dmaq_qatten_learner.py
```
chosen_action_qvals = ans_chosen + ans_adv
```

Here chosen_action_qvals represents the $$Q_{tot}$$, ans_chosen represents the $$V_{tot}$$ (but as mentioned in the previous point, mathematically it is $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$ )and ans_adv represents $$A_{tot}$$ (but as mentioned in the previous point, mathematically it is $$\sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$).



## How to Add a New Algorithm

In order to add a new algorithm, the following steps need to be followed. Note: Depending on the specific algorithm, there might be more steps.

1. YAML file for the algorithm - Create a .yaml file for the algorithm in src -> config -> algs.
2. Controller - If the new algorithm needs a different controller, create a new controller in src -> controllers. 
3. Agent - If the new algorithm needs a different agent network, create a new agent network in src -> modules -> agents. 
4. Mixer - If the new algorithm needs a different mixer, create a new mixer in src -> modules -> mixers. In the case of QPLEX, the mixer encapsulates both the transformation network and the mixing network.
5. Algorithm - If the new algorithm needs a different learner file, create a new learner in src -> learners.
6. Action selecting scheme - If the existing action selecting schemes (e.g., epsilon-greedy) are not sufficient, create a new action selecting scheme in src -> components -> action_selectors.py. 
7. Registering all new components - Remember to register all the new components created (e.g., agent, mixer, controller, etc.). This is an important part to making the algorithm work and also a common place to make an error if you forget to register. 



## Common Pitfalls

1. The SMACv2 environment is procedurally generated, therefore setting the seed is tricky ([github issue](https://github.com/oxwhirl/smacv2/issues/34)).
2. We never really use the joint trajectory $$\mathbf{\tau}$$. It is assumed that adding the information of the state to $$\tau_i$$ gives us a good representation of $$\mathbf{\tau}$$.
3. `batch_size` and `batch_size_run` in the config files refer to two different concepts. `batch_size` represents the training batch size used to sample episodes from the replay buffer, while `batch_size_run` denotes the number of parallel environments configured for algorithm execution. However, they are sometimes used interchangeably in the codebase, e.g., in src -> runners -> parallel_runner.py
```
self.batch_size = self.args.batch_size_run
```
4. All the values in the replay buffer are first initialized with zeros for the corresponding `batch_size_run` (i.e., number of parallel training environments) and timesteps. The values are updated as the episode progresses timestep-by-timestep. 


## Future Work 

If the community finds this blog useful, we would like to add explanations of other algorithms like VDN, Qatten, MAPPO, etc.


## Conclusion

In this blog, we explain the overall structure of the [pymarl2 repository](https://github.com/benellis3/pymarl2) that contains value factorisation algorithms implemented on the benchmark environment [SMACv2](https://github.com/oxwhirl/smacv2/tree/main). We explain the implementation of one of these algorithms (QPLEX) in detail while highlighting the key similarities and differences between the theory and the implementation. We further give detailed instructions on how to add a new algorithm to the repository. 
Recognizing that some terms in the repository may cause confusion, this blog aims to clarify common misunderstandings. While not exhaustive, it addresses several key components that are important for avoiding mistakes in code implementation and ensuring accurate results. For those looking to apply their algorithms in a complex multi-agent environment like SMACv2, this blog serves as a practical starting point.