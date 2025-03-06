---
layout: distill
title: Is MAPPO All You Need in Multi-Agent Reinforcement Learning?
description: Multi-agent Proximal Policy Optimization (MAPPO), a very classic multi-agent reinforcement learning algorithm, is generally considered to be the simplest yet most powerful algorithm. MAPPO utilizes global information to enhance the training efficiency of a centralized value function, whereas Independent Proximal Policy Optimization (IPPO) only uses local information to train independent value functions. In this work, we discuss the history and origins of MAPPO and discover a startling fact, MAPPO does not outperform IPPO. IPPO achieves better performance than MAPPO in complex scenarios like the StarCraft Multi-Agent Challenge (SMAC). Furthermore, the global information can also help improve the training of IPPO. In other words, IPPO with global information is all you need.

date: 2025-04-28
future: true
htmlwidgets: true

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
bibliography: 2025-04-28-is-mappo-all-you-need.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Background
    subsections:
    - name: Multi-agent RL
    - name: From PPO to Multi-agent PPO
  - name: Enviroment
  - name: Code-level analysis
  - name: Experiments
  - name: Discussion

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

  .center {
    margin:0 auto;
  }

  .width1 {
    width: 480px;
  }

  .width2 {
    width: 420px;
  }
---

## Background

### Multi-agent RL

Multi-Agent Reinforcement Learning (MARL) is a approach where multiple agents are trained using reinforcement learning algorithms within the same environment. This technique is particularly useful in complex systems such as robot swarm control, autonomous vehicle coordination, and sensor networks, where the agents interact to collectively achieve a common goal <d-cite key="bucsoniu2010multi"></d-cite>.

In the multi-agent scenarios, agents typically have a limited field of view to observe their surroundings. This restricted field of view can pose challenges for agents in accessing global state information, potentially leading to biased policy updates and subpar performance. These multi-agent scenarios are generally modeled as Decentralized Partially Observable Markov Decision Processes (Dec-POMDP) <d-cite key="png2009pomdps"></d-cite>.

Despite the successful adaptation of numerous reinforcement learning algorithms and their variants to cooperative scenarios in the MARL setting, their performance often leaves room for improvement. A significant challenge is the issue of non-stationarity <d-cite key="oliehoek2016concise"></d-cite>. Specifically, the changing policies of other agents during training can render the observation non-stationary from the perspective of any individual agent, significantly hindering the policy optimization of MARL. This has led researchers to explore methods that can utilize global information during training without compromising the agents’ ability to rely solely on their respective observations during execution. The simplicity and effectiveness of the Centralized Training with Decentralized Execution (CTDE) <d-cite key="lowe2017multi"></d-cite> paradigm have garnered considerable attention, leading to the proposal of numerous MARL algorithms based on CTDE, thereby making significant strides in the field of MARL.

###  From PPO to Multi-agent PPO

**Proximal Policy Optimization (PPO)** <d-cite key="schulman2017proximal"></d-cite> is a single-agent policy gradient reinforcement learning algorithm. Its main idea is to constrain the divergence between the updated and old policies when conducting policy updates, in order to ensure not overly large update steps. 

**Independent PPO (IPPO)** <d-cite key="de2020independent"></d-cite> extends PPO to multi-agent settings where each agent independently learns using the single-agent PPO objective. In IPPO, agents do not share any information or use any multi-agent training techniques. 

Each agent $i$:

1. Interacts with the environment and collects its own set of trajectories $\tau_i$
2. Estimates the advantages $$\hat{A}_i$$ and value function $$V_i$$ using only its own experiences
3. Optimizes its parameterized policy $\pi_{\theta_i}$ by minimizing the PPO loss:

$$L^{IPPO}(\theta_i)=\hat{\mathbb{E}}_t[\min(r_t^{\theta_i}\hat{A}_t^i, \textrm{clip}(r_t^{\theta_i},1−\epsilon,1+\epsilon)\hat{A}_t^i)]$$

Where for each agent $i$, at timestep $t$:

$\theta_i$: parameters of the agent $i$

$$r_t^{\theta_i}=\frac{\pi_\theta(a_t^i|o_t^i)}{\pi_{\theta_{\text{old}}}(a_t^i|o_t^i)}$$: 
probability ratio between the new policies $\pi_\theta_i$ and old policies $\pi_{\theta_\text{old}}^i$, $a^i$ is action, $o^i$ is observation of the agent $i$

$$\hat{A}_t^i=r_t + \gamma V^{\theta_i}(o_{t+1}^i) - V^{\theta_i}(o_t^i)$$: estimator of the advantage function
$$V^{\theta_i}(o_t^i) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|o_t^i]$$: value function

$\epsilon$: clipping parameter to constrain the step size of policy updates

This objective function considers both the policy improvement $r_t^{\theta_i}\hat{A}^i_t$ and update magnitude $\textrm{clip}(r_t(\theta), 1−\epsilon, 1+\epsilon)\hat{A}_t$. It encourages the policy to move in the direction that improves rewards, while avoiding excessively large update steps. Therefore, by limiting the distance between the new and old policies, IPPO enables stable and efficient policy updates. This process repeats until convergence.

While simple, this approach means each agent views the environment and other agents as part of the dynamics. This can introduce non-stationarity that harms convergence. So while IPPO successfully extends PPO to multi-agent domains, algorithms like MAPPO tend to achieve better performance by accounting for multi-agent effects during training. Still, IPPO is an easy decentralized baseline for MARL experiments.

**Multi-Agent PPO (MAPPO)** <d-cite key="de2020independent"></d-cite> <d-cite key="yu2022surprising"></d-cite> leverages the concept of centralized training with decentralized execution (CTDE) to extend the Independent PPO (IPPO) algorithm, alleviating non-stationarity in multi-agent environments:

During value function training, MAPPO agents have access to global information about the environment. The shared value function can further improve training stability compared to Independent PPO learners and alleviate non-stationarity, i.e.,

$$\hat{A}_t=r_t + \gamma V^{\phi}(s_{t+1}^i) - V^{\phi}(s_t)$$: estimator of the shared advantage function
$$V^{\phi}(s_t) = \mathbb{E}[r_{t + 1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots|s_t]$$: the shared value function

But during execution, agents only use their own policy, likewise with IPPO.

**MAPPO-FP** <d-cite key="yu2022surprising"></d-cite> found that mixing the features of the agent $i$ and MAPPO's global features $s$ into the value function can improve MAPPO's performance:

$$V_i(s) = V^\phi(\text{concat}(s, f^i))$$

where $f^i$ including the features of agent $i$ or the observation from agent $i$.

**Noisy-MAPPO**: <d-cite key="hu2021policy"></d-cite> To improve the stability of MAPPO in non-stationary environments, Noisy-MAPPO adds Gaussian noise into the input of the the shared value network $V^\phi$:

$$V_i(s) = V^\phi(\text{concat}(s, x^i)), \quad x^i \sim \mathcal{N}(0,\sigma^2I)$$

Then the policy gradient is computed using the noisy advantage values $A^{\pi}_i$ calculated with the noisy value function $V_i(s)$. This noise regularization prevents policies from overfitting to biased estimated gradients, improving stability.

MAPPO is often regarded as the simplest yet most powerful algorithm due to its use of global information to boost the training efficiency of a centralized value function. While IPPO employs local information to train independent value functions.

## Enviroment

We use the StarCraft Multi-Agent Challenge (SMAC) <d-cite key="samvelyan2019starcraft"></d-cite> as our benchmark. SMAC uses the real-time strategy game StarCraft as its environment. In SMAC, each agent controls a unit in the game (e.g. marines, medics, zealots). The agents need to learn to work together as a team to defeat the enemy units, which are controlled by the built-in StarCraft AI, as shown in Figure (a):

<div class="center"> 
{% include figure.html path="assets/img/2025-04-28-is-mappo-all-you-need/smac.jpg" class="img-fluid width1" %}
</div>
<div class="caption">
    (a) The StarCraft Multi-Agent Challenge (SMAC).
</div>

The key aspects of SMAC are:

1. Complex partially observable Markov game environment, with challenges like sparse rewards, imperfect information, micro control, etc.
2. Designed specifically to test multi-agent collaboration and coordination.
3. Maps of different difficulties and complexities to evaluate performance, such as the super hard map`3s5z_vs_3s6z` and the easy map `3m`.

## Code-level Analysis

In order to thoroughly investigate the actual changes from PPO to MAPPO and Noisy-MAPPO, we delved deeply into their differences at the code level with SMAC. Fundamentally, their main difference lies in the modeling of the value function during the training stage.

**Independent PPO (IPPO)**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1144)

For the input of the policy function and value function, IPPO uses the `get_obs_agent` function to obtain the environmental information that each agent can see. The core code here is the `dist < sight_range`, which is used to filter out information that is outside the current agent's field of view, simulating an environment with local observation. `agent_id_feats[agent_id] = 1` is used to set the one-hot agent identification (ID).

{% highlight python %}
def get_obs_agent(self, agent_id):
        ...
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        ...
            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if (dist < sight_range and e_unit.health > 0):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
        ...
            # One-hot agent_id
            agent_id_feats[agent_id] = 1.
            agent_obs = np.concatenate((ally_feats.flatten(),
                                          enemy_feats.flatten(),
                                          move_feats.flatten(),
                                          own_feats.flatten(),
                                          agent_id_feats.flatten()))
        ...
        return agent_obs
{% endhighlight %}

**Multi-Agent PPO (MAPPO)**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1152)

For the input of the value function, MAPPO removes `dist < sight_range` to retain global information of all agents. This means MAPPO only has one shared value function, because there is only one state input into the value function. The input of the policy function in MAPPO is the same as in IPPO.

{% highlight python %}
    def get_state(self, agent_id=-1):
        ...
        ally_state = np.zeros((self.n_agents, nf_al), dtype=np.float32)
        enemy_state = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        move_state = np.zeros((1, nf_mv), dtype=np.float32)

        # Enemy features
        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                enemy_state[e_id, 0] = (e_unit.health / e_unit.health_max)  # health     
        ...
        state = np.append(ally_state.flatten(), enemy_state.flatten())
        ...
        return state
        
{% endhighlight %}

**MAPPO-FP**

[Code permalink](https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/envs/starcraft2/StarCraft2_Env.py#L1327)

For the input of the value function, MAPPO-FP concatenates own_feats (including `agent ID`, `position`, `last action` and others) of the current agent with global information. The input of the policy function in MAPPO-FP is the same as in IPPO.

{% highlight python %}
def get_state_agent(self, agent_id):
        ...
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        ...
            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    if unit.health > 0:
                        enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # available
            ...
            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            ... 
            if self.state_last_action:
                own_feats[ind:] = self.last_action[agent_id]

        state = np.concatenate((ally_feats.flatten(), 
                                enemy_feats.flatten(),
                                move_feats.flatten(),
                                own_feats.flatten()))
        return state
{% endhighlight %}

**Noisy-MAPPO**

[Code permalink](https://github.com/hijkzzz/noisy-mappo/blob/e1f1d97dcb6f1852559e8a95350a0b6226a0f62c/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py#L151)

For the input of the value function, Noisy-MAPPO concatenate a `fixed noise vector` with global information.
**We found in the code that this noise vector does not need to be changed after being well initialized.**
The input of the policy function Noisy-MAPPO is the same as the IPPO.

{% highlight python %}
class R_Critic(nn.Module):
  ...
  def forward(self, cent_obs, rnn_states, masks, noise_vector=None):
        ...
        # global states
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.args.use_value_noise:
            N = self.args.num_agents
            # fixed noise vector for each agent
            noise_vector = check(noise_vector).to(**self.tpdv)
            noise_vector = noise_vector.repeat(cent_obs.shape[0] // N, 1)
            cent_obs = torch.cat((cent_obs, noise_vector), dim=-1)

        critic_features = self.base(cent_obs)
        ...
        values = self.v_out(critic_features)
        return values, rnn_states
        
{% endhighlight %}

Based on code-level analysis, **both MAPPO-FP and Noisy-MAPPO can be viewed as instances of IPPO**, where the `fixed noise vector` in Noisy-MAPPO is equivalent to a Gaussian distributed `agent_id`, while MAPPO-FP is simply IPPO with supplementary global information appended to the input of the value function. Their common characteristic is that each agent has an independent value function with global information, i.e., IPPO with global information. And during the training phase, each agent only uses its own value function to calculate the advantage value and loss, be equivalent to IPPO. We refer to this type of algorithm as Multi-agent IPPO.

## Experiments

We then reproduced some of the experimental results from IPPO, MAPPO, and Noisy-MAPPO using their open-sourced code ([IPPO,MAPPO,MAPPO-FP](https://github.com/zoeyuchao/mappo), [Noisy-MAPPO](https://github.com/hijkzzz/noisy-mappo)),

| Algorithms        | 3s5z_vs_3s6z           | 5m_vs_6m  | corridor | 27m_vs_30m | MMM2 | 
| ------------- |-------------:| -----:| | -------------: |-------------:| -----:|
| MAPPO       |   53% |  26%  |  3% |  95% |   93% |
| IPPO        |  84%  |   88% |  98% |  75% |  87%  |
| MAPPO-FP    |   85% |  89%  |  100% |  94% |   90% |
| Noisy-MAPPO |   87%  |   89% |  100% |  100% |  96%  |

<div class="caption">
    (b) Experimental results for SMAC, the data in the table represents the win rate.
</div>

We also cite the experimental results from these papers themselves here. 


| Map          | corridor | 6h_vs_8z | 3s5z_vs_3s6z | 3m  | 2m_vs_1z |
|--------------|----------|----------|--------------|------|----------|
| IPPO         | 80       | 60       | 90           | 100  | 100      |
| MAPPO        | 0        | 8.95     | 80           | 99.875 | 100    |

<div class="caption">
    (c) IPPO vs MAPPO results for SMAC (from the Figure 2 in <d-cite key="de2020independent"></d-cite>), the data in the table represents the win rate.
    Due to the issues with hyperparameter tuning, the win rates in our results differ slightly from this figure.
</div>

<div class="center"> 
{% include figure.html path="assets/img/2025-04-28-is-mappo-all-you-need/mappo.png" class="img-fluid" %}
</div>
<div class="caption">
    (d) MAPPO-FP (i.e., FP) vs MAPPO (i.e., CL) results for SMAC (from the Figure 16 in <d-cite key="yu2022surprising"></d-cite>).
</div>

<div class="center"> 
{% include figure.html path="assets/img/2025-04-28-is-mappo-all-you-need/noisy.png" class="img-fluid width2" %}
</div>
<div class="caption">
    (e) Noisy-MAPPO (i.e., NV-MAPPO) vs MAPPO results for SMAC (from the Figure 4 in <d-cite key="hu2021policy"></d-cite>).
</div>

From the experimental results, we can see that 
>💡The centralized value function of MAPPO does not provide effective performance improvements. The independent value functions for each agent make the multi-agent learning more robust. 

>💡Introducing global information into the value function improves the learning efficiency of IPPO.

### Discussion

In this blog post, we take a deeper look at the relationship between MAPPO and IPPO from the perspective of code and experiments. Our conclusion is: **IPPO with global information (Multi-agent IPPO) is all you need**. According to the principle of CTDE, the centralized value function in MAPPO should be easier to learn than IPPO and unbiased. Then why is IPPO, better than paradigms like MAPPO, more useful?

We continue to discuss the different implementations of IPPO with global information in MAPPO-FP and Noisy-MAPPO. MAPPO-FP utilizes an agent's own features, including the `agent ID`, `position`, `last action` and others, to form independent value functions. In contrast, Noisy-MAPPO just only uses `agent ID based on Gaussian noise`. Essentially, they both aim to form a distinct set of value functions.

Therefore, there are several reasons for employing independent value functions over a centralized value function:

>💡The independent value functions increase policy diversity and improve exploration capabilities. 

>💡The independent value functions constitute ensemble learning <d-cite key="krawczyk2017ensemble"></d-cite> , making the PPO algorithm more robust in unstable multi-agent environments.

>💡Each agent having its own value function can be seen as an implicit credit assignment <d-cite key="foerster2018counterfactual"></d-cite>.

With this blog post, we aim to broaden awareness for the Multi-agent IPPO, beyond the well-known MAPPO method alone. We believe the Multi-agent IPPO holds untapped potential which deserves exploration. 



