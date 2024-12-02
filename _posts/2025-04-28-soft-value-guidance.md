---
layout: distill
title: Soft Value Guidance and Sequential Sampling
description: Fine-tuning, controlled generation, and sampling in sequential models has attracted a flurry of recent attention in a variety of settings.   For language modeling in discrete spaces, we would often like to align responses with human preferences or generate correct responses to complex reasoning questions.  For diffusion models, we may be interested in steering generation to output samples which belong to a certain class, images which score highly on metrics such as realism, preferences, or text-to-image consistency, and proteins or molecules with desired properties such as binding affinity or synthesizability.    Diffusion-based samplers have also drawn attention for sampling from arbitrary target probability densities on continuous spaces such as Boltzmann distributions, where we can only assume access to a unnormalized density or energy function.  <br> In this blog post, we provide overview of these sampling or controlled generation tasks from a probabilistic perspective, which incorporates notions from soft reinforcement learning, stochastic optimal control, and Sequential Monte Carlo.  A key role will be played by the soft value function, which yields both importance sampling weights and gradient guidance for diffusion processes.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2025-04-28-soft-value-guidance.bib  


toc:
  - name: Setting & Notation
  - name: Target Distributions
    # subsections:
    # - name: Interactive Figures
  - name: Objective Functions
    subsections:
      -name: Stochastic Optimal Control
      -name: Path Consistency
      -name: Monte Carlo Reward
  - name: Soft Value Functions
  - name: Examples
    subsections:
      -name: Language Models
      -name: Diffusion Models
  - name: Sequential Monte Carlo Sampling

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## Setting & Notation
We consider a pretrained model $p^{\text{ref}}$ such as a language or diffusion model, which we will seek to condition or modulate to achieve some target properties or distribution at the endpoint (see [Targets](#targets)).   We begin by defining sequential models over states $\mathbf{x}_t$ in a discrete or continuous space with $t \in [0,T]$, adapting autoregressive models to have Markovian structure in order to facilitate unified notation later on.


In the language modeling setting, we consider the state $\mathbf{x}_t = \text{concat}(\mathbf{x}_0, x_1, x_2, ... x_{t})$ as the concatenation of output tokens $x_\tau$ generated in response to a prompt or initial state $\mathbf{x}_0$.   We view a reference policy $p^{\text{ref}}_{\text{LM}}(a_t=x_{t+1}|\mathbf{x}_t)$ as selecting a next token $x_{t+1}$ as the action $a_t$ with context state $\mathbf{x}_t$, with deterministic environment transitions $p(\mathbf{x}_{t+1}|a_t = x_{t+1}, \mathbf{x}_t) = \mathbb{I}[\mathbf{x}_{t+1} = \text{concat}(\mathbf{x}_t, x_{t+1})]$ which concatenate the generated token $x_{t+1}$ with the context $(\mathbf{x}_0, x_1, x_2, ... x_{t})$.  The policy is usually given by an autoregressive model $\mathbf{x}_t \sim \prod_{\tau=0}^{t-1} p^{\text{ref}}_{\text{LM}}(x_{\tau+1}|\mathbf{x}_{\tau})$.   For convenience, we will write the full state transition as $p^{\text{ref}}_{t+1}(\mathbf{x}_{t+1}|\mathbf{x}_{t})=p^{\text{ref}}_{\text{LM}}(x_{t+1}|\mathbf{x}_t) \mathbb{I}[\mathbf{x}_{t+1} =\text{concat}(\mathbf{x}_t, x_{t+1})]$.   This leads to an abuse of notation in which we can write the probability of a (partial) sequence as either $p^{\text{ref}}_t(\mathbf{x}_t)=\prod_{\tau=0}^{t-1} p^{\text{ref}}_{\text{LM}}(x_{\tau+1}|\mathbf{x}_{\tau})$ or $p^{\text{ref}}_{t}(\mathbf{x}_{0:t}) = \prod_{\tau=0}^{t-1} p^{\text{ref}}_{\tau+1}(\mathbf{x}_{\tau+1}|\mathbf{x}_{\tau})$.

Our goal will be to construct new transition kernels $q(\mathbf{x}_{t+1}|\mathbf{x}_t)$ or $q_{\text{LM}}(x_{t+1}|\mathbf{x}_t)$ which approximate a [target](#targets) $\pi^*(\mathbf{x}_{t+1}|\mathbf{x}_t)$ or $\pi^*_{\text{LM}}(x_{t+1}|\mathbf{x}_t)$.
<!---= \mathbb{I}[\mathbf{x}_{t+1} = \text{concat}(\mathbf{x}_0,... x_{t-1}, x_t)].$ --->

For diffusion processes, let $\mathbf{x}_t \in \mathbb{R}^d$ represent the current (noisy) state, where $\mathbf{x}_T$ corresponds to clean data.  <d-footnote>We focus on continuous diffusion models here.  While many concepts introduced will be relevant to discrete diffusion guidance, this remains an active area of research.<d-cite key=""></d-cite></d-footnote>
We consider a reference stochastic differential equation with time-dependent drift $b_t^{\text{ref}}$, which may correspond to a physical force or pretrained score-basd diffusion model
$$
P^{\text{ref}}:  \quad d\mathbf{x}_t  =  b_t^{\text{ref}}(\mathbf{x}_t) dt + \sigma_t dW_t \qquad \mathbf{x}_0 \sim p_{0}^{\text{ref}}
$$
We further consider a controlled stochastic differential equation with time-dependent reference drift $b_t^{\text{ref}}$, control drift $u_t$, and diffusion coefficient $\sigma_t$,
$$
Q^{u}:  \quad d\mathbf{x}_t = \left( b_t^{\text{ref}}(\mathbf{x}_t ) + u_t(\mathbf{x}_t ) \right) dt + \sigma_t dW_t \qquad \mathbf{x}_0 \sim p_{0}^{\text{ref}}
$$
We can approximately model these continuous-time stochastic processes using discrete-time Gaussian kernels for small $dt$.   We consider the control drift as an action $a_t = u(\mathbf{x}_t, t)$, with stochastic environment transitions drawn from $p(\mathbf{x}_{t+1}|a_t = u(\mathbf{x}_t,t), \mathbf{x}_t)= \mathcal{N}(\mathbf{x}_{t+1}; \mathbf{x}_t + b_t^{\text{ref}}(\mathbf{x}_t)dt + u_t(\mathbf{x}_t) dt, \sigma_{t} \mathbb{I}_d)$ via Euler discretization.  For convenience,  we combine action selection and state transition into the policy $q_{t+1}^u(\mathbf{x}_{t+1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t+1}; \mathbf{x}_t +  b_t^{\text{ref}}(\mathbf{x}_t)dt + u(\mathbf{x}_t,t) dt, \sigma_{t} \mathbb{I}_d)$.


## Target Distributions
<!---For continuous models, $\mathbf{x}_t \in \mathbb{R}^d$ or in discrete models, $\mathbf{x}_t \in \mathbb{R}^{d\cdot V}$ where $V$ is the size of the vocabulary (or possible outcomes).
--->
Along with the standard problem setting for sampling, we will proceed to view many controlled generation or fine-tuning tasks as sampling from a target probability distribution at the final step $T$, where the target is only known up to a normalization constant.<d-cite key="zhao2024probabilistic"></d-cite>
To ease notation and facilitate interpretations in terms of posterior sampling, we define an observation random variable $\mathbf{y}$ which is emitted as a function of the final state according to $p(\mathbf{y}|\mathbf{x}_T)$.
Our goal is to sample from the posterior distribution over all states,
$$ 
\pi^*(\mathbf{x}_{0:T}|\mathbf{y}) = \frac{1}{\mathcal{Z}(\mathbf{y})} p^{\text{ref}}(\mathbf{x}_{0:T})p(\mathbf{y}|\mathbf{x}_T) \qquad \mathcal{Z}(\mathbf{y}) =\int p^{\text{ref}}(\mathbf{x}_{0:T})p(\mathbf{y}|\mathbf{x}_T) d\mathbf{x}_{0:T}
$$
Via the sequential factorization $\pi^*(\mathbf{x}_{0:T}|\mathbf{y})=\prod_{t=0}^{T-1} \pi^*(\mathbf{x}_{t+1}|\mathbf{x}_{t}, \mathbf{y})$, we will eventually be interested in sampling from the endpoint marginal $\pi^*(\mathbf{x}_{T}|\mathbf{y})$ over full-length language responses $\mathbf{x}_T =\text{concat}(\mathbf{x}_0, x_1, ...x_T)$ or final diffusion states $\mathbf{x}_T \in \mathbb{R}^d$.

<!--- <d-footnote>For discrete models, note that $\mathcal{Z}(\mathbf{y})$ integrates with respect to the counting measure on full sequences $\mathbf{x}_T$.  The joint distribution in this expand
    <d-cite key=""></d-cite></d-footnote>
--->

While a conditioning on a particular class $\mathbf{y}=c$ or noisy observation $\mathbf{y}= \mathcal{A}(\mathbf{x}_T) + \epsilon$ are naturally written using $p(\mathbf{y}|\mathbf{x}_T)$, we can accommodate a more general family of targets. 
<!---  In the table below, we emphasize the endpoint marginal distribution $\pi^*(\mathbf{x}_{T}|\mathbf{y})$ associated with  $\pi^*(\mathbf{x}_{0:T}|\mathbf{y})$ above.--->



| Setting      |   $p(\mathbf{y}\|\mathbf{x}_T)$     | $\pi^*(\mathbf{x}_{T}\|\mathbf{y})$  |
| ------------- |:-------------:| :-----:|
| Constraint |  $\mathbb{I}[\mathbf{x}_T \in \mathcal{B}]$|   $\frac{1}{\mathcal{Z}(\mathcal{B})}p^{\text{ref}}(\mathbf{x}_{T})\mathbb{I}[\mathbf{x}_T \in \mathcal{B}]$  |
| Classifier or Observation Likelihood |  $p(\mathbf{y}\|\mathbf{x}_T)$     | $\frac{1}{\mathcal{Z}(\mathbf{y})} p^{\text{ref}}(\mathbf{x}_{T})p(\mathbf{y}\|\mathbf{x}_T)$  |
| Reward or Energy Modulation |  $\frac{1}{M}\exp\{ \beta~ r(\mathbf{x}_T) \}$    | $\frac{1}{\mathcal{Z}(\beta,r)} p^{\text{ref}}(\mathbf{x}_{T})\exp\{ \beta~ r(\mathbf{x}_T) \}$    |
| Arbitrary Target | $\frac{1}{M}\frac{\tilde{\pi}_T(\mathbf{x}_T)}{p^{\text{ref}}(\mathbf{x}_T)}$ | $\pi_T(\mathbf{x}_{T}) = \frac{1}{\mathcal{Z}} \tilde{\pi}_T(\mathbf{x}_T)$|

<!---
$\frac{1}{\mathcal{Z}(\mathcal{B})}p^{\text{ref}}(\mathbf{x}_{0:T})\mathbb{I}[\mathbf{x}_T \in \mathcal{B}]$
$\frac{1}{\mathcal{Z}(\mathbf{y})} p^{\text{ref}}(\mathbf{x}_{0:T})p(\mathbf{y}\|\mathbf{x}_T)$
$\frac{1}{\mathcal{Z}(\beta,r)} p^{\text{ref}}(\mathbf{x}_{0:T})\exp\{ \beta~ r(\mathbf{x}_T) \}$  
--->

Note that reward or energy modulation can be viewed as a special case of the arbitrary target marginal with $\tilde{\pi}_T(\mathbf{x}_T) =  p^{\text{ref}}(\mathbf{x}_{T})\exp\{ \beta~ r(\mathbf{x}_T) \}$. 
In these cases, the constant $M = \max \limits_{\mathbf{x}_T}\frac{\tilde{\pi}_T(\mathbf{x}_T)}{p^{\text{ref}}(\mathbf{x}_T)}$ corresponds to performing rejection sampling of candidate $\mathbf{x}_T$ via the binary acceptance probability $p(\mathbf{y}=1 |\mathbf{x}_T) = \frac{1}{M}\frac{\tilde{\pi}_T(\mathbf{x}_T)}{p^{\text{ref}}(\mathbf{x}_T)} \leq 1$.   Accepted $\mathbf{x}_T$ can then be shown to be distributed according to $\pi_T(\mathbf{x}_T)$. However, we will see that $M$ does not need to be estimated in practice, since it is absorbed into the normalization constant $\mathcal{Z}$.   This construction is simply used to provide a posterior interpretation of the target endpoint distribution, in similar fashion as the tutorial of Levine 2018 <d-cite key="levine2018reinforcement"></d-cite>. 

We will discuss specific instances in both language and diffusion models in [Examples](#examples) below, but first proceed to introduce common mathematical tools for sequential sampling.


## Soft Value Function
We first characterize the posterior $\pi^*(\mathbf{x}_{0:T}|\mathbf{y})$ and log normalization constant $\log \mathcal{Z}(\mathbb{y})$ via the solution to a variational optimization<d-cite key="knoblauch2022optimization, hartmann2017variational"></d-cite> 
$$\begin{equation}
\log \mathcal{Z}(\mathbf{y}) = \max \limits_{q(\mathbf{x}_{0:T})} ~ \mathbb{E}_{q(\mathbf{x}_{0:T})}\big[ \log p(\mathbf{y}|\mathbf{x}_{T}) \big] - D_{KL}\big[q(\mathbf{x}_{0:T}): p^{\text{ref}}(\mathbf{x}_{0:T})\big] \tag{1}
\end{equation}
$$
where the posterior $q(\mathbf{x}_{0:T}) = \pi^*(\mathbf{x}_{0:T}|\mathbf{y})$ achieves the maximum.  

    
However, a challenge arises from the fact that conditioning or reward information at the terminal state $\mathbf{x}_T$, whereas generation needs to be performed sequentially using $q(\mathbf{x}_{0:T}) = \prod_{\tau=0}^{T-1} q_{t+1}(\mathbf{x}_{t+1}|\mathbf{x}_{t})$. 

The optimal *soft value function* translates the terminal target information to intermediate steps in order to facilitate sampling the exact posterior.   In particular, consider the optimization above starting from a given partial sequence or intermediate state $\mathbf{x}_t$,
$$
\begin{align}
V_\mathbf{y}^*(\mathbf{x}_t) &= \max \limits_{q(\mathbf{x}_{t+1:T}|\mathbf{x}_{t})} ~ \mathbb{E}_{q(\mathbf{x}_{t+1:T}|\mathbf{x}_{t})}\big[ \log p(\mathbf{y}|\mathbf{x}_{T}) \big] - D_{KL}\big[q(\mathbf{x}_{t+1:T}|\mathbf{x}_{t}): p^{\text{ref}}(\mathbf{x}_{t+1:T}|\mathbf{x}_{t})\big] \\
&=\log \int p^{\text{ref}}(\mathbf{x}_{t+1:T}|\mathbf{x}_{t}) p(\mathbf{y}|\mathbf{x}_{T}) d\mathbf{x}_{t+1:T}
\tag{2}
\end{align}
$$
The soft value function measures the expected target likelihood under rollouts from the reference policy, where a rollout may involve generating tokens $x_{t+1:T}$ or running diffusion sampling until time $T$.
The value function $V_{\mathbf{y}}^*(\mathbf{x}_t)= \log \mathcal{Z}_t(\mathbf{x}_t;\mathbf{y})$ is also the normalization constant for $q(\mathbf{x}_{t+1:T}|\mathbf{x}_{t}) = \pi^*(\mathbf{x}_{t+1:T}|\mathbf{x}_{t},y)$, in similar fashion to above.

To see how the soft value informs sequential sampling, we can recognize the one-step-ahead optimal value function $V_\mathbf{y}^*(\mathbf{x}_{t+1})$ as an inner optimization above to write
$$
\begin{align}
V_\mathbf{y}^*(\mathbf{x}_t) = \log \mathcal{Z}_t(\mathbf{x}_t;\mathbf{y})&= \max \limits_{q(\mathbf{x}_{t+1}|\mathbf{x}_{t})} ~ \mathbb{E}_{q(\mathbf{x}_{t+1}|\mathbf{x}_{t})}\big[ V_\mathbf{y}^*(\mathbf{x}_{t+1}) \big] - D_{KL}\big[q(\mathbf{x}_{t+1}|\mathbf{x}_{t}): p^{\text{ref}}(\mathbf{x}_{t+1}|\mathbf{x}_{t})\big]\\
\pi^*_{t+1}(\mathbf{x}_{t+1}|\mathbf{x}_{t},\mathbf{y}) & = p^{\text{ref}}(\mathbf{x}_{t+1}|\mathbf{x}_{t}) \exp\{  V_\mathbf{y}^*(\mathbf{x}_{t+1}) -  V_\mathbf{y}^*(\mathbf{x}_{t}) \}
\end{align}
$$
where $V_\mathbf{y}^*(\mathbf{x}_t) = \log \mathcal{Z}_t(\mathbf{x}_t;\mathbf{y})$ again provides the normalization constant.   Finally, we use the expression for the one-step policy to write the intermediate target marginals using
$$
\pi^*_{t}(\mathbf{x}_{t}|\mathbf{y}) = \frac{1}{\mathcal{Z}(\mathbf{y})} p^{\text{ref}}(\mathbf{x}_{t}) \exp\{  V_\mathbf{y}^*(\mathbf{x}_{t}) \} 
$$
which can be seen by marginalizing either forward $\pi^*_{t}(\mathbf{x}_{t}|\mathbf{y})= \int \prod_{\tau=1}^{t-1} \pi^*_{\tau+1}(\mathbf{x}_{\tau+1}|\mathbf{x}_{\tau},\mathbf{y}) d\mathbf{x}_{0:t-1} = \frac{1}{\mathcal{Z}(\mathbf{y})} \int p^{\text{ref}}(\mathbf{x}_{0:t}) \exp\{  V_\mathbf{y}^*(\mathbf{x}_{t})\} d\mathbf{x}_{0:t-1}$ or backward in time $\pi^*_{t}(\mathbf{x}_{t}|\mathbf{y}) = \frac{1}{\mathcal{Z}(\mathbf{y})} \int p^{\text{ref}}(\mathbf{x}_{t:T}) p(\mathbf{y}|\mathbf{x}_T) d\mathbf{x}_{t+1:T}$.


## Objective Functions
We next discuss several classes of objective functions for learning value functions, taking particular care to highlight special properties of the diffusion SDEs.

### Evidence Lower Bound

When $\log p(\mathbf{y}|\mathbf{x}_{T}) = \beta ~ r(\mathbf{x}_{T}) - \log M$, we obtain a common objective for reinforcement from human feedback in language models <d-cite key="ouyang2022training"><d/cite>

### Path Consistency
### Monte Carlo Reward
    
    


## Examples


### Language Models
### Diffusion Models

        
## Sequential Monte Carlo Sampling

