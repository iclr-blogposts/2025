---
layout: distill
title: Where are we getting our environments from?
description: To get the policies that can perform all the interactive tasks we care about we need to train them in the appropriate environments. In this blog post we focus on how we plan on designing these environments to train in. Current methods focus on hand-designing real environment families and leveraging foundation models to, and we propose that we should should also focus on methods that can give us capabilites past the data we already have. We showcase that a simple end-to-end dynamics model can be trained with RL on a simple objective to get structured dynamics and propose more thought on this family of approaches. 
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    affiliations:
      name: Anonymous
  - name: Anonymous
    affiliations:
      name: Anonymous
  - name: Anonymous
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-where-are-we-getting-our-environments-from.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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

<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/html/2025-04-28-where-are-we-getting-our-environments-from/example.gif" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/html/2025-04-28-where-are-we-getting-our-environments-from/example2.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cherry-picked clips of an end-to-end evolved neural dynamics model trained on a simple structural objective. 
</div>

## Introduction

We use the term environment to refer to the general process that the agent interacts with to generate all the data it sees and learns from. To unlock the full potential of what kinds of tasks we can solve we should train our agents interactively. If the policies we train are sponges, this comes down to shaping the experiences that our agents learn from. The community calls this enviornment generation, it has been called Unsupervised Environment Design (UED) and more generally Open-Ended Learning. 

A recent Deepmind position paper <d-cite key="hughes2024openendednessessentialartificialsuperhuman"></d-cite> classifies open-endedness as a learnable process that will surprise the learner ad infimum. However for a not very powerful learner lots of systems could be open-ended. 

## What exactly do we value in environment generation

We have a bunch of realistic environments that are often hand crafted by humans or are found in the real world that we can verify are extremely useful to solve and can determine success for. The downside is that they have inherently sparse rewards making our learning algorithms struggle to solve them, needing to resort to clever reward shaping and autocurricula that we have yet to find general non-task specific solutions to. Current pipelines make “realistic” environments (more often than not sparse reward tasks) by hand and then reward shape to solve them. However since we can't solve them we need to look further. 

We need environments for what they can model in the world, yet they are ony useful to the extent that our agents can solve them. 

In this direction, perhaps it makes sense to more clearly differentiate between what sources really lead to new capabilities being learned in the agentic processes past the prior LLM knowledge. We shouldn't be too surprised that that we can generate the environments that use the knowledge already found in our prior data.

How do we build past that? There are several ways to frame what it means to go about solving this:
1. Generating enviornments 
2. Selecting for enviornments that help out agnts learn better. 

Famous work on the former, like GENIE<d-cite key="bruce2024geniegenerativeinteractiveenvironments"></d-cite>, often leverages internet data to replicate common modalities. But with low dimensional action spaces it is still unclear if they can generate environments that can evoke an endless stream of interesting behaviour. 

It’s very exciting to see the work on leveraging large scale internet data for robotics, actual deployed robots and language  model chatbots giving us more and more samples of the dynamics of our world. Those approaches will keep cooking, but they will take time, traversing costly interaction with the real world, with the data flywheel unavoidable obstacle. Our simulators are unrealistic and in effort to train our models faster we trade off fidelity for speed. We’re craving tractable interactive environments that we can train policies to master and understand — but where are we planning on getting them?

We want to  1) Use real-world data to bootstrap what kinds of environments we can create 2) Tractably simulate them 3) Be able to interpret results .

You’d imagine that any random task/environment you’d design would the VAST majority of the time be either trivial/memorized or way too difficult/noninsighful.

We have so much hand crafted heuristics and ideas that we want to shape our learning processes for modeling the world with: it doesn’t violate the bitter lesson to incorporate those, they should just be incorporated as objectives not as bottlenecks on how they should be found. 

If we have a bunch of ideas on how the world evolves, instead of stressing about designing the dynamics of our simulators to reflect those and suffer the constraints on tractability of simulation or complexity of design from them, what if we just guided our dynamics models to satisfy those properties and get it that way instead?

We have dynamics that if we just run we get evolution and all this kinds of stuff. How do we generate a set of dynamics that has all of that behavior?

Open ended being learnable and novel is great and all but it suffers from a limitation that if the learner isn’t that smart then seemingly boring sources of novelty will let the learner improve forever on challenging tasks for itself, and if the learner is super smart, then any source of novelty that was forged together might quickly expose itself to be finite. 

There are still important work to do to figure out how to exactly use LLM priors in environment generation. LLMs for densifying rewards to solve existing tasks. OMNI, OMNI-EPIC for using LLMs to design new environments and select among them <d-cite key="faldor2024omniepicopenendednessmodelshuman"></d-cite>. 

Need a way to interpret as a human what is technology and how it is being used. Main tools for how to go about this right now:
1. Have objectives the agents/dynamics are evolved to pursue that are interpretable (survival etc.). Potential downsides that we expect this to emerge stranger behaviors
2. Have the ecosystem be factored in a way that allows to interpret both dynamics, agents, life, intelligence, technology. Downsides in that perhaps is less expressive and more complicated. Upsides that it would be much faster to train, potentially tractably represent universe simulation, easier to interpret because you can treat the system equivariantly, you can interpret different element on a variety of different levels (DALIT), you can substitute specific components with others to gain insight on how the ecosystem functions on all levels. Perhaps underrated interp technique
3. Train ways of interpreting elements of the ecosystem as entities. This could be some segmentation on videos, can think of some simple way of identifying something as a a whole on a way that would encourage survival. This ties into thinking about what level heirarchichal life encourages survival and how that comes about.
Rather than serve as an attempt to fully fledge out the theory of artificial life, what exactly qualifies the boundaries between technology, intelligence, life, or just dynamics of our universe

It’s been some time since the startings of artificial intelligence research in the late 20th century, and for reasons of possibly 1) believing that replicating existing data is the true driver of AI progress (which uberfocuses researchers on using existing data sources to bootstrap intelligence) or 2) simulating the real universe is incredibly hard due to all the complexity that exists in it

Should Rich Sutton's bitter lesson confine us to not use our knowledge of the processes that shaped the world to become what it is today?



## Motivation to directly try neural dynamics

Dynamics, Agents, Life, Intelligence, and Technology are all similar in how they are just different transition kernels with the world. Where they differ is implicitly in how hard it is to randomly find them, and perhaps how stable and effective they are in presence of the others.

In some sense life evolved from some set of interaction principles (physics) on basic building blocks (atoms/protons neutrons electrons/quarks) to essentially have reproducing living things. These living things have become regular users of some large scale stable equilibria/processes over these building blocks that we term technologies (e.g. the bonding of atoms into ionic/covalent/metallic compounds, the conversion of water to ice, the concentration differences of ions driving biological processes over membranes, DNA replication, creating fire, humans as an organism, pulleys, levers, Archimedean contraptions, metallurgy, axes, catapults, gunpowder, buildings, steam engines, cars, planes, computers, AI, etc.). Note that it becomes to distinguish between technology, nature, and the lifeforms that interact with them under this categorization.

The nature of intelligence and how life and the universe came to be are two major questions. While we have pretty good theories about how the universe works on a low level (what atoms are created of and what forces govern them) as well as on a high level (we can predict how a good amount of everyday situations evolve, objects staying together and stationary until we move them etc.) there is a big gap in being able to tie these two together through being able to explicitly reconstruct reality.

This gives us nice properties, such as being able to change certain ground rules and seeing how the result evolves, or predict what is going to happen without it having to really happen in the world.

However classical simulation techniques immediately become intractable, rightly so since the world is rather complicated. Even by the stage of having some amount of particles simulate a simple object we are no longer actually using the simulation rules that govern reality, but instead some further abstracted rules we decided are always true. This leads to a repeated cycle of having to conjecture new rules for what is true and what situations they could be true to be able to create new technology, as we are living in that higher layer of abstraction. 

Conventional simulators will continue to develop, trading off speed and fidelity, general accuracy for edge case accuracy, which situations we prioritize modeling first. As they develop, it could be useful to have a separate approach that could have orthogonal insight to repeatedly hitting these walls. 

This motivates the idea of trying to just make the smallest possible simulation paradigm/environment that can evolve technology and technology use. Finite examples are okay to start, but really building towards the point where more scale leads to significantly more and longer of a never-ending evolutionary path. 

We should solve them simultaneously. Instead of confining ourselves to a setting and then getting roadblocked by a certain abstraction of evolution or simulation being intractable, let us codesign the entire system from dynamics, to emergent physics, to agents, to intelligence all at once. What are the important framings and constraints to allow it to emerge?


## Trying out neural dynamics 

To concretize this we try to actually create a neural dynamics model to see whether we can get interesting behaviours, and it turns out that we can get an end-to-end trained dynamics model to exhibit structured behaviours just with multi-agent RL training towards a goal of reproducing current and future structures in the grid. 

### Parameterization and Model

We choose our dynamic model to function as a memoryless stochastic transition function between pixel grids colored in 10 colors (similar to AutoVerse <d-cite key="earle2024autoverseevolvablegamelanguage"></d-cite>). This offers a simple parametrization of an environment to generate neural dynamics in that may be more likely to be interpretable without much extra effort due to being inherently visual <d-cite key="earle2023pathfindingneuralcellularautomata"></d-cite>. 

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/2025-04-28-where-are-we-getting-our-environments-from/grid-graphic-cropped.png" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">
    Our dynamics model is a neural network local update rule applied to every neighborhood in the environment state.
</div>

In effort to choose the simplest model that can still reasonably be expected to express the stochastic transitions that could function as agents, we train a tiny transformer $$f_\theta$$ to take in a 3x3 grid of colors and stochastically output a 3x3 grid of colors for every single 3x3 in state $$S_t$$, and then take the output at any given pixel for our next state $$S_{t+1}$$ to be an aggregation of all the outputted pixels at that position with a linear layer $$g_\psi$$. 

### Objective and Training

We want to get our dynamics to exhibit some structure without us having to specify what it exactly should be. In the absence of any data (ways to incorporate data into an objective might include rewards for direct replication of it <d-cite key="mordvintsev2020growing"></d-cite> ), RL fits as a solution to train our differentiable dynamics model to output transitions that were more likely. We set a simple objective of maximizing the total number of 2x2 cells that shift by one tile in some direction while keeping their color over an update. 

We treated every single call to $$f_\theta$$ as an agent taking an action, and ran independent PPO on this shared hand-designed objective. We used a discount rate of 0.99, with regular advantage estimation, updating our dynamics model every 50 steps. 

### Results

It turns out that a fully neural dynamics model can emerge some interesting somewhat interpretable dynamics, which means that with extra care there may be more results we can get in this direction. 








