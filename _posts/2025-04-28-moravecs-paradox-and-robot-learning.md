---
layout: distill
title: "Moravec's Paradox and Robot Learning"
description: "A Perspective on the Current Robot Learning Playbook: Sim-to-Real Transfer, Large-Scale Demonstration Data, and Vision-Language Model"
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: NA

# must be the exact same name as your blogpost
bibliography: 2025-04-28-moravecs-paradox-and-robot-learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Moravec's paradox
  - name: A Line of Attack for Moravec's Paradox?
  - name: Sim-to-real
  - name: The Road to Lifelong Robot Learning
  - name: Real-Time Learning
  - name: Closing Thoughts
  
---

## Moravec's paradox
<p align="justify">
Early Artificial Intelligence (AI) researchers focused on tasks they found challenging, like games and activities requiring reasoning and planning. Unfortunately, they often overlooked the learning abilities observed in animals and one-year-olds. The deliberate process we call reasoning is effective only because its supported by a much more powerful, though usually unconscious, sensorimotor knowledge. We are exceptionally good in perceptual and motor learning, so good that we make the difficult look easy. For example, compared to robots, we might as well be Olympians in walking, running, recognizing objects, etc.  Sensorimotor skills took millions of years to evolve, whereas abstract thinking is a relatively recent development. Keeping this in mind, Moravec wrote in 1988, <i>"it is comparatively easy to make computers exhibit adult level performance on intelligence tests or playing checkers, and difficult or impossible to give them the skills of a one-year-old when it comes to perception and mobility"</i> <d-cite key="moravec1988mind"></d-cite>.
</p>

There has been impressive headway in robotics research in recent years, largely driven by the strides made in machine learning. 
While the realm of AI research is currently heavily dominated by Large Language Model (LLM) researchers, there's still a notable upswing in the enthusiasm for robotics research <d-cite key="spectrum2023solve"></d-cite>. 
In fact, works like Google's [RT-2](https://www.deepmind.com/blog/rt-2-new-model-translates-vision-and-language-into-action)<d-cite key="zitkovich2023rt"></d-cite> tantalizingly dangle the prospect of embodied AGI being just around the corner.
For folks unfamiliar with the term, Artificial General Intelligence (AGI) refers to a hypothetical type of intelligent agent that, if realized, could learn to accomplish any intellectual task that human beings or animals can perform. 

The integration of LLMs with robots is an exciting development mainly because it finally enables us to communicate with robots in a way that was once confined to the realm of science fiction. 
However, the current use of LLMs has been more focused on symbolic planning, requiring additional low-level controllers to handle the sensorimotor data. 
<i>Despite the captivating demonstrations, it's important to note that the foundational issues of Moravec's paradox still persist... </i>

## A Line of Attack for Moravec's Paradox?
In a recent [TED talk](https://youtu.be/LPGGIdxOmWI?si=Wq-C17pjX_LI2lS5), [Prof. Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/) argues that while evolution required millions of years to endow us with locomotion priors/skills, but the development of logic & reasoning abilities occurred more swiftly, driven by the presence of underlying learning capabilities. 
His core argument is that simulation can compensate for millions of years of evolution, allowing us to acquire the necessary priors and [inductive biases](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21483/). 

Simulators can provide a potentially infinite source of data for training robotic systems, which can be cost-prohibitive or impractical to obtain in the real world. 
Additionally, using simulations alleviates safety concerns associated with training and testing on real robots. 
Hence, there's a LOT of interest in learning a control policy purely in simulation and deploying it on a robot. 
This line of research, popularly known as sim-to-real <d-cite key="bousmalis2018using"></d-cite><d-cite key="peng2018sim"></d-cite> in robot learning, refers to the process of transferring a robotic control policy learned in a simulated environment to the real world.

The sim-to-real transfer process typically involves training a robotic control policy in a simulated environment and then adapting it to the real world. 
This adaptation is necessary due to the differences between simulation and reality, such as sensory noise, dynamics, and other environmental factors. 
Typically, there is a significant gap between simulated and real-world environments, which can lead to a degradation in the performance of policies when transferred to real robots. 

## Sim-to-Real: The Silver Bullet? 
The basic idea behind the generative AI revolution is simple: Train a big neural network with a HUGE dataset from the internet, and then use it to do various structured tasks. 
For example, LLMs can answer questions, write code, create poetry, and generate realistic art. 
Despite these capabilities, we're still waiting for robots from science fiction that can do everyday tasks like cleaning, folding laundry, and making breakfast. 

Unfortunately, the successful generative AI approach, which involves big models trained on internet data, doesn't seamlessly scale to robotics. 
Unlike text and images, the internet lacks abundant data for robotic interactions. 
Current state-of-the-art robot learning methods require data grounded in the robot's sensorimotor experience, which needs to be slowly and painstakingly collected by researchers in labs for particular tasks. 
The lack of extensive data prevents robots from performing real-world tasks beyond the lab, such as making breakfast. Impressive results usually stay confined to a single lab, a single robot, and often involve only a few hard-coded behaviors. 

> Drawing inspiration from Moravec's paradox, the success of generative AI, and [Prof. Rich Sutton](http://incompleteideas.net/)'s post on [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)<d-cite key="sutton2019bitter"></d-cite>, the robot learning community's biggest takeaway so far is that we do not have enough data.

While there is some validity to criticisms, like [The Better Lesson](https://rodneybrooks.com/a-better-lesson/)<d-cite key="brooks2019better"></d-cite> by Dr. Rodney Brooks, we can all still agree that we're going to need a lot of data for robot learning. 

<i> The real question is, where does that data come from? </i>
Currently, I see three data sources, and it's worth noting that they do not have to be independent of each other:
1. Large-scale impressive efforts, like Open X-Embodiment <d-cite key="o2023open"></d-cite>, where a substantial group of researchers collaborated to collect robot learning data for public use.
2. Massive open world simulators which can be used for Sim-to-Real Transfer for Robot Learning.
3. Real-Time Learning on Real World Robots <d-cite key="vasan2017teaching"></d-cite><d-cite key="hayes2022online"></d-cite><d-cite key="wang2023real"></d-cite>   
    - This may be painstakingly slow and tedious. But this is also a necessary feature if we envision truly intelligent machines that can learn and adapt on the fly as they interact with the world. 

For points 1 & 3, the downside is the amount of human involvement required. 
It is incredibly challenging to autonomously collect robot learning data, which can indeed pose a significant barrier. 
Now let's talk about why most roboticists are currently paying a lot of attention to sim-to-real learning methods.

### ✅ The Appeal of Sim-to-Real 
1. <b>Infinite data</b>: Simulators offer a potentially infinite source of training data for robots, as acquiring large amounts of real-world robot data is often prohibitively expensive or impractical.
2. <b>The pain of working with hardware</b>: Researchers often favor simulators in their work due to the convenience of avoiding hardware-related challenges. Simulated environments provide a controlled and reproducible setting, eliminating the need to contend with physical hardware issues, allowing researchers to focus more on algorithmic and learning aspects. 
    - I love this sentiment shared by Anonymous: "Robots break your heart. They break down at the worst possible moment - right before a demo or a deadline." 
3. <b>Differentiable physics</b>: There are simulators with differentiable physics. This fits nicely with certain approaches, especially with trajectory optimization in classical robotics. This also helps with estimating gradients of the reward or gradient of a value function with respect to the state in RL.
    - *In the real world, we do not have differentiable physics.*
4. <b>Domain randomization</b> is a crucial step in all sim-to-real approaches. This has more to do with robustness to changes rather than online adaptation. With sim-to-real, we want to expose the agent to as many scenarios as possible in the simulator before deployment. OpenAI's solving a Rubik's cube with a robot hand <d-cite key="akkaya2019solving"></d-cite> demo is a fantastic showcase of this approach. 
    -  *The focus is not really on learning on the fly, but rather being robust to perturbations.*
5. <b>World models</b>: The simulator is a pseudo-reinforcement-learning-model which can help learn a value function better (assuming its a good simulator)
    - *This, however is not the same as learning a world model, rather trying to replicate the world to solve a very specific real-world task*


While sim-to-real has its merits, I believe it may not be sufficient as there are key limitations that still need to be addressed.

### ❌ Limitations of Sim-to-Real 
1. <b>Sim-to-Real Gap</b>: We are limited in our ability to replicate the real world. Our simulators, for example, cannot faithfully replicate friction or contact dynamics. Small errors in simulations can compound and significantly degrade the performance of policies learned in simulation when applied to real robots. 
2. <b>Accuracy of simulators</b>: While techniques such as [domain randomization](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/), and domain adaptation can help mitigate these limitations, they may not be sufficient for tasks requiring detailed simulations, especially in domains like agricultural robotics. 
3. <b>Cost of building simulators</b>: Another concern for me is that no one discusses the computational expenses associated with simulators. High-fidelity simulators can be extremely costly to both develop and maintain. 

To answer the question posed at the start of this section - No, but... 
I believe we can benefit immensely by incorporating simulators into our learning methods when appropriate. The issue is more nuanced than a simple yes or no :)

## The Road to Lifelong Robot Learning
Learning from experience and continual adaptation to changing environments is a hallmark of animal intelligence. Humans and animals have a remarkable ability to continually acquire, fine-tune, and transfer knowledge and skills throughout their lifespan. This ability, referred to as lifelong learning <d-cite key="thrun1995lifelong"></d-cite>, is crucial for computational learning systems and autonomous robots that can solve an open-ended sequence of tasks in an increasingly efficient manner <d-cite key="ring1994continual"></d-cite>. With robots becoming an integral part of our society, they must also continue to learn over their lifetime to adapt to the ever-changing environments.

Typical data-driven learning systems have distinct training and deployment phases. In order to adapt to changes in the real-world post-deployment (e.g., changes in lighting, wear and tear of hardware, and sensor calibration), the entire system is often rebuilt rather than making incremental changes to the model. The majority of commercially deployed robots rely heavily on human supervision and intervention when deployed out in the world.  

General-purpose robots must have the ability to learn on-the-fly as they interact with the physical world, also known as <i>real-time learning</i>. Reinforcement Learning (RL) is a natural way of formulating real-time control learning tasks. Although many deep RL methods, which use artificial neural networks with many layers, have been developed to solve complex motor control problems, they do not easily extend to the real-time learning setting that operates under time and resource constraints <d-cite key="vasan2024deep"></d-cite>, for example, in quadrotors and mobile robots. While approaches including learning from demonstrations, sim-to-real, and offline RL have been used to develop pre-trained agents, there has been relatively little interest in studying real-time learning in the real world.

### The Missing Piece: Real-Time Learning
In the research community, I've noticed the widespread adoption of the term 'Out-of-distribution' (OOD). In robot learning, it denotes the challenge of handling data that deviates from the training data. Personally, I loathe this term. By the very nature of the learning problem, we acknowledge that preparing robots for all conceivable scenarios in advance is impossible. If we could predict every scenario, the need for learning methods would be obsolete! OOD essentially makes a case for integrating the test set into the training set, a notion that was once considered blasphemous in machine learning before the era of LLMs. 

Our current robot learning playbook seems to involve throwing a vast amount of data at our limited, not-so-great algorithms in simulation via domain randomization, with the hope that it encompasses all potential deployment scenarios. In addition, there's also a huge aversion to learning directly on the robot for an extended period of time. In my opinion, this strategy is doomed to fail because we possibly cannot, and will not be able to, model the entire world with a high degree of precision and accuracy.

I do believe that pre-training models using sim-to-real can serve as a good starting point. It can be a fantastic litmus test to rule out ineffective approaches. But clearly, we cannot expose our robots to all possible scenarios via training in simulation. The general-purpose robots we dream of must also have the ability to learn on-the-fly as they interact with the physical world in <i>real-time</i>. I believe we are making significant strides in refining sim-to-real methods. But there has been relatively little interest in studying the real-time adaptation and fine-tuning abilities of these learning systems in the real world. While success in our controlled lab settings is promising, it's imperative that we stress-test our ideas by deploying them in unstructured real-world scenarios. And it might surprise us to discover that there's an entirely new set of challenges awaiting us in real-world deployment, issues that are often overlooked or not encountered within controlled lab settings. 

{% include figure.html path="assets/img/2025-04-28-moravecs-paradox-and-robot-learning/sim-rl.gif" class="img-fluid" %}
<div class="caption">
    In typical simulated learning tasks, the world stands still during computations, and they can be executed sequentially without any consequence in learning performance (as visualized in this GIF). However, the real world moves on during all computations and executing them concurrently can be desirable for minimizing delays.
</div>

<blockquote> “The real world does not pause while the agent computes actions or makes learning updates” </blockquote>

Asynchronous methods hold significant promise for real-time learning, particularly in robotics, where the world does not pause for computations. Unlike traditional synchronous methods, where learning and decision-making processes are tightly coupled, asynchronous methods allow robots to perform multiple tasks — such as action execution, perception, and learning — concurrently and independently. This decoupling minimizes delays, enabling robots to adapt and respond to their environment in real-time while continuously updating their models <d-cite key="mahmood2018benchmarking"></d-cite>. This approach is especially vital for real-world deployments, where efficiency and adaptability are crucial.

## Closing Thoughts

The future of robotics lies at the intersection of sim-to-real learning and real-time adaptation. These two paradigms, rather than competing, can complement each other to bridge the gap between controlled simulation environments and the dynamic, unpredictable nature of the real world. Sim-to-real techniques provide a robust foundation by enabling robots to acquire priors in safe, cost-effective simulated environments, while real-time learning equips them to refine and adapt these priors as they encounter novel situations in the real world. Together, they promise to overcome the enduring challenges posed by Moravec’s paradox, moving us closer to creating robots capable of learning and evolving autonomously.

The road ahead demands collaboration across disciplines, innovations in hardware and software, and a relentless focus on deploying learning systems in realistic scenarios. By combining the scalability of simulation-based approaches with the flexibility of real-world adaptation, we can aspire to build truly intelligent, general-purpose robots that not only understand the world but continually grow with it. As we push the boundaries of robot learning, the question is not whether we will achieve these breakthroughs but how soon—and how profoundly—they will reshape our lives.

