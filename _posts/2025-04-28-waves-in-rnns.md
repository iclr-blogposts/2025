---
layout: distill
title: Waves in RNNs
description: "Recurrent Neural Networks (RNNs) have long been the go-to models for sequence processing in AI. Over the years, these models have evolved significantly, culminating in modern architectures like Transformers and State Space Models (SSMs) that employ specialized structured connectivity patterns for faster training and inference. In this blog post, we revisit the core principles of RNN architectures through the lens of the traveling wave theory of memory storage. The theory shows that, a wide range of RNN-based models, including recent innovations, share a fundamental architectural component: a shift operation for temporal memory storage. This shift matrix plays a crucial role in enabling reliable storage of recent sequence history, mitigating the diminishing gradient problem, and facilitating optimizations for efficient computation. By using the new perspective obtained from the traveling wave theory, this blogpost aims to show a unified framework for understanding the different RNN models and offer insights into how they can be interpreted, refined and extended."
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
bibliography: 2025-04-28-waves-in-rnns.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: "Background: Evolution of RNNs"
    subsections:
    - name: Elman RNNs
    - name: Transformers
    - name: State Space Models
  - name: Traveling Wave Theory
    subsections: 
    - name: The Traveling Wave Model of Memory Storage
    - name: Gradient Propagation with Traveling Waves
  - name: Traveling Waves Across RNNs
    subsections:
       - name: Elman RNNs
       - name: Transformers
       - name: SSMs
  - name: Future of Waves in RNNs

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


# Introduction

Sequence modeling lies at the heart of many breakthroughs in artificial intelligence, powering applications from natural language processing to time-series analysis and speech recognition <d-cite key="Brown2020LanguageMA"></d-cite>. For years, Recurrent Neural Networks (RNNs) were the go-to architectures for such tasks. However, their initial designs faced significant challenges, such as difficulty in capturing long-term dependencies and computational inefficiencies during training and inference. This led to a wave of innovations, culminating in modern architectures like Transformers <d-cite key="Vaswani2017AttentionIA"></d-cite> and State Space Models (SSMs) <d-cite key="Gu2021EfficientlyML"></d-cite>, which have largely supplanted traditional RNNs in state-of-the-art applications.

Despite this shift, the principles underlying RNNs remain central to sequence modeling. In fact, we find that many of the recent innovations in models like SSMs share an intriguing connection to classic RNN memory mechanisms. At the heart of this connection lies a deceptively simple yet powerful component: the shift operator, which acts as a mechanism for temporal memory storage. This architectural motif enables models to propagate information efficiently, allowing them to store recent sequence history, mitigate the infamous diminishing gradient problem, and achieve computational efficiency.

In this blog post, we revisit RNNs and their modern counterparts through the lens of a _traveling wave theory_ of RNN memory <d-cite key="pmlr-v235-karuvally24a,Keller2023TravelingWE"></d-cite>, a framework for understanding how information is stored and propagated in sequence models. We show how the shift operation inspired by the theory, has become a conserved feature across many state-of-the-art models, from early RNNs to modern transformers and SSMs.

Our exploration serves two purposes: first, to provide a fresh perspective on the design and evolution of sequence models, and second, to offer actionable insights into how these architectures can be further refined. By bridging foundational theory with practical advancements, we aim to shed light on the enduring relevance of RNN memory principles in shaping the future of AI. 

# Background: Evolution of RNNs

In this section, we review some of the most prominent ideas for creating RNNs over the years. The field of sequence modeling has seen a remarkable evolution over the years. Early successes with Recurrent Neural Networks (RNNs) like Elman RNNs <d-cite key="Elman1990FindingSI"></d-cite> demonstrated their ability to model sequential data by maintaining a hidden state that evolves over time. However, these initial designs struggled with key challenges, such as vanishing gradients <d-cite key="Arjovsky2015UnitaryER"></d-cite>, difficulty in capturing long-range dependencies, and computational inefficiencies. These limitations spurred the development of more advanced architectures, such as Long Short-Term Memory networks (LSTMs) <d-cite key="Hochreiter1997LongSM"></d-cite> and Gated Recurrent Units (GRUs) <d-cite key="Cho2014LearningPR"></d-cite>, which introduced gating mechanisms to address these issues. More recently, the rise of Transformers and State Space Models (SSMs) has pushed sequence modeling to new heights, offering specialized structures for fast training and inference <d-cite key="pmlr-v202-orvieto23a"></d-cite>. Yet, despite these advancements, many of these architectures retain core principles rooted in RNNs, particularly when it comes to memory storage and temporal processing. Understanding this shared lineage provides a unique opportunity to unify our view of sequence modeling architectures and explore new directions for innovation.

We add a disclaimer that this is not a full review of various RNN architectures over the years, the section is 
merely broad strokes on what is a complex and beautiful collection of models. We omit some major ones like unitary 
RNNs and gated architectures. 

## Elman RNNs

Elman RNNs were a groundbreaking development because they demonstrated how recurrent connections could enable a neural network to model sequential data and maintain a form of memory over time. They consist of a single hidden layer with recurrent connections, allowing information from the previous time step to influence the current hidden state.

The architecture of an Elman RNN can be summarized as:

$$  
\begin{cases}  
h_t = \sigma(W_{hh} \, h_{t-1} + W_{xh} \, x_t + b) \\  
y_t = W_{hy} \, h_t
\end{cases}  
$$

where $h_t$ is the hidden state at time $t \geq 0$ with $h_{-1}=0$, $x_t$ is the input, $y_t$ is the output,  
$W_{xh}$, $W_{hh}$, and $W_{hy}$ are the parameters of the network. $\sigma(\cdot)$ is a non-linear activation function.
A good exercise to gain intuition for how the system works is to expand it out for a couple of timesteps, for conciseness we
set $b=0$.

$$h_1 = \sigma(W_{xh} x_1)$$

$$h_2 = \sigma( W_{hh} \sigma(W_{xh} x_1) + W_{xh} x_2 )$$

$$h_3 = \sigma( W_{hh} \sigma( W_{hh} \sigma(W_{xh} x_1) + W_{xh} x_2 ) + W_{xh} x_2 )$$

$$\vdots$$

Each step of the recursion adds a layer of non-linearly to the existing hidden state and all the RNN 
parameters are separated by the non-linearity. This means that simulating the Elman RNN 
requires recusively processing the input - from first to last one step at a time. 

Despite the sequential processing limitations, Elman RNNs were used to model a range of tasks involving sequential data, including simple language models and time-series predictions. Their success, however, was limited by the **vanishing gradient problem** (discussed in the Gradient Propagation section below), which made learning long-term dependencies difficult.  Long Short-Term Memory networks (LSTMs) and Gated Recurrent Units (GRUs) were introduced to address the limitations of Elman RNNs and achieved better performance in more complex tasks. These advancements cemented RNNs as a powerful tool for sequence modeling until architectures like Transformers emerged. 
We note that gated architectures are also a very interesting class of models which we do not cover in the post as we 
want to focus on the common architectural motifs in non-gated architectures. 

## Transformers

Transformers, introduced in a seminal 2017 paper *"Attention is All You Need"* <d-cite key="Vaswani2017AttentionIA"></d-cite>, revolutionized sequence modeling by addressing the limitations of earlier architectures like RNNs and LSTMs. Unlike recurrent architectures, which process sequences step-by-step, Transformers process entire sequences in parallel using the _self-attention mechanism_, enabling them to model long-range dependencies more efficiently.

The key innovation in Transformers was the introduction of the _self-attention mechanism_, which computes relationships between all elements in a sequence simultaneously. This allowed Transformers to focus on relevant parts of the input regardless of their position, a significant departure from the sequential RNNs. Since self attention is invariant to the position of the tokens, transformers also use positional encodingss. These encodings, added to the input embeddings, allow the model to differentiate between positions in the sequence.  The final autoregressive loop of the transformer has the following form

$$
h_{t} = \sum_{i=1}^{s} \frac{\exp( h_{t-i}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})}{\sum_{k} \exp( h_{t-k}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})} \mathbf{V} h_{t-i} \\
$$

where:
- $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are query, key, and value matrices derived from the input,
- $s$ defines the size of the context vector

Note that we have rewritten the transformer dynamics in a lag form (where $-i$ indicates the lag from the curren time) to clearly show how the history of the tokens stored in the context vector results in the next token element. The relation between this formulation and traveling waves will be easily evident in this notation when we discuss traveling waves in the next sections.

Transformers quickly became the state-of-the-art for a variety of tasks, including machine translation, natural language understanding, and image processing. Their success has spawned numerous adaptations and extensions, such as BERT (Bidirectional Encoder Representations from Transformers) <d-cite key="Devlin2019BERTPO"></d-cite> for understanding contextual relationships in text, GPT (Generative Pre-trained Transformer) <d-cite key="Radford2018ImprovingLU"></d-cite> for autoregressive generation tasks, Vision Transformers (ViTs) <d-cite key="Dosovitskiy2020AnII"></d-cite> adapted for image processing tasks.

Despite their success, Transformers were not without limitations. Their reliance on the self attention operation meant that the inner product had to be computed which has quadratic complexity <d-cite key="keles2022computationalcomplexityselfattention"></d-cite> limiting their ability to be scaled to very long sequences. This limitation prompted research into linear attention mechanisms and alternative architectures like State Space Models (SSMs).

## State Space Models

State Space Models are most commonly introduced as a continuous time dynamical system following the differential equations:

$$  
\begin{aligned}  
\mathbf{\dot{x}}(t) & = \mathbf{A}\mathbf{x}(t) + \mathbf{B}u(t), \\  
y(t) & = \mathbf{C}\mathbf{x}(t) + \mathbf{D}u(t),  
\end{aligned}  
$$

where $\mathbf{x}(t) \in \mathbb{R}^N$ is the state vector, $u(t) \in \mathbb{R}^1$ is the time-dependant input signal, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the recurrent connectivity matrix, $\mathbf{B} \in \mathbb{R}^{N \times 1}$ maps from the input to the hidden state, $\mathbb{C} \in \mathbb{R}^{1 \times N}$ maps from the hidden state to the target signal $y(t) \in \mathbb{R}^1$. Most work asserts that $\mathbf{D} \in \mathbb{R}^{1 \times 1}$ is set to $0$ (and we will also follow this convention in this post), but it can equivalently be viewed as a skip-conection.

These continuous-time equations are then often discretized with a specific timestep size $\Delta$, through a Zero-Order-Hold method (ZOH) - an analog to digital converstion approach <d-cite key="Moir_2022"></d-cite>. ZOH approximates the input signal $u(t)$ as constant between timesteps $t$ and $t + \Delta$. This yields the following

$$  
\begin{aligned}  
\mathbf{x}_{k} & = \mathbf{\bar{A}}\mathbf{x}_{k-1} + \mathbf{\bar{B}}u_k, \\  
y_k & = \mathbf{\bar{C}}\mathbf{x}_k + \mathbf{\bar{D}} u_k,  
\end{aligned}  
$$

where $\mathbf{x}_{-1} = \mathbf{0}$, $\mathbf{\bar{A}} = \mathrm{exp}(\Delta \mathbf{A})$, $\mathbf{\bar{B}} = (\mathrm{exp}(\Delta \mathbf{A}) - I) (\Delta \mathbf{A})^{-1} \Delta \mathbf{B}$, and $\mathbf{C}$ & $\mathbf{D}$ remain unchanged (i.e., $\bar{\mathbf{C}} = \mathbf{C}$, $\bar{\mathbf{D}} = \mathbf{D}$). We want to draw caution here that the $\exp(.)$ operations here are the 
matrix exponentials and not the elementwise exponentials of $A$. This yields exact numerical integration of the continuous time equations, assuming the approximation of the input signal is correct <d-cite key="Jacquot_2019"></d-cite>.

The above equations can be seen to represent a Linear Time Invariant (LTI) system <d-cite key="Hespanha2009LinearST"></d-cite>, and they have a long history of study in the control theory literature. The useful property which enabled improvements in training and inference is the structure of the recursion. Similar to Elman RNNs, we can expand out the recurrence to reveal

$$y_1 = \mathbf{\bar{C}} \mathbf{\bar{B}} u_1,$$

$$y_2 = \mathbf{\bar{C}} \mathbf{\bar{A}}  \mathbf{\bar{B}} u_1 + \mathbf{\bar{C}} \mathbf{\bar{B}} u_2,$$

$$y_3 = \mathbf{\bar{C}} \mathbf{\bar{A}}^2 \mathbf{\bar{B}} u_1 + \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}} u_2 + \mathbf{\bar{C}} \mathbf{\bar{B}} u_3,$$

$$\vdots \, .$$

Here, unlike the Elman RNN, the input sequence is independent of each other and even be processed in parallel if the 
collection of matrices $(\mathbf{\bar{C}} \mathbf{\bar{A}}^k \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}^{k-1} \mathbf{\bar{B}} , ..., \mathbf{\bar{C}} \mathbf{\bar{B}})$ is precomputed. Early work in the SSM community leveraged this linearity of the operations to efficiently accelerate computation by effectively pre-computing the associated operators for each timestep, converting the full recurrence into a single global convolution over the sequence length <d-cite key="Gu2021EfficientlyML,Gu2022HowTT"></d-cite>. The current state-of-the-art optimization for computing the convolution in parallel is the parallel scan algorithm <d-cite key="Gu2023MambaLS"></d-cite> which has a complexity logarithmic in sequence length when sufficient
parallelism is involved. To obtain the power of the parallel scan algorithm, it meant that the convolution kernel $(\mathbf{\bar{C}} \mathbf{\bar{A}}^k \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}^{k-1} \mathbf{\bar{B}} , ..., \mathbf{\bar{C}} \mathbf{\bar{B}})$ is precomputed. Doing this was non-trivial for a general class of $A$ matrices as computing the power requires
either sequential multiplication or computing the singular value decomposition - both involving significant complexity.
Thus, the focus turned to choosing structured $A$ matrices that admitted compute efficient power and matrix multiplications.
The most popular choice for $A$ became purely diagonal which meant the matrix exponential is the elementwise exponential
and computing powers was a low complexity operation. A notable change in this idea came with the introduction of Shift-SSMs.

**Shift-SSM** In Dao 2022<d-cite key="Dao2022HungryHH"></d-cite>, the authors aim to tackle the relative underperformance of prior state space models (such as S4) on language modeling tasks in particular. Through synthetic experiments (two variants of simple associative recall), they note that SSMs struggle with both recall of past tokens, and subsequently comparison across tokens.  To address these limitations  they propose to build a repeatable 'block' (called the H3 Layer) which incorporates two new components and is loosely inspired by Linear Attention <d-cite key="Katharopoulos2020TransformersAR"></d-cite>. The first component is a mechanism to remember tokens from the past in a manner that is ammenable to subsequent cross-comparisons, and the second component is a multiplicative interaction between the input and the hidden state to implement such cross-comparisons. A picture of the proposed H3 Block is reproduced below:

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/H3_Block.png" class="d-block mx-auto img-fluid w-25"%}

How then might one imagine implementing a mechanism to explicitly store the recent past in a recurrent architecture? One of the simplest mechanisms could be thought of as a fixed-size queue, where inputs $u_k$ are written onto the queue (enqueued) at each timestep, and once the queue is full, the oldest items 'fall off' (dequeued) making space for new items. In the H3 block, the authors propose to implement such a fixed-size queue explicitly within the hidden state of an SSM through a structured matrix called the Shift-SSM. Explicitly, this is given as:

$$  
\begin{aligned}  
\mathbf{x}_{k} & = \mathbf{\bar{A}_{shift}}\mathbf{x}_{k-1} + \mathbf{\bar{B}_{shift}}u_k, \\  
y_k & = \mathbf{\bar{C}}\mathbf{x}_k,  
\end{aligned}  
$$

exactly identical to a standard SSM, but crucially with the recurrent connectivity matrix ($\mathbf{\bar{A}}$) set to be a shift operator, and the input projection ($\mathbf{\bar{B}_{shift}} = e_1$) set to be the first canonical basis vector:

$$  
\begin{aligned}  
\mathbf{\bar{A}_{shift}} =  
\begin{bmatrix}  
0 & 0 & 0 & \cdots & 0 & 0 \\  
1 & 0 & 0 & \cdots & 0 & 0 \\  
0 & 1 & 0 & \cdots & 0 & 0 \\  
0 & 0 & 1 & \cdots & 0 & 0 \\  
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\  
0 & 0 & 0 & \cdots & 1 & 0  
\end{bmatrix},  
\quad  
\mathbf{\bar{B}_{shift}} =  
\begin{bmatrix}  
1 \\  
0 \\  
0 \\  
0 \\  
\vdots \\  
0  
\end{bmatrix}.  
\end{aligned}  
$$

**Example**

If we consider the sequence $u_0, u_1, \ldots u_T$, given an initial hidden state of $\mathbf{x}_0 = \mathbf{0}$, we see that running the equations forward yields a hidden state exactly as desired:

$$  
\mathbf{x}_0 =  
\begin{bmatrix}  
0 \\ 0 \\ 0 \\ \vdots \\ 0  
\end{bmatrix}, \quad  
\mathbf{x}_1 =  
\begin{bmatrix}  
u_0 \\ 0 \\ 0 \\ \vdots \\ 0  
\end{bmatrix}, \quad  
\mathbf{x}_2 =  
\begin{bmatrix}  
u_1 \\ u_0 \\ 0 \\ \vdots \\ 0  
\end{bmatrix}, \quad  
\cdots \quad  
% \]  
% \[  
% \mathbf{x}_N =  
% \begin{bmatrix}  
% u_{N-1} \\ u_{N-2} \\ \vdots \\ u_0  
% \end{bmatrix},\quad  
\mathbf{x}_{N+1} =  
\begin{bmatrix}  
u_N \\ u_{N-1} \\ \vdots \\ u_1  
\end{bmatrix}, \quad  
\cdots \quad  
\mathbf{x}_{T} =  
\begin{bmatrix}  
u_{T-1} \\ u_{T-2} \\ \vdots \\ u_{T-N}  
\end{bmatrix}.  
$$

### Why unify?

Transformers and SSMs aimed to improve on the base RNNs and ended up with the same operation - a shift along
the sequence dimension that stores the most recent collection of tokens.

While the move away from traditional RNNs has addressed key challenges like vanishing gradients and slow sequential training, it has also led to a fragmented understanding of memory mechanisms in RNNs. RNNs introduced the foundational concept of maintaining memory through hidden state dynamics, but their reliance on sequential processing imposed significant bottlenecks. Transformers, with their self-attention mechanism, bypassed these bottlenecks by processing sequences in parallel, but at the cost of losing a clear temporal structure. Similarly, SSMs have reintroduced elements of recurrence and time-aware modeling, yet they often frame these mechanisms through a lens far removed from traditional RNN theory.

This is where the traveling wave theory offers a unique perspective. By examining how information propagates through RNNs through propagating neural activity, we can uncover a unifying principle that ties together these seemingly diverse architectures. Traveling wave theory reveals that many modern models implicitly rely on principles originally present in RNNs, such as the use of structured state transitions for temporal memory. For instance, the shift matrix, a core component in many State Space Models, can be understood as a structured implementation of traveling wave dynamics, enabling reliable memory propagation and alleviating issues like vanishing gradients. In transformers, computing the context also results in the shift operation.

Revisiting RNN memory mechanisms through this lens serves two purposes. (1) It highlights the hidden connections between seemingly disparate architectures, offering a coherent framework to understand their similarities and differences. (2) It opens up new opportunities for innovation by reframing modern designs as extensions of a common foundation. For example, understanding how traveling waves encode memory could inspire more efficient parameterizations or hybrid models that combine the strengths of RNNs, Transformers, and SSMs.

In this context, unification is not about reverting to old designs but about integrating their best ideas into a broader framework. By bridging the gap between classical RNN principles and modern innovations, we may be able to chart a course for the next generation of sequence models—models that are not only powerful but also _principled_ and _mathematically sound_ in their design.

# Traveling Wave Theory

The notion that information may be stored as propagating neural activity comes from biology. Traveling waves are ubiquitous in neurobiological experiments of human memory <d-cite key="Davis2020SpontaneousTW"></d-cite> where they have been observed during awake and sleep states throughout the brain, including the cortex and hippocampus and have been shown to impact behavior <d-cite key="Davis2020SpontaneousTC"></d-cite>. There are several prominent hypotheses based on experimental evidence which suggest the utility of these waves as memory stores <d-cite key="Benigno2023WavesTO"></d-cite>. One of these hypotheses suggests external stimuli interacts with neural networks to form neural activity that propagates through the brain <d-cite key="Muller2018CorticalTW,Perrard2016WaveBasedTM"></d-cite>. A snapshot of this wave field in the brain provide the information necessary to reconstruct the recent past, a mechanism ideal for the memory storage.

<script type="application/ld+json">{"@context":"https://schema.org","@type":"VideoObject","name":"elife-17267-media1","description":"","thumbnailUrl":[["https://video.gumlet.io/67417280f41b76923ec141b2/674172b5c0118cccd82c4559/thumbnail-1-0.png?v=1732342466327"]],"uploadDate":"2024-11-23T06:14:13.897Z","duration":"PT20.125S","embedUrl":"https://play.gumlet.io/embed/674172b5c0118cccd82c4559"}</script><div style="position:relative;aspect-ratio:100/100;align-content: center;">

      <iframe 

            loading="lazy" title="Gumlet video player"

            src="https://play.gumlet.io/embed/674172b5c0118cccd82c4559?preload=true&autoplay=true&loop=true&disable_player_controls=false"

            style="border:none; position: absolute; top: 0; left: 0; height: 50%; width: 50%;"

          allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture; fullscreen;">

          </iframe>

</div>

### The Traveling Wave Model of Memory Storage

Improving the memory storage capabilities of RNNs has been a longstanding problem. Traveling waves represent a dynamic approach to memory storage, where information is not confined to fixed locations within the network's state but instead propagates as neural activity over time. This phenomenon is inspired by biological neural systems, where due to the spatial locality of brain connectivity, patterns of neural activity propagate to encode and transmit information across large populations of neurons. In RNNs, traveling waves _emerge_ as a mechanism to encode temporal sequences and maintain information over long time horizons by spreading task relevant information across the hidden state space. Unlike static memory mechanisms, which rely on fixed attractors or localized states, traveling waves enable distributed and dynamic forms of memory, providing adaptability to varying input patterns. There is an emerging trend that aims to utilize the traveling wave principle of memory storage to improve memory storage in RNNs.

_What mechanisms existing (trained but otherwise unrestricted) Recurrent Neural Networks use for storing history_? Since this is a broad question with many possible answers depending on what the definition of the RNN and the task it is trained to do, we focus on a restricted class of problems where the dependence of recent history is well-defined for the solution. The canonical problem of this class studied in the machine learning literature is the Vector AutoRegressive Moving-Average model (VARMA) <d-cite key="Hansen1995TIMESA"></d-cite>. A general $\text{VARMA}(s)$ process that utilizes a history of size $p$ to produce the next elements in the time seies which and is recursively defined as

$$  
\begin{aligned}  
X_t = \sum_{i=1}^s \phi_i X_{t-i} + \epsilon_t \, .  
\end{aligned}  
$$

Here the variables are $X_t \in \mathbb{R^d}$ is the sequence of vectors describing the state space of the problem, $\phi_i \in \mathbb{R}^d \to \mathbb{R}^d$ is a linear operator that maps the vector at some point in history to the next vector in sequence, and $\epsilon_t$ is uncorrelated Gaussian noise. For some intuition into what the VARMA process represents, imagine ChatGPT producing the answer to a prompt given to it. Here, the sequence of words in the answer is the vector sequence $X_t$ and the process by which the context of the prompt and the previous words of the answer is pulled in to generate the next element in the sequence is the operator $\phi$. The VARMA process has the unique property that the operator $\phi$ is constant for a given lag value of $i$ which provides a nice invariant property for solving the process using RNNs. Other more concrete examples are the Fibonacci series and the Copy problem. In the Fibonacci sequence generation the operator $\phi_{1}= \phi_{2}=1$ and all the other $\phi_i$ zeros. In the Copy problem that is used as a toy problem in analyzing RNNs, an element some $s$ timesteps back is the next sequence element.

Modeling a $\text{VARMA}(s)$ process in an RNN requires efficient storage of the past $s$ states in a manner that is accessible to compute future states much like the requirements for the Shift SSMs.  Further, the storage mechanism should also enable updating these past states as needed.  We can use the traveling wave principle to formulate a memory model where each dimension of the state history is represented by a 1-D traveling wave of activity in a neural substrate. The wave propagation in the brain is assumed to travel in independent neural strings. We would like to add that this is not a realistic description of the brain but it contains the core features of wave propagation in an analytically tractable manner. 

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/brain2wave.png" class="d-block mx-auto img-fluid w-50"%}

To mathematically describe the wave propagation, we define the two aspects of wave propagation - (1) Wave components and how they interact as the waves travel in the substrate (2) Conditions for what happens to the waves at the boundaries of the substrate.  For the first component, we consider each dimension of the state as an independent wave that does not interact with the other waves as it travels in the substrate.  The interactions are only at the start boundary, which applies the VARMA function $\sum_{i=1}^s \phi_i X_{t-i}$ generating new states to propagate. The end boundary is left open so that the waves do not reflect and interfere.

For the process, we end up with the following equations for the traveling waves

$$  
\begin{cases}  
h_{i, j}(t+1) = h_{i-1, j}(t) & 1 \leq i < s \, ,\\  
h_{s, j}(t+1) = f(h_{s}(t), ..., h_{1}(t))_{j} & \text{otherwise} \, ,  
\end{cases}  
$$

The equations describe waves traveling in the following neural lattice

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/twwm.png" class="d-block mx-auto img-fluid w-50"%}

### Gradient Propagation with Traveling Waves

A consistent problem in RNNs is the diminishing/vanishing gradient problem, which has been the subject of intense investigations and model improvements <d-cite key="Hochreiter2001GradientFI,Pascanu2012OnTD"></d-cite>.  The main argument for the gradient problem is that the repeated application of the RNN weight matrix to the gradient results in the gradient magnitude diminishing or exploding the farther back it is propagated <d-cite key="Arjovsky2015UnitaryER"></d-cite>.  Traveling waves provide an alternate solution to the problem by representing the state history spatially in the RNN hidden state without resorting to advanced architectures.  This representation thus removes the requirement for the gradient to be propagated backward in time, entirely eliminating the diminishing gradient problem.

To figure out the effect of traveling waves in gradient propagation, we consider two cases of RNNs - one that does not
have the traveling wave memory vs one that have traveling wave mechanisms.

**Without Traveling Waves**

Let's consider a random initialization of the RNN devoid of any wave phenomenon.  Let $C(y(s+1), x(s+1))$ be the cost function computed at time $s+1$.  We will now analyze the influence of the input at time - $x(1)$ on the cost function evaluated at time $s+1$.

$$
\frac{\partial C}{\partial x(1)} = \frac{\partial C}{\partial h(s+1)} \left( \prod_{k=1}^{s} \frac{\partial h(k+1)}{\partial h(k)} \right) W_{xh} $$

$$
\frac{\partial C}{\partial x(1)} = \frac{\partial C}{\partial h(s+1)} \left( \prod_{k=1}^s \mathcal{J}(\sigma) W_{hh} \right) W_{xh} 
$$

Now, if we are to upper bound the norm of the cost gradient,  

$$
\lVert \frac{\partial C}{\partial x(1)} \rVert \leq \Vert \frac{\partial C}{\partial h(s)} \Vert \, \Vert \left( \mathcal{J}(\sigma) \, W_{hh}\right) \Vert^s \, \Vert W_{xh} \Vert  
$$

$$
\lVert \frac{\partial C}{\partial x(1)} \rVert \leq \Vert \frac{\partial C}{\partial h(s)} \Vert \, \Vert W_{xh} \Vert \, \mu^s   
$$

Assuming the matrix norm of the Jacobian, $\mathcal{J}(\sigma) \, W_{hh}$, is upper bounded by $\mu$, we have the inequality above which states that the upper bound of the cost derivative norm decays by a factor of $\mu$ based on how high $s$ is.  In other words, increasing the size of the context ($s$), any perturbations of the input far back in context do not propagate to the loss gradient.

**With Traveling Waves**

With the Traveling Waves in defined previously, we have 

$$h(s) = \epsilon + \sum_{\mu=1}^s \sum_{i}^d (x(\mu))^i \, \psi_{(\mu-1)d+i} \, .$$ 

Here $\epsilon$ is any activity in addition to the wave activity, and $\psi$ is an arbitrary choice of bases to represent the dynamics.  We derive the gradient of the cost function with respect to the input at timestep 1.  \

$$
\frac{\partial C}{\partial x(1)} = \frac{\partial C}{\partial h(s+1)} \, \mathcal{J}(\sigma) \, W_{hh} \left( \sum_{i}^d \psi_{i} +  \frac{\partial \epsilon}{\partial x(1)} \right)  
$$

Similar to the previous analysis, the upper bound of the norm is  

$$
\Vert \frac{\partial C}{\partial x(1)} \Vert \leq \Vert \frac{\partial C}{\partial h(s+1)} \Vert \, \left( \mu d +  \mu \Vert \frac{\partial \epsilon}{\partial x(1)} \Vert \right)  
$$
  
It can be noted here that compared to the analysis without traveling waves, there is no power on the $\mu d$ term when computing the gradient. There can be higher powers of $\mu$ in $\mu \Vert \frac{\partial \epsilon}{\partial x(1)} \Vert$ terms, however, these terms contribute only additively to the upper bound. As a result, the gradient upper bound need not vanish for the traveling waves.

In other words, since each of the RNN inputs $x(t)$ is stored explicitly spatially in the hidden state as the activity of propagating waves, how far back the input is, does not effect gradient computation just like the context in a transformer. This holds as long as the gradient is propagated within the length of the wave substrate. 

# Traveling Waves across RNNs

## Elman RNNs

To apply the traveling wave theory to Elman RNNs, we can consider a generalized version of the model 
introduced above

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/rnn_twwm.png" class="d-block mx-auto img-fluid w-50"%}

$$  
\begin{cases}  
h_t = \pmatrix{C & O} \, A \, \pmatrix{C^{\dagger} \\ O^{\dagger}} h_{t-1} \, , \\  
y_t = C^{\dagger} h_t \, .  
\end{cases}  
$$

Here the $h_t$ and $y_t$ has the usual meaning and the $O$ defines an auxiliary vector space where the part of the  
hidden layer activity not relevant to the current output lies. The generalized inverse $O^{\dagger}$ and $C^{\dagger}$  
such that $C^{\dagger} (C \quad O) = (I \quad 0)$.  
Note that the number of possible RNN solutions to the problem is infinite as any linearly independent collection columns of  
$C$ and $O$ can span a basis for encoding the traveling wave solution <d-cite key="karuvally2023episodic"></d-cite>. The main restriction in the solution is that  
the recurrent matrix $A$ is structured of the following form

$$  
A = \pmatrix{0 & 0 & 0 & \cdots & \phi_{-1} \\  
I & 0 & 0 & \cdots & \phi_{-2} \\  
\vdots & \vdots & \vdots & \ddots & \vdots \\  
0 & 0 & 0 & \cdots & \phi_{-s} } \, .  
$$

At this point, it is instructive to note how the structure of $A$ is related to the Shift SSM described above.  The $A$ matrix is written generally as $A = \bar{A}_{\text{shift}} + \Phi$ where $\Phi$ is the VARMA lag operator. Thus, the infinite RNN solutions to the $\text{VARMA}(s)$ process is unified under the traveling wave principle of memory storage.

Our analysis of the possible RNN solutions to the VARMA process shows that to analyze the learned parameters of an  
RNN, we need to resolve two points:
- There is infinite possible solutions to the VARMA process for the infinite possible settings of $C$ and $O$.
- The $A$ is structured with a unique distribution of eigenvalue signature that depends on what the VARMA operators are.

To address our original question about how trained RNNs store memory,  
we therefore look at the eigenvalue signature of the Jacobian of RNNs trained to solve a collection of $\text{VARMA(s)}$ processes <d-cite key="Sussillo2013OpeningTB,Maheswaranathan2019UniversalityAI"></d-cite>.  
We note that the eigenvalue signature works as an analysis tool because it is invariant under  
basis changes (the infinite possible settings of $C$ and $O$). The plots below show the converging eigenvalue spectrum for the Jacobian of trained Elman RNNs on different VARMA processes whose theoretical eigenvalue distribution can be analytically derived from the $A$ above. The gray circle is the unit cirlce, red $\times$ are the theoretical eigenvalues, blue $\times$ are the experimental ones. The number of eigenvalues along each annotated with the corresponding colors. 

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/spectrum_T1.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/spectrum_T2.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/spectrum_T3.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/spectrum_T4.png" class="d-block mx-auto img-fluid w-50"%}

The above plots show that there is a remarkable convergence of the eigenvalue spectrum to the structure implied by the traveling wave solutions of the problem. If we can find the  
correct basis change, this structure can be revealed in the learned parameters of the network and the hidden state. To find the correct basis change, we remark that according to  
the theoretical model, the learned operator for extracting $y$ is $C^{\dagger}$ which means that a simple power iteration  
method can be used to extract the basis change. This reveals the following structure in the learned parameters.

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/Atransformed_T1.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/Atransformed_T2.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/Atransformed_T3.png" class="d-block mx-auto img-fluid w-50"%}

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/Atransformed_T4.png" class="d-block mx-auto img-fluid w-50"%}

In the transformed basis, the hidden traveling waves storing the recent history is revealed.

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/hidden_state.png" class="d-block mx-auto img-fluid w-100"%}

The diagonal lines in the figure shows the propagating waves of activity which in this case moves from the hidden neurons with index 0 to higher indices. Moreover, the content in them can be revealed by using the basis change. The following figure shows the input that was given to the network (a sequence of "315") traveling within the neural activity of the network.

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/wave_prop_RNN.png" class="d-block mx-auto img-fluid w-50"%}

The amazing insight from the above results is that traveling waves _naturally_ emerge in Elman RNNs and is a fundamental RNN memory mechanism during sequence processes.  Unlike Shift-SSMs and Transformers, where structures like the shift matrix or the context vector are explicitly designed into the architecture, Elman RNNs lack such constraints. The result thus suggests that traveling waves are not merely a product of specific architectural choices but arise _naturally_ from the process of training on sequential data. It may be that the inherent statistical structure in sequential data—such as the Vector Autoregressive Moving Average (VARMA) processes often present in real-world time-series—induces these wave dynamics. The emergence of traveling waves also highlight a remarkable synergy between the data and the model: sequential data inherently guides RNNs toward organizing their internal dynamics as traveling waves, enabling them to encode and propagate memory effectively.

A very counterintuitive result of the emergence of traveling waves in Elman RNNs is that the RNN improves the diminishing
gradient issue as training progresses. That is, RNNs before training have poor gradient properties compared to RNNs
after training.

{% include figure.html path="/assets/img/2025-04-28-waves-in-rnns/gradient_propagation.png" class="d-block mx-auto img-fluid w-50"%}


## Transformers

Recall that we defined the autoregressive evolution of transformers as 

$$
h_{t} = \sum_{i=1}^{s} \frac{\exp( h_{t-i}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})}{\sum_{k} \exp( h_{t-k}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})} \mathbf{V} h_{t-i} \\
$$

This may be seen as a generalization of the $\text{VARMA}(s)$ process by redefining $\phi$ to be more generally depend
on the input at the current step and the sequence some time steps back. That is 

$$
h_{t} = \sum_{i=1}^{s} \left( \phi(h_{t-i}, h_{t-1}) \right) h_{t-i}  \quad \text{and} \quad \phi(h_{t-i}, h_{t-1}) = \frac{\exp( h_{t-i}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})}{\sum_{k} \exp( h_{t-k}^\top \mathbf{K} \, \mathbf{Q} h_{t-1})} \mathbf{V}
$$

This is a different way to look at transformers, the traveling wave theory suggests that transformers also store the
recent history by propagating waves, but the $\phi$ functions can be input-dependent allowing them to change over time
unlike RNNs. This suggests that the representative power of transformers are strictly higher than Elmann RNNs as they 
are capable of learning only input independent $\phi$ parameters. However, the common structure is present. 

## SSMs

Analogously, consider the Shift-SSMs that are defined as

$$  
\begin{aligned}  
\mathbf{x}_{k} & = \mathbf{\bar{A}_{shift}}\mathbf{x}_{k-1} + \mathbf{\bar{B}_{shift}}u_k, \\  
y_k & = \mathbf{\bar{C}}\mathbf{x}_k,  
\end{aligned}  
$$

with

$$  
\begin{aligned}  
\mathbf{\bar{A}_{shift}} =  
\begin{bmatrix}  
0 & 0 & 0 & \cdots & 0 & 0 \\  
1 & 0 & 0 & \cdots & 0 & 0 \\  
0 & 1 & 0 & \cdots & 0 & 0 \\  
0 & 0 & 1 & \cdots & 0 & 0 \\  
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\  
0 & 0 & 0 & \cdots & 1 & 0  
\end{bmatrix},  
\quad  
\mathbf{\bar{B}_{shift}} =  
\begin{bmatrix}  
1 \\  
0 \\  
0 \\  
0 \\  
\vdots \\  
0  
\end{bmatrix}.  
\end{aligned}  
$$

The shift SSM is exaclty the traveling wave parameterization with the $\phi_i = 0$ which means that there are
no boundary effects. Without boundary effects the recent history is precisely stored in SSMs without interference.
This may be one of the reasons that a pure ShiftSSM is effective as a starting block of frontier SSM models. We would like to
note here that with just the shift matrix, there is only memory store and no computations suggesting that Shift-SSMs alone
are uninteresting RNN models and their power comes from plugging additional diagonal SSM as found in Mamba. 

# Future of Waves in RNNs

The results above should be convincing enough to claim that the simple traveling wave model of RNN mmemory storage is at the heart of RNN architectures - from the simplest Elman RNNs to the more complex frontier transformer and Mamba models. Over the years, 
despite deviations from the Shift formulation in the form of diagonal SSMs, the Shift SSMs still play a major role
in sequence processing. The prevalence of the approach suggests that the next generation of sequence processing models 
may be obtained by generalizing the wave theory. Here I list some potential avenues:

- The traveling wave theory assumes 1D wave propagation. While this has nice properties by representing memories non-overlapping, it may restrict the otherwise complex wave behaviors that are observed in neurobiological experiments. Thus, there is a potential to extend the wave theory to higher dimensions
- Bounds on the length of memory. The waves constraint the memory capacity to the most recent tokens that are seen so far, although they have the capability to compress to an extend (especially in the input dependent $\phi$ of transformers). A more natural form of memory store would selectively choose the items that can be stored while ignoring irrelevant ones. Implementing such a system may be challenging as the analysis of RNNs show that wave dynamics may be an inescapable state of RNN models.

