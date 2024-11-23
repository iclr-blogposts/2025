---
layout: distill
title: MasterMind: Empowering LLMs for Strategic Game Mastery
description: When LLMs encounter strategic decision-making games such as DouDiZhu or Go, a fascinating interplay emerges between advanced computational intelligence and intricate human-designed game mechanics. Our pioneering exploration delves into the complex reasoning logic and the upper limits of LLMs' strategic capabilities within these games. Furthermore, these games may serve as potent evaluative tools and catalysts for enhancing the logical reasoning prowess of LLMs, offering the potential to uncover new dimensions of their thought processes and inference mechanisms.
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
bibliography: 2025-04-28-mastermind.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Abstract
  - name: Introduction
    subsections:
    - name: Interactive Figures
  - name: MasterMind-Dou
    subsections:
    - name: Data Collection
    - name: Training Pipeline
  - name: MasterMind-Go
    subsections:
    - name: Data Collection
    - name: Training and Inference Pipeline
  - name: Experiments
    subsections:
    - name: Evaluation Settings
    - name: Results
    - name: Ablations
  - name: Discussion
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

## Abstract

Large Language Models (LLMs) have exhibited impressive capabilities across numerous domains, yet they often struggle with complex reasoning and decision-making tasks. As playing games often requires a diversity of reasoning logic, we believe them good sandboxes to evaluate and boost the reasoning ability of LLMs. In this work, we first explore whether LLMs can master complex decision-making games through targeted post-training. To this end, we collect extensive offline datasets from two classic games, Doudizhu and Go, and develop a suite of techniques to effectively incorporate this data into LLM training, resulting in two novel agents: Mastermind-Dou and Mastermind-Go. Our experimental results demonstrate that these Mastermind LLMs achieve competitive performance in their respective games. Additionally, we explore whether integrating decision-making data can enhance the general reasoning abilities of LLMs. Our findings suggest that such post-training improves certain aspects of reasoning, providing valuable insights for optimizing LLM data collection.

## Introduction

Language serves as an important role in human communication and the reflection of intricate thought processes. In recent years, the advancement of large language models (LLMs) has made it possible for artificial intelligence to understand and master human language <d-cite key="brown2020language"></d-cite><d-cite key="chowdhery2023palm"></d-cite><d-cite key="touvron2023llama"></d-cite>, achieving human-level performances in various linguistic tasks <d-cite key="wang2019superglue"></d-cite><d-cite key="adiwardana2020towards"></d-cite>. Although modern LLMs can be applied to a variety of tasks in a zero-shot manner, their logical reasoning abilities remain less than satisfactory, due to the absence of a comprehensive understanding of tasks at a deep and holistic level <d-cite key="dziri2024faith"></d-cite>. 

Research related to the reasoning capabilities of language models often focuses on code or mathematical problems. Researchers typically aim to evaluate and improve the reasoning capabilities of models on these tasks and have devised many approaches, including Self-Consistency (SC) <d-cite key="wang2022self"></d-cite> and Chain-of-Thought (CoT) <d-cite key="wei2022chain"></d-cite><d-cite key="kim2023cot"></d-cite>. However, while code and math serve as structured and symbolic data, it may not enclose the full reasoning process and lack textual format diversity expressed in natural language <d-cite key="ma2023training"></d-cite>, which tends to be ambiguous. Consequently, adhering to the paradigm of evaluating and boosting LLMs with code and math problems may be not enough to generalize across varied tasks. A natural question arises: *Is there another alternative tasks that can complement code and math, thereby better evaluating and elevating LLMs from the perspective of data quality and diversity?* In this work, we explore the potential of decision-making games as a complementary approach. We find that game data, when represented with appropriate textual structures, can provide valuable reasoning examples, broadening the scope for evaluating and enhancing language models.

{% include figure.html path="assets/img/2025-04-28-mastermind/overview.png" class="img-fluid" %}

Decision-making games, akin to code and math, usually posses a strict logical structure owing to its native mechanisms such as the state transition and the legal action range <d-cite key="zha2019rlcard"></d-cite>. And the reasoning process in games can usually be split into several stages, simulating humans' step-by-step reasoning process. Moreover, games often contain complex strategies that may involve hierarchical planning <d-cite key="levy2017hierarchical"></d-cite>, gambling operations <d-cite key="silver2017mastering"></d-cite>, cooperative and competitive multi-agent behaviours <d-cite key="lowe2017multi"></d-cite>, which can closely mirror reasoning strategies employed by humans. Additionally, responses from decision-making games under similar actions tend to exhibit consistency or follow a predictable distribution <d-cite key="silver2018general"></d-cite>, fostering stability in the reasoning process.

Furthermore, game data exhibits a unique set of attributes. While code and math data often provides explicit information, game data include uncertainty and imperfect information, akin to real-world decision-making contexts. This imperfection can be further dynamically controlled when preparing training data for LLMs, as elucidated in the next section . Additionally, there are many available open-source agents on games <d-cite key="niu2024lightzero"></d-cite>, enabling automated data collection across diverse environments. And a variety of game simulators <d-cite key="brockman2016openai"></d-cite> allows for the customized generation of desired data. These properties not only diminish the requirements for manual annotation, but also enable the accumulation of larger training datasets for LLMs, which is easy to use for the development of more general agents. Moreover, many popular games like StarCraft <d-cite key="starcraft_ii"></d-cite> and DOTA2 <d-cite key="dota_2"></d-cite> necessitate multi-modal information, stochasticity modelling, and intricate game numerical design, which can therefore provide large models with multifaceted capability requirements.

In this blog, since we focus only on processing textual information, we conduct experiments on Doudizhu and Go, which challenge models with incomplete information and complex multi-step search capabilities, respectively. Our results show that current language models perform poorly on these tasks, struggling to identify valid actions even with few-shot learning and strategies based on similarity calculations. To address this, we aim to train models using tailored datasets to master these tasks effectively.

To implement, we initially collect datasets encompassing varying levels of strategies and game opponents. We meticulously implement some methods to convert the original game data to proper textual representations, such as the dynamic candidate actions and opponent responses in Doudizhu and the step-by-step natural language analysis in Go. Moreover, we design two techniques to prevent the overfitting of LLMs to complicated game rules, thereby ensuring a focus on the core reasoning process. To address the issue posed by the imperfect information in Doudizhu, we co-train an opponent strategy prediction model to anticipate the most probable actions of opponents in the next turn, providing more insights for LLMs' following selections. For the accuracy of the score difference calculation in Go, we integrate some tool functions (like code interpreter <d-cite key="schick2024toolformer"></d-cite>). This integration serves to alleviate the cognitive burden on LLMs, enabling them to focus less on numerical details and more on developing a comprehensive game understanding.

By adopting the above data collection and training techniques, we fine-tune some popular open-source LLMs and enhance their logic reasoning and game skills across various dimensions. We call derived models as Mastermind LLMs (agents). The experimental findings reveal that, our Mastermind LLMs not only reach competitive levels in corresponding games, but also display performance enhancements in unseen and general reasoning tasks. 

The main contributions can be summarized as follows:


- We develop a series of techniques to fine-tune LLMs on two games, *Doudizhu* and *Go*, showcasing competitive performance in their respective tasks.
- We introduce decision-making games as a new data source for LLMs, highlighting their potential over code or math problems for enhancing step-by-step reasoning capabilities.
- We conduct thorough ablation experiments to analyze the design of our data collection and training pipeline. All the code, data, and trained agents will be released soon.

## MasterMind-Dou

Doudizhu, a popular Chinese card game played by three players, stands out as an proper choice for above-mentioned purposes, primarily due to its turn-based gameplay and controllable imperfect information. In Doudizhu, players bid to become the 'landlord' and aim to empty their hands first. In each turn, players need to play higher-ranking cards, with strategies including bluffing and reading opponents' moves. By unmasking opponents' handcards, the decision-making data transform from a strategy reliant on gambling to one focused on selecting optimal actions.  Additionally, the game's observation space and actions are based on a standard 54-card deck, which is simple to convert it into text data for training. Furthermore, Doudizhu's multi-step reasoning demands make it an ideal source for decision-making data.

### Data Collection

**Card Textual Representation**
To simply and clearly represent the data in Doudizhu, we have chosen to build upon and refine the approach introduced in DouZero <d-cite key="zha2021douzero"></d-cite>. Specifically, each card is translated into an integer, and possible combinations of cards are represented as sorted lists of integers. Their correspondences are listed as below:

| **Card Face** | **Integer** | **Card Face** | **Integer** | **Card Face** | **Integer** | **Card Face**  | **Integer** |
|---------------|-------------|---------------|-------------|---------------|-------------|----------------|-------------|
| 3             | 3           | 4             | 4           | 5             | 5           | 6              | 6           |
| 7             | 7           | 8             | 8           | 9             | 9           | 10             | 10          |
| J             | 11          | Q             | 12          | K             | 13          | A              | 14          |
| 2             | 17          | Black Joker   | 20          | Red Joker     | 30          |                |             |

**Preliminary Filtering of Candidate Actions**
We discretize and simplify the action space by providing a list of legal actions, a critical step in reducing the complexity of the space that LLM need to consider when making decisions. However, the extensive legal action space in Doudizhu, exceeding 27,000 combinations, poses a challenge in context length and reasoning complexity for LLM. Furthermore, the fluctuating length of legal actions poses a significant obstacle for LLMs to develop adaptable strategies. To tackle this, we utilize a pre-trained Q network from DouZero to obtain the logits for each action and employ Top-$p$ sampling, choosing actions with a combined probability of 25%, thus reducing the action space size while preserving the optimal action sets for winning.

**Data Generating Agent**
We require diverse datasets of different player strategies across distinct positions.  This is essential as, in any given position, we need to predict not only the optimal action set for the current player but also opponents' potential moves. Furthermore, the network we intend to train need to generalize to any new opponent strategies, underscoring the need for a varied data source. Therefore, we utilize three types of agents: rule-based <d-cite key="zha2019rlcard"></d-cite>, supervised learning-based, and DouZero agents <d-cite key="zha2021douzero"></d-cite> enhanced by RL. Each agent contributes to generating strategy datasets by providing prompts that specify their proficiency levels, mapping states to Top-$N$ actions ($s\rightarrow a$).

### Training Pipeline

Doudizhu's outcome is not deterministic but heavily influenced by concrete handcards and strategies, where optimal actions hinge on both own observations and unseen opponents' strategies.  For instance, in a scenario where a player consistently leads with high-value cards early in the game, it may indicate an too aggressive (or conservative) strategy. To address it, we introduce an effective approach that initiates with the deductive reasoning of one's own possible actions and the subsequent anticipation of opponents' responses. The entire training paradigm can be divided into three parts: possible action prediction, opponent strategy prediction, and final action selection.

{% include figure.html path="assets/img/2025-04-28-mastermind/doudizhu_pipeline.png" class="img-fluid" height="600px" width="600px"%}

**Possible Action Prediction:** Based on the textual representation of historical records and the player's current handcards, Mastermind-Dou is trained to predict the Top-N possible actions associated with a heightened probability of winning. The concrete format is:

> My handcards are ..., while the opponents' handcards are ... My possible actions are ...

**Opponent Strategy Prediction: **Subsequent to the ascertainment of possible actions at hand, we design an additional neural network head on the shelf of the LLM. This module aims to predict the following action of opponents after the player performs a specific current action. Firstly, we fine-tune the LLM to learn the common pattern among expert Doudizhu players, thereby enabling it to predict the most likely actions accurately in the following format:

> My handcards are ..., while the opponents' handcards are ... My action is ...

The optimization objective is the cross-entropy between the actual action of opponent and the predicted action.  During inference, since the output of LLM may occasionally transgress the bounds of legal actions, we utilize sentence-BERT <d-cite key="reimers2019sentence"></d-cite> to map the original output to the legal action with the most similar embedding. In addition, akin to in-context learning, we provide this fine-tuned model with samples of current opponents, facilitating its rapid adaption to a particular opponent's policy during evaluation.

By training on our curated datasets, the LLM agent can better anticipate opponents' behaviours and refine its own strategy.
This enhancement significantly amplifies the agent's overall efficacy since the LLM is not confined to a fixed strategy.
Instead, it engages in dynamic and adaptive decision-making processes within the context of Doudizhu. This paradigm also controls the imperfect information and stochastic elements in Doudizhu, making it more suitable for the augmentation of the LLM's reasoning capabilities.

**Final Action Selection:** Finally, by prognostically evaluating one's own possible actions opponents' reactions, we can concatenate these analysis to to construct the comprehensive query. Then the LLM will understand this information and generate the answer, namely, the final selected action. The format of query is:

> If I play $$action$$, the next two players will play $$action_1$$, $$action_2$$ ... Therefore, I will play $$action$$

Some example contents of these dataset can be viewed in the following table.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-mastermind/doudizhu_dataset.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>


## Mastermind-Go

Complementing our exploration of Doudizhu, we extend our focus to Go as an additional domain to enhance the reasoning ability of LLMs. In the rules of Go, black and white pieces take turns to be placed on the board, with the ultimate goal of surrounding as large territories on the board as possible.  Compared to Doudizhu, Go possesses higher complexity both in rule and game strategy, thus it requires stronger logical reasoning ability. At the same time, since Go is played in two-dimensional board while the model can only accept one-dimensional sequential input, LLMs must perform implicit modeling of two-dimensional space. This requirement for sequential modeling ability even surpasses that of Doudizhu.

To better introduce our design approach, we will first provide a brief overview of the rules of Go. In Go, two players, black and white, alternately place their stones on the intersections of a 19x19 board, with the goal of surrounding the most territory by the end of the game. Our analysis of the game situation can be broken down into several steps, as illustrated in the diagram below:

- The top-left diagram shows the current state of the board, including all past moves made by both black and white players.
- The top-right diagram shows the ownership status of each intersection on the board (occupied by black, white, or currently undetermined). It is important to note that even if an intersection is currently occupied by one side, it could potentially be reclaimed by the opponent in subsequent moves. This task requires a deep understanding of specific board areas, which may span multiple lines, in order to be executed accurately.
- The bottom-left diagram presents the current difference in the number of territories held by black and white, as well as the win probability for black.
- The bottom-right diagram contains a commentary on the current game situation. As can be seen, although black currently leads in territory, the win probability slightly favors white. This is because the black stones in the upper area are surrounded by white, giving white a potential opportunity to gain an advantage. Therefore, evaluating the win probability is a complex process that involves taking multiple factors into account.

{% include figure.html path="assets/img/2025-04-28-mastermind/go_dataset.png" class="img-fluid" height="600px" width="600px"%}

Through the above analysis, we can see that playing Go requires various types of logical reasoning skills and involves a lengthy process. Considering the this property, we design a curriculum of training tasks in a gradual and step-by-step progression. Our approach prioritizes the infusion of comprehensive thought processes into the LLM's cognitive capabilities underpinning gameplay. This principle is similar to procedure cloning <d-cite key="wei2022chain"></d-cite>, enabling the model to learn more about the underlying logic behind decision-making rather than the simple state-action mapping. This method also aligns with the concept of CoT, which encourages the model to solve problems via a systematic and step-by-step methodology, thereby enhancing its reasoning capabilities.

{% include figure.html path="assets/img/2025-04-28-mastermind/go_pipeline.png" class="img-fluid" height="600px" width="600px"%}

As is shown in the figure, Task 1 is from the rule level, where we have the model predict the next state based on the current board state and actions. Task 2 is from short-term and state understanding level, where we use the open-source Go agent, KataGo, to generate evaluations for current board. Task 3 is from long-term and natural language understanding level, where we utilize information from Go books to extract several state-explanation pairs, thereby enhancing the model's understanding of the game states through natural language. Task 4 is from the decision level, where we combine all information mentioned above to make the final decision in a step by step manner. We will detail the specific methods of data collection in the next subsection.

### Data Collection

**Board Textual Representation:** Firstly, we explain the method of converting the original Go board into its textual form.  According to the rules of Go, the dimensions of board is fixed at 19x19, with each slot potentially occupied by a black stone, a white stone, or a remaining vacant. Since the board is 2-dimensional, it is challenging to convert it into 1-dimensional language sequence without losing any spatial information. In our settings, we use symbols "o", "#" and "•" to represent white stones, black stones, and unoccupied positions. To facilitate precise reference to each position on the board, we assign labels to 19 rows and 19 columns, ranging from 1 to 19 and A to T, respectively, with the position at the bottom-left corner denoted as "A1". Additionally, to encode historical information, we assign numbers to the most recent moves of both sides, such as "o(1)" and "#(3)" to denote the first move by a white stone and the third move by a black stone.  This approach completes the representation of the board, converting all necessary information on the board into textual format. 

**Data Collection for Basic Rules:** To ensure the model grasps the core rules of Go (especially capturing stones), we begin with predicting the state-action transition of the game.  Typically, there are two types of pre-training tasks to achieve this goal: 1) $s,s'\rightarrow a$, i.e. predicting corresponding action given adjacent board states; 2) $s,a\rightarrow s'$, i.e. predicting next board state given current state and action. Ultimately, the collected data is presented as follows:

> The following is a game record of 19x19 Go ... The next move is on ... Please predict the next Go board after this move ...

**Data Collection from Agents:** After understanding the fundamental rules of Go, we choose KataGo <d-cite key="wu2019accelerating"></d-cite> as the agent for generating evaluations about current game states. KataGo is a reinforcement learning agent based on Monte Carlo Tree Search (MCTS), displaying capabilities that exceed human-level in Go. 

In building our dataset, we didn't mimic MCTS search paths due to the unknown logic behind Go moves in MCTS and the exponential growth in search space, which lowers efficiency with too many tokens. Our approach aims to help the model grasp the reasoning behind move placements, implemented as follows: Firstly, we record KataGo self-play games and save the state of the board at each move. Next, we analyze each board state to identify several potential move candidates. After obtaining these potential moves, we simulate the board state after playing these moves and passed the updated board state to KataGo for analysis. We include three aspects of analysis information on the simulation result: 1) the ownership of territories on the board; 2) the lead in points for current player; 3) the winning probability in the current board state. Finally, all the above information is converted into text form and stored in the training data. 

In crafting the model's response, we have adopted a similar design to the CoT approach. Initially, the model is guided to predict the basic ownership map, subsequently gauging the respective point difference based on the size of territories owned by black and white sides, respectively. Finally, it predicts the winning probability. By employing this progressive thinking method, the LLM is able to better utilize decision-making data for reasoning. The training samples finally adopt this following template:

> The following is a game board of Go. ... Here is the board state: ... Please predict the ownership map ... leading score ... and win rate.

**Data Collection from Books:** The next step is to enable the model to accurately assess the current state on from a long-term and natural language perspective. 
Thus, we leverage resources from existing Go books <d-cite key="LeeSedol"></d-cite>, extracting the pairs of game states and corresponding explanations about judgement of current situation and long-term strategy.
The training sample thus adopts the following format:

> The following is a game board of Go. ... Here is the board state: ... Please generate the corresponding explanations. ...

Some example contents of these datasets can be viewed in the following table.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-mastermind/go_dataset.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

**Combined Training:**
Finally, we combine the above data for training as a complete task to boost the LLM. In doing so, Mastermind-Go fully utilizes Go training data from various levels of thought, integrating all knowledge to deliver the optimal decision.

To enrich our dataset with more diversity, we choose two distinct levels of agents for data-generation. The first one is called KataGo-9d, an agent that consistently employs the optimal move; the second one is KataGo-suboptimal, which uses the Top-$p$ sampling to randomly stochastically actions according to the policy network in Katago. We choose $p=0.4$ in our implementation.
This dual-tiered approach facilitates comprehensive game trajectories.

### Training and Inference Pipeline

After data collections across different tasks, we mix up these samples with calibrated proportions to fine-tune the LLM. Within each training instance, a question-answer pair is presented.
We denote the question of the $i$-th sample as $X_i$ and the answer as $Y_i$. Our training regimen employs the cross-entropy loss function, a methodological choice that facilitates the model's ability to maximize the likelihood probability of yielding the complate answer upon receiving an input query. Formally, this loss function can be defined as 

$$L_{\theta} = \frac{1}{N}\sum_{i}^{N} \log_{\theta} (Y_i \| X_i)$$.

This equation represents the sum of the logarithms of the conditional probabilities of the answers $Y_i$ given the questions $X_i$ across all training samples. The goal is to maximize these probabilities, which in turn minimizes the cross-entropy loss, leading to a model that is better at predicting the answers to the given questions.

However, we find that the LLM performs poorly in counting the territories of both black and white sides when given an ownership map, which adds much difficulty to the subsequent prediction of winning probabilities. To further enhance the performance, we also designed utility functions for the LLM to use when generating answers, thereby obtaining more reliable results. Specifically, we designed a "Count" function to count the number of black and white stones in the generated ownership map. During the training phase, this step is not operational and is treated as ordinary text data for learning through the cross-entropy loss. However, during the testing and generation phase, when the model generates "-$$>$$", the generation process will be interrupted and shifted to the utility function for calculation. After the utility function's calculation result is concatenated back into the generated sentence, the model will continue to generate, ultimately completing the overall response.

## Experiments

In the next experiment subsections, we empirically explore the impact of fine-tuning LLMs on decision-making game data in following two aspects:

- How does our proposed data collection and training regime facilitate LLMs in grasping the intricacies and knowledge of decision-making games including Doudizhu and Go.
- Is this approach of fine-tuning LLMs with decision-making game data opens up a discourse on the potential impact it may have on enhancing their reasoning capabilities.

### Evaluation Settings

In terms of experimental setup, this study selected the LLaMA-2-7B model as the primary base model for training, with the main training conducted on a machine equipped with 8$$\times$$A100 GPUs. Regarding hyperparameter selection, the primary hyperparameters chosen include:

| Hyper-Parameter | Value | Hyper-Parameter        | Value |
| --------------- | ----- | ---------------------- | ----- |
| Optimizer       | AdamW | LR Scheduler           | cos   |
| Learning Rate   | 5e-5  | Gradient Clipping Norm | 1.0   |
| Beta1           | 0.99  | Warm-up Iterations     | 3000  |
| Beta2           | 0.999 | LoRA-r                 | 32    |
| Weight Decay    | 0.01  | LoRA-α                 | 64    |
| Batch Size      | 8     | Dropout                | 0.1   |

### Results

#### Doudizhu Tasks

To assess the performance of Doudizhu tasks, our evaluation concentrates on three primary metrics: alignment with expert data via action accuracy (**Act. Acc.**) and thought process matching (**CoT RLsum**), accuracy in forecasting the opponent's card-play probability (**Pred. Acc**). The results are listed in the following table.

To facilitate a better comparison, we evaluated five different methods:

- LLaMA-2-7B (0-shot): Directly inputting the question into the model, allowing the model to generate subsequent thoughts and responses, predicting the final move strategy.
- LLaMA-2-7B (few-shot): Providing several examples to the model as context, prompting the model to respond in a similar manner and give the final move strategy.
- LLaMA-2-7B (few-shot): Selecting the most relevant action from all possible actions by calculating the similarity of embeddings, and using it as the final move strategy.
- Mastermind-Dou w/o prob: Inputting the question into the model, which directly provides the final answer without intermediate reasoning.
- Mastermind-Dou with prob: Inputting the question into the model, allowing the model to perform step-by-step intermediate reasoning and then provide a comprehensive final answer.

|                                    | **Act. Acc. ↑** | **CoT. RLsum ↑** | **Pred. Acc. ↑** |
| ---------------------------------- | --------------- | ---------------- | ---------------- |
| LLaMA-2-7B (0-shot)                | 0%              | 0.52             | 0%               |
| LLaMA-2-7B (few-shot)              | 21.18%          | N/A              | N/A              |
| LLaMA-2-7B (few-shot + similarity) | 42.65%          | N/A              | N/A              |
| Mastermind-Dou w / o prob          | 78%             | N/A              | N/A              |
| Mastermind-Dou with prob           | **90%**         | **0.98**         | **39%**          |

Firstly, the performance on the Doudizhu task, as shown in the table above, demonstrates that the model before training is not able to grasp the rule of such a card game, resulting in poor performances. After training, the model has clearly acquired card-playing abilities and exhibits strong generalization capabilities. On previously unseen hands, the model achieves an 90% match with expert data and demonstrates a 98% similarity in thought processes. This indicates that the language model has not merely memorized specific card-playing strategies for certain scenarios but has learned an intrinsic logic of the Doudizhu game, showcasing excellent generalizability. 

Additionally, comparing the results with and without Chain-of-Thought (CoT) training reveals that incorporating a thought process can enhance the model's ability to generalize in its final decision-making. It is worth noting, however, that although the overall similarity in thought process is high, achieving perfect prediction of the entire thought chain is challenging, with an accuracy rate of only around 39%. This is likely due to Doudizhu's characteristic of imperfect information, where neither side can fully know the opponent’s hand, making it difficult to perfectly replicate the correct sequence of predictions about the opponent’s moves.

|                           | v.s. RLCard ↑ | **v.s. DouZero ↑** |
| ------------------------- | ------------- | ------------------ |
| LLaMA-2-7B                | 0%            | 0%                 |
| LLaMA-2-7B (few-shot)     | 12%           | 3%                 |
| Mastermind-Dou w / o prob | 78%           | 33%                |
| Mastermind-Dou with prob  | **90%**       | **41%**            |
| DouZero (Expert)          | 90%           | 43%                |

We further evaluated the trained agent by conducting match-ups with various open-source solutions, repeating each round 100 times. To ensure fair testing and eliminate the influence of teammate capabilities, our proposed agent was assigned the "landlord" role, while the two "farmer" roles were consistently filled by open-source agents. The results, as shown in the table above, indicate that the original model cannot even recognize which actions are legal. On the other hand, our proposed agent demonstrates strong performance. Against RLCard as the opponent, our agent achieved a 90% win rate, effectively inheriting the expertise-level performance. Similarly, when pitted against the expert model DouZero<d-cite key="zha2021douzero"></d-cite>, it achieved a 41% win rate, which is very close to DouZero's own expert-level win rate limit of 43%.

Interestingly, analysis reveals that our agent is not merely imitating DouZero’s actions; its strategy is not always aligned with DouZero’s. In some situations, despite DouZero ultimately losing, our agent—using a different strategy—achieved victory. This suggests that the language model may indeed be learning certain high-level strategies rather than simply performing behavior cloning. Here is a game replay, demonstrating one game that DouZero fails but MasterMind wins using a different strategy.

{% include figure.html path="assets/img/2025-04-28-mastermind/compare.gif" class="img-fluid" height="600px" width="600px"%}

#### Go Tasks

Due to the highly complex nature of Go, with its vast array of variations and its two-dimensional spatial input, the game is not ideally suited for language model learning. After initial attempts, we found that teaching a language model through supervised learning in this context incurs significant costs. Based on this analysis, we only evaluated Mastermind’s performance in Go on a few proxy tasks. 

Firstly, from the perspective of rule comprehension, we tested the prediction accuracy for next board states (**s' Acc.**). Secondly, to assess understanding of the game state, we measured the mean absolute error in score differences (**Score MAE**) and win rates (**Winrate MAE**). Lastly, on the natural language interpretation level, we examined the similarity between generated explanations and actual explanations (**expl. RLsum**), as well as the perplexity of true explanations (**expl. ppl.**).

|                                          | **s' Acc. ↑** | **expl. RLsum ↑** | **expl. ppl. ↓** | **Winrate MAE ↓** | **Score MAE ↓** |
|------------------------------------------|---------------|-------------------|------------------|-------------------|-----------------|
| LLaMA-2-7B                               | 0%            | 0.28              | 11.45            | N/A               | N/A             |
| Single-task                              | **99.44%**    | **0.44**          | 5.23             | 5.14              | 1.80            |
| Multi-task                               | 96.08%        | 0.43              | **3.64**         | **4.49**          | **1.74**        |

From the table above, it is evident that the MasterMind agent has developed a considerable understanding of Go’s rules, with nearly flawless accuracy in predicting the next game state. Go’s rules are inherently complex, as a single move can significantly alter the status of a large area on the board. To illustrate this complexity and the model’s predictive accuracy, we present three examples of successful predictions (localized sections), arranged in order of increasing difficulty. These examples showcase the intricacy of Go’s rules and the model's proficiency in accurately anticipating game state changes.

{% include figure.html path="assets/img/2025-04-28-mastermind/go_example.png" class="img-fluid" height="600px" width="600px"%}

Additionally, we found that the model performs well in analyzing more challenging game situations. Go situation analysis involves two main aspects: first, estimating the current territory size, and second, predicting the overall win rate. Since Go's ultimate outcome is determined by territory size, these two metrics are somewhat correlated. However, win rate estimation also requires assessing future potential moves, making it a more complex prediction task. 

The results show that the model accurately estimates the territorial scope for both players. In some cases, the model even reflects potential future developments in its territory size estimations. Furthermore, Mastermind's win rate predictions demonstrate accuracy, with the variance kept within 5%. Below are examples of Mastermind’s predictions for territory ownership and win rate.

{% include figure.html path="assets/img/2025-04-28-mastermind/go_example2.png" class="img-fluid" height="600px" width="600px"%}

#### General Reasoning Tasks

To detect the improvement in reasoning ability of the model trained in decision-making games, we validate Mastermind LLMs on BIG-Bench Hard (BBH)<d-cite key="suzgun2022challenging"></d-cite>, a challenging reasoning dataset for LLM.

| Acc.           | **TempSeq** | **PengTab** | **Snarks** | **RuinNames** | **Hyper.** | **Nav.**   |
| -------------- | ----------- | ----------- | ---------- | ------------- | ---------- | ---------- |
| LLaMA-2-7B     | 12.00%      | 31.51%      | 47.75%     | 32.80%        | 51.60%     | 53.60%     |
| Mastermind-Dou | 20.00%      | **35.62%**  | 49.44%     | 35.60%        | **62.80%** | 56.80%     |
| Mastermind-Go  | **20.40%**  | 29.45%      | **51.69%** | **39.20%**    | 51.60%     | **60.00%** |

Firstly, we observed a significant performance improvement in the model on specific subsets following Mastermind training, as shown in the table above. This improvement appears most prominently in reasoning tasks that require long-sequence modeling, likely due to the demands of both Go and Doudizhu, which test the model’s ability to make accurate predictions over extended outputs. 

| Acc.           | **Disambiguation qa** | **Date Understanding** | **Geometric Shapes** | **Deduction Three Objects** |
| -------------- | ----------- | ---------- | ------------- | ------------- |
| LLaMA-2-7B     | **46.8%** | **60.4%** | **19.2%** | **55.2** |
| Mastermind-Dou | 32.4% | 22.8% | 0% | 41.6 |
| Mastermind-Go  | 44.8% | 57.6% | 7.6% | 51.2 |

However, it is important to note that the model exhibited a decline in performance on certain tasks, as highlighted in the table above. This decline is likely attributable to catastrophic forgetting—a phenomenon where certain skills or knowledge, such as information about dates and spatial figures, were not adequately reinforced during the Mastermind training process.

To fully address this issue, we believe that introducing these data into the pretraining phase may be necessary to enhance overall performance. However, due to resource constraints, we were unable to implement this idea in the current study, leaving it as a potential direction for future work.

### Ablations

We also conducted additional ablations on other types of large language models (LLMs), selecting Google’s Gemma <d-cite key="team2024gemma"></d-cite> and fine-tuning it on the Doudizhu dataset. As shown in the table, similar improvements are observed in the Doudizhu-related metrics across different types and sizes of LLMs. Furthermore, the overall performance gains are more pronounced in larger models, indicating a scaling law effect. This experiment supports the idea that decision-making game data could serve as a valuable data source for open-source LLMs in general.

| **Model**     | **Act. Acc. ↑**   | **CoT RLsum ↑**  |
|---------------|-------------------|------------------|
| Gemma-2B | 0% | 0.40 |
| Gemma-7B | 0% | 0.44 |
| Gemma-2B-Mastermind | 76.69%         | 0.97       |
| Gemma-7B-Mastermind | **86.27%**     | **0.98**   |

## Discussion

Concurrently, some recent groundbreaking models, such as LLaMA <d-cite key="touvron2023llama"></d-cite>, Alpaca <d-cite key="alpaca"></d-cite> and Deepseek <d-cite key="shao2024deepseekmath"></d-cite>, owe a significant part of their progresses to the integration of code and math problems during training. These works argue that leveraging code and math data improves the intrinsic reasoning skills of foundation models. This claim is also supported by subsequent experiments showing that CoT further enhances this advantage <d-cite key="fu2022gptroadmap"></d-cite><d-cite key="wei2022chain"></d-cite>.

Several insights help explain this phenomenon: 1) Pre-training provides LLMs with exposure to the logic inherent in programming and mathematical languages, thus facilitating the development of systematic reasoning <d-cite key="ma2023training"></d-cite>. 2) The dependencies and logic flow within these contents  contribute to the model's capabilities on capturing long-term dependency <d-cite key="wang2022self"></d-cite><d-cite key="ma2023training"></d-cite>. 3) The deterministic nature of code execution and math derivation ensures non-duality in output for the same input, mitigating ambiguity in LLMs' logical deductions. 

We posit that data generated from decision-making games shares similar advantages, while introducing unique challenges such as **uncertainty** and **imperfect information**. Our experiments reveal that such data contributes to improving general reasoning abilities. Consequently, we propose that similar synthetic data generation strategies be incorporated into future pre-training phases of large models to enhance the logical structure and diversity of reasoning challenges in the training dataset.

## Conclusion

In this blog, we have delved into empowering LLMs for mastering complex reasoning games. Unlike many existing works that focus on code and math problems, we firstly incorporate games to evaluate and boost the reasoning capacity of LLMs. Through a suite of our designed techniques in data collection and training, we have developed Mastermind agents, demonstrating commendable performance in both Doudizhu and Go. Empirical experiments serve to substantiate the potential of this approach in improving reasoning capabilities of LLMs. However, our experiments are conducted in the fine-tuning stage, yet game data harbors potential for improvement in the pre-training stage. Additionally, we aspire to broaden the scope of games to which our methods can be applied, especially in those that feature multi-modal representations and more intricate numerical designs.