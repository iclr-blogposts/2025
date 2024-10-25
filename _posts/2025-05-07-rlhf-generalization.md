---
layout: distill
title: Generalization Progress in RLHF - Insights into the Impact of Reward Models and PPO
description: In this blog, we summarize recent research and design experiments to explore the generalization process in RLHF. The generalization in RLHF involves two primary aspects - generalization originating from the training of the reward model and generalization resulting from PPO training. The generalization of the reward model primarily stems from the preference datasets and the inherent generalization capabilities of pre-trained models. Meanwhile, the generalization achieved through PPO training encompasses two key components - generalization derived from on-policy samples and generalization stemming from token-wise rewards. Based on these findings, we offer some recommendations for data construction in RLHF.
date: 2025-05-07
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-05-07-rlhf-generalization.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: The RLHF Process
  - name: Generalization Progress in Reward Model
    subsections:
    - name: Generalization from Preference Dataset
    - name: Generalization from Pre-trained Model
  - name: Generalization Progress in PPO
    subsections:
    - name: Generalization from On-policy Samples
    - name: Generalization from Token-Wise Reward
  - name: Data Construction Strategies for RLHF
  - name: Reference

---

## Introduction

Reinforcement Learning with Human Feedback (RLHF) methods, particularly Proximal Policy Optimization (PPO) <d-cite key="schulman2017proximal"></d-cite>, have proven to be effective in aligning large language models (LLMs). However, it remains unclear whether PPO can generalize to unseen prompts during training, and how exactly it generalizes to such prompts. To delve deeper into this issue, we have gathered recent research insights in this blog and carried out a series of experiments on the code generation task. Our discoveries highlight that the generalization process in RLHF comprises two essential aspects: generalization originating from the training of the reward model and generalization arising from the PPO training itself. Notably, the generalization from the training of the reward model is conveyed to the RLHF process through the diverse PPO prompts.

In summary, our findings are as follows:

- The generalization stemming from the reward model primarily originates from the preference datasets. During training, the reward model learns coarse patterns from these datasets and generalizes them to unseen samples. Additionally, the reward model inherits certain generalization abilities from the pre-trained models, enabling it to identify some easy errors in unseen negative responses.
- The generalization achieved through PPO training encompasses two primary components: generalization derived from on-policy samples and generalization stemming from token-wise rewards.

Finally, we give some recommendations for data construction in RLHF based on the above findings. 

## The RLHF Process

 The RLHF process consists of two main components: 

**Reward Model Training:** The reward model provides LLMs with a signal that guides the reinforcement learning process. In general, we first gather datasets consisting of prompts, LLM responses, and corresponding human feedback (ratings, rankings, or other forms of evaluation). This feedback serves as the ground truth for training the reward model. Then, we train the reward model using supervised learning techniques on the collected data. The model learns to predict the reward associated with each LLM response given a prompt and the corresponding human feedback <d-cite key="huggingface-blog-rlhf"></d-cite>.

{% include figure.html path="assets/img/2025-05-07-rlhf-generalization/reward-model.png" class="img-fluid" %} <div class="caption">Figure 1: Reward model training process (from <d-cite key="huggingface-blog-rlhf"></d-cite>).</div>

**Fine-tuning with RL:** Given a reward model, we employ RL to fine-tune the policy of a LLM. First, the **policy** is a language model that takes in a prompt and returns a sequence of text (or just probability distributions over text). The **action space** of this policy is all the tokens corresponding to the vocabulary of the language model and the **observation space** is the distribution of possible input token sequences, which is also quite **large given previous uses of RL** (the dimension is approximately the size of vocabulary ^ length of the input token sequence). The **reward function** is a combination of the preference model and a constraint on policy shift. Finally, the **update rule** is the parameter update of the policy from PPO that maximizes the reward metrics in the current batch of data <d-cite key="newfacade-notes-on-reinforcement-learning"></d-cite>.

{% include figure.html path="assets/img/2025-05-07-rlhf-generalization/image.png" class="img-fluid" %} <div class="caption">Figure 2: The reinforcement learning from human feedback (from <d-cite key="newfacade-notes-on-reinforcement-learning"></d-cite>).</div>

## Generalization Progress in Reward Model

### Generalization from Preference Dataset

Generally, we gather human rankings of pairs of LLM responses to create a human feedback dataset, commonly referred to as a preference dataset. The reward model is capable of uncovering patterns within pairs of ranked responses and applying these patterns to unseen pairs. It is generalization stemming from the preference dataset. 

OpenAI has demonstrated the generalization ability from reward model in RLHF progress:

- OpenAI first discovered that reward models (verifiers) scale significantly better with increased data compared to supervised fine-tuning <d-cite key="cobbe2021training"></d-cite>.
    They show that, compared to a fine-tuning baseline, the use of verifiers (reward model) results in approximately the same performance boost as a 30x model size increase, and that verifiers scale significantly better with increased data.
- Subsequently, OpenAI discovered the scaling law of the reward model and RL benefits from the scaling of the reward model.<d-cite key="gao2023scaling"></d-cite>
    * The losses on reward validation dataset break down by increasing preference dataset size and reward model size (Figure 3).
    * Both best of N strategy and RL benefits from the scaling of reward model size and preference dataset size (Figure 4).
      
{% include figure.html path="assets/img/2025-05-07-rlhf-generalization/image1.png" class="img-fluid" %} <div class="caption">Figure 3: RM losses broken down by data size and RM size (from <d-cite key="gao2023scaling"></d-cite>).</div>     

{% include figure.html path="assets/img/2025-05-07-rlhf-generalization/image2.png" class="img-fluid" %} <div class="caption">Figure 4: RM data scaling experiments (from <d-cite key="gao2023scaling"></d-cite>). RM size is held constant (12M), while RM data is varied. The x-axis has a square root scale. Note that the plots have different axes. Dotted lines indicate proxy rewards, solid lines indicate gold rewards.</div>

Additionally, we demonstrate the generalization ability of reward model through our experiments utilizing leetcode datasets. Specifically, we initially gathered 1200 Python prompts from the leetcode website spanning from September 2022 to September 2023 for our training set, and another 474 prompts from September 2023 to June 2024 for our test set. Subsequently, we collected five responses per prompt from various models, including GPT4, GPT4o, Deepseek, among others, for both the training and test sets. Finally, we submitted these responses to the leetcode website to ascertain the ground truth for each response, yielding 30,000 training preference pairs and 12,000 test preference pairs.

As illustrated in the table below, we observe that by training these pairs using pairwise loss, the deepseek 7B code model demonstrates the ability to generalize to pairs that were not seen during the training process.

| Precision on Leetcode Training Dataset | Precision on Leetcode Test Dataset | Best of 10 on Leetcode Training Dataset | Best of 10 on Leetcode Test Dataset |
| --- | --- | --- | --- |
| 98% | 82.5% | 100% | 71.2% |

Furthermore, we have created two preference datasets utilizing MBPP and HumanEval prompts. These datasets also include algorithm prompts that are simpler than those found in LeetCode. Our findings indicate that the reward model trained on the LeetCode preference dataset exhibits a certain level of generalization capability on these other datasets as well.

| Precision on MBPP Dataset | Precision on Humaneval Dataset | Best of 10 on MBPP Dataset | Best of 10 on Humaneval Dataset |
| --- | --- | --- | --- |
| 58.1% | 53.7% | 60.4% | 70.4% |

### Generalization from Pre-trained Model

Generally, the reward model incorporates an additional scalar head that outputs a reward value based on the pre-trained model, such as the deepseek 6.7B code model. Consequently, we speculate that the reward model can inherit the generalization capabilities from the pre-trained LLM, akin to other downstream tasks.

In this blog, we conduct several case analyses to investigate how reward models inherit generalization abilities from pre-trained models. Our findings are as follows:

- Despite the absence of syntactic errors in the negative samples within the training dataset, the reward model easily finds syntactic errors in the responses of the Humaneval and MBPP test datasets. This indicates that the ability to distinguish syntactic errors, which is inherited from the pre-trained model, can be effectively generalized to unseen datasets.
- We trained the reward models utilizing both the pre-trained DeepSeek Coder and a version that had undergone fine-tuning for debug tasks. Our objective was to enable the reward model to acquire the debugging capabilities of the "debug model," thereby assisting it in identifying errors within negative samples. Nevertheless, we discovered that the performance of these two reward models was quite similar, suggesting that the reward model was unable to adequately inherit the debugging skills from the pre-trained debug tasks.

| # Training Samples in Debug Tasks | # Training Samples in Preference Datasets | Precision on Leetcode Test Dataset | Precision on Humaneval Dataset | Precision on MBPP Dataset |
| --- | --- | --- | --- | --- |
| 0 | 12000 | 82.5% | 53.7% | 58.1% |
| 32000 | 12000 | 82.7% | 53.1% | 58.3% |

- The high attention scores observed in the final transformer layer of the LLM for both positive and negative samples do not consistently indicate the presence of "real" errors in the negative samples. This suggests that the patterns captured by the reward model are coarse and cannot reflect specific errors, unlike methods such as GenRM <d-cite key="zhang2024generative"></d-cite> or LLM as a judge <d-cite key="zheng2023judging"></d-cite>. Consequently, we hypothesize that the reward model is unable to explore the pre-trained generalization abilities in the same manner as these two methods.

|  | Test Samples | Syntax Error | Logical Error | Comment Error | Hallucination Problems |
| --- | --- | --- | --- | --- | --- |
| Amount | 200 |  32 | 48 | 90 | 30 |
| Precision | 50.5% | 100.0% | 50.0% | 36.6% | 40% |


As shown in the table above, we tested 200 samples to evaluate how well high attention scores indicate the presence of actual errors in negative samples. We categorized these samples into Syntax Error samples, Logical Error samples, Comment Error samples (where the error is mentioned in the comment following the code), and Hallucination Problems (where non-existent functions are used). However, the precision with which high attention scores accurately pinpoint the presence of real errors is only 50.5%.


## Generalization Progress in PPO

### Generalization from On-policy Samples

Before the occurrence of reward hacking, the reward model provides a satisfactory signal to guide the model in updating its parameters. When a Large Language Model (LLM) generates an on-policy response to a prompt, the reward model assigns a scalar reward, directing the LLM to either increase or decrease the likelihood of producing that specific response. Furthermore, the LLM not only adjusts the probability of the given prompt-response pair but also generalizes this adjustment to unseen pairs, akin to the DPO algorithm <d-cite key="rafailov2024direct"></d-cite>. However, it should be noted that we have not conducted a dedicated experiment to verify this generalization. Instead, we utilized a preference dataset comprising a limited number of prompts (400) to implement both the RFT <d-cite key="wang2024math"></d-cite> and  DPO <d-cite key="rafailov2024direct"></d-cite> algorithms with on-policy preference pairs, and subsequently tested their performance on a small test dataset (100).

|  | Pass@1 of SFT model | Pass@1 of RFT model | Pass@1 of DPO model |
| --- | --- | --- | --- |
| Training set | 45.4 | 55.7 | 64.2 |
| Test set | 20.8 | 23.2 | 27.3 |

From this experiment, we have the following findings:

- The RFT method outperforms the SFT method, indicating that fine-tuning with positive on-policy samples can enhance performance on both training and test datasets. This suggests that the model is capable of extracting more patterns from on-policy samples and generalizing these patterns to unseen samples during training.
- The DPO method on on-policy preference pairs also surpasses the RFT method, demonstrating that fine-tuning with both positive and negative on-policy samples can consistently enhance the model's performance on both training and test datasets.

These findings are consistent with the research from DeepMind, titled "RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold" <d-cite key="setlur2024rl"></d-cite>. 

### Generalization from Token-Wise Reward

PPO trains a value network to compute token-wise rewards, which offer more precise guidance to the policy on how to adjust the probability of each token. To showcase the generalization capability derived from these token-wise rewards, we attempted to utilize the value network from the final PPO iteration for Value-Guided Monte-Carlo Tree Search (VG-MCTS) decoding <d-cite key="liu2022dont"></d-cite>. Specifically, this approach integrates the value network from PPO with the MCTS method to guide the decoding process.

|  | pass@1 of SFT model | pass@1 of PPO model | pass@1 of PPO with  VG-MCTS |
| --- | --- | --- | --- |
| Training set | 45.4 | 71.2 | / |
| Test set | 20.8  | 31.2 | 35.2 |

The experimental results demonstrate that the value network in PPO can provide more precise guidance during the decoding stage. Additionally, it suggests that token-wise rewards play a crucial role in PPO, as they also enable generalization to test samples.

## Data Construction Strategies for RLHF

Based on our analysis of the generalization process in RLHF, we have gained several insights and formulated recommendations for data construction within RLHF:

- As demonstrated in the OpenAI paper, training an effective reward model requires both sufficient preference data and a substantial model size.
- When constructing preference datasets for the reward model, we recommend gathering a diverse range of prompts and multiple on-policy responses for each prompt.
- To fully leverage the generalization capabilities of the pre-trained model, a more thoughtful design for the reward model is necessary. For example, we can generate a substantial amount of synthetic data using techniques such as GenRM or LLM as a judge to pre-train the reward model.
- For constructing the PPO prompt set, we recommend using a wider variety of queries within the reward query domain. This approach will enable the full utilization of the reward model's generalization capabilities.
- To fully utilize the generalization capabilities of the reward model, we recommend using a novel PPO variant called Policy Filtration in RLHF for fine-tuning LLMs <d-cite key="shen2024policy"></d-cite>.

