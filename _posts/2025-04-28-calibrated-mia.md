---
layout: distill  
title: "Reassessing EMNLP 2024’s Best Paper: Does Divergence-Based Calibration for MIAs Hold Up?" 
description: "<strong>TL;DR: No.</strong><br>
A critical analysis of the EMNLP Best Paper proposing a divergence-based calibration for Membership Inference Attacks (MIAs). We explore its experimental shortcomings, issues with temporally shifted benchmarks, and what this means for machine learning awards."
date: 2025-04-28  
future: true  
htmlwidgets: true  
hidden: false  

authors:  
  - name: Anonymous  

bibliography: 2025-04-28-calibrated-mia.bib  

toc:  
  - name: Introduction  
  - name: Method Overview  
  - name: Experimental Evaluation  
    subsections:  
    - name: True Positive Rate Experiment  
    - name: False Positive Rate Experiment  
  - name: The Problem with Temporally Shifted Benchmarks
  - subsections:
    - name: Why These Benchmarks Are Misleading
  - name: Reflections on Machine Learning Awards  
  - name: Conclusion  

---

## Introduction

At EMNLP 2024, the [Best Paper Award](https://x.com/emnlpmeeting/status/1857176180128198695/photo/1) was given to **"Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method"**<d-cite key="zhang2024pretraining"></d-cite>. The paper addresses Membership Inference Attacks (MIAs), a key issue in machine learning related to privacy. The authors propose a new calibration method and introduce **PatentMIA**, a benchmark utilizing temporally shifted patent data to validate their approach.

The method initially seems promising: it recalibrates model probabilities using a divergence metric between the outputs of a target model and a token-frequency map derived from auxiliary data, claiming improved detection of member and non-member samples. However, upon closer examination, we identified significant shortcomings in both the experimental design and evaluation methodology.  In this post, we critically analyze the paper and its broader implications.

---

## Method Overview  

The proposed method tries to fix a known issue with MIAs: models often fail to properly separate member and non-member samples. To address this, the authors use an auxiliary data-source to compute token-level frequencies, which are then used to recelibrate token-wise model logits. This normalization aims to adjust token-level model probabilities based on their natural frequency or rarity, aligning with membership inference practices such as reference model calibration<d-cite key="carlini2022membership"></d-cite>.

They also introduce **PatentMIA**, a benchmark that uses temporally shifted patents as data. The idea is to test whether the model can identify if a patent document was part of its training data or not.  

While this approach sounds interesting, our experiments suggest that the reported results are influenced by limitations in the benchmark design.

---

## Experimental Evaluation  

We ran two key experiments to test the paper's claims: one for true positives and another for false positives.  

### True Positive Rate Experiment  

This experiment checks if the method can correctly distinguish member data from non-member data when both are drawn from the **same distribution**.
We used train and validation splits from **The Pile** dataset, which ensures there are no temporal or distributional differences between the two sets.
Below we report results for the *Wikipedia* split.

| Model              | AUC | TPR@5%FPR |
| :---------------- | :---------: | ----: |
| [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b) |   0.542 | 0.071 |
| [GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125m) | 0.492 | 0.054 |
| [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) | 0.600 | 0.103 |

**Result:**  
The method performed only a bit better than the LOSS attack, and remains comparable to most standalone membership inference attacks. For reference, AUC with the baseline LOSS and zlib <d-cite key="carlini2021extracting"></d-cite> attacks for Pythia-6.9B are 0.526 and 0.536 respectively, while it is 0.618 when using a reference-model (Table 12 in <d-cite key="duan2024membership"></d-cite>). Similarly, using LOSS and zlib yeild AUCs of 0.563 and 0.572 respectively.

Reported improvements in the paper (Table 2 <d-cite key="zhang2024pretraining"></d-cite> showing AUCs of 0.7 and higher) were thus likely due to exploiting differences in the data distribution, rather than actual improvements in detecting membership.  

### False Positive Rate Experiment  

Next, we checked how often the method falsely identifies data as "member" when it has never been part of the training set. To do this, we used the **WikiMIA**<d-cite key="shi2023detecting"></d-cite> dataset but replaced the training data with unrelated validation data from the *Wikipedia* split of **The Pile**. This means that we can say with certainty that the Pythia and GPT-neox models did not train on either split. We follow the experimental setup of in Section 3 of <d-cite key="maini2024llm"></d-cite> for this analysis.

**Result:**  
The method flagged a high number of false positives. It frequently identified non-member data as part of the training set, revealing that it was relying on temporal or distribution artifacts rather than truly detecting membership.  

Below we report results for the *Wikipedia* split. Note that in this setting, a scorecloser to 0.5 is better since both splits are non-members.

| Model              | AUC for DC-PDD <d-cite key="zhang2024pretraining"></d-cite> | AUC for LOSS <d-cite key="carlini2021extracting"></d-cite> |
| :---------------- | :---------: | ----: |
| [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b) |   0.667 | 0.636 |
| [GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125m) | 0.689 | 0.671 |
| [GPT-Neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) | 0.637 | 0.656 |


The results in the table above show that DC-PDD is susceptible to showing high false positives, akin to simple methods like LOSS. Notably, the AUC values in such a false-positive setting are significantly higher than in the previous experiment, indicating that the method clearly captures temporal or distributional artifacts rather than true membership.

---

## The Problem with Temporally Shifted Benchmarks  

The introduction of **PatentMIA** highlights a broader problem with MIA research: benchmarks that rely on temporal shifts. These benchmarks often make it easy for attack models to exploit simple artifacts, like whether a document contains terms that didn’t exist during training (e.g., "COVID-19" or "Tesla Model Y"). This creates an illusion of success but doesn’t address the real challenge of membership inference.  

### Why These Benchmarks Are Misleading  

The issues with temporally shifted benchmarks are not new. Several prior works have already established the dangers of using such benchmarks:  

1. **Spurious Patterns**: Temporal shifts introduce artifacts that are easily exploitable by attack models. As noted by Duan et al. <d-cite key="duan2024membership"></d-cite>, temporal markers (e.g., "COVID-19" or recent events) allow models to cheat by detecting new concepts rather than true membership.  
2. **Misleading Evaluations**: Maini et al. <d-cite key="maini2024llm"></d-cite> show how temporal shifts can inflate the perceived success of MIAs, even when no meaningful membership inference occurs.  
3. **Blind Baselines Work Better**: Das et al. <d-cite key="das2024blind"></d-cite> demonstrate that blind baselines often outperform sophisticated MIAs on temporally shifted datasets, highlighting how these benchmarks fail to test real inference ability.  

Despite these well-established issues, the EMNLP Best Paper continues to rely on temporally shifted data like **PatentMIA** for its evaluations. This undermines the robustness of its claims and contributes little to advancing membership inference research.  

---

## Machine Learning Awards: A Problem of Incentives  

This situation raises important questions about the role of awards in machine learning research.  

1. **Do Awards Encourage Rushed Work?** Highlighting work with known flaws, like relying on misleading benchmarks, can discourage researchers from investing time in more rigorous evaluations.  
2. **Harming the Field**: Awards that celebrate flawed work set a bad precedent and can mislead the community into thinking these methods are the gold standard.  
3. **Losing Credibility**: Over time, the reputation of awards themselves suffers, as researchers may start viewing them as less meaningful.  

If awards are to truly highlight excellence, they must emphasize thoroughness, reproducibility, and robustness over surface-level novelty.  

---

## Conclusion  

The EMNLP 2024 Best Paper sought to address a pressing challenge in membership inference but falls short under careful scrutiny. The proposed method fails both in distinguishing members and non-members under rigorous conditions and in avoiding false positives when the data is untrained. Furthermore, its reliance on **PatentMIA** exemplifies a larger issue with using temporally shifted benchmarks to claim progress.  

For the field to advance meaningfully, greater emphasis must be placed on rigorous evaluation practices. Awards should reflect this by rewarding work with robust and thorough evaluations, rather than methods that (knowingly or otherwise) exploit well-known flaws in evaluation practices. Only then can we ensure that the field moves forward in a meaningful way.
