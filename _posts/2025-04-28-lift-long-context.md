---
layout: distill
title: "LIFT: Improving Long Context Understanding Through Long Input Fine-Tuning"
description: This blog introduces Long Input Fine-Tuning (LIFT) for Long Contexts, a novel framework that enhances LLM performance on long-context tasks by adapting model parameters to the context at test time.
  Experimental results highlight consistent improvements across popular benchmarks, demonstrating LIFT's effectiveness in boosting model performance and memory efficiency.
  This blog also provides a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-lift-long-context.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Challenges in long context understanding for Large Language Models (LLMs)
  - name: 2. Why do we need Long Input Fine-Tuning for long contexts?
  - name: 3. What is Long Input Fine-Tuning (LIFT) for Long Context?
  - name: 4. Experiements and deep analysis
  - name: 5. Limitations and future works
  - name: Appendix
---

# 1. Challenges in long context understanding for Large Language Models (LLMs)

Lengthy sequences, often spanning millions of tokens, are prevalent in real-world applications such as long-form books <d-cite key="kočiský2017narrativeqareadingcomprehensionchallenge"></d-cite>, weeks or months of multi-turn conversations <d-cite key="feng2021sequencetosequenceapproachdialoguestate"></d-cite><d-cite key="wu2024longmemevalbenchmarkingchatassistants"></d-cite><d-cite key="cho2024spectrumspeakerenhancedpretraininglong"></d-cite>, high-resolution video or audio signals <d-cite key="tapaswi2016movieqaunderstandingstoriesmovies"></d-cite>, and multimodal scenarios <d-cite key="zhang2024longclipunlockinglongtextcapability"></d-cite>. Expanding the context window enables models to capture dependencies across broader spans of text, enhancing coherence, comprehension, and accuracy in tasks that demand reasoning over extended content.

The limited context window of LLMs often hampers their ability to fully grasp a user's query or task input and deliver optimal performance. Modern long context LLMs have extended the size of their context windows to 128K, 200K, or even 2M tokens. However, this introduces several challenges:

- The computational complexity of the self-attention mechanism grows quadratically with the context length, restricting the model's capacity to handle extended texts efficiently.

- Managing the storage of vast attention weights and intermediate states places a significant strain on hardware resources.

- Additionally, capturing long-range dependencies across dispersed information in raw texts remains challenging, making advanced comprehension and reasoning even more difficult.

# 2. Why do we need Long Input Fine-Tuning for long contexts?

Various techniques have been developed to extend the sequence length that LLMs can handle, including efficient transformers, external memory mechanisms, and recurrent memory structures<d-cite key="lu2024controlled"></d-cite> <d-cite key="wang2024beyond"></d-cite> <d-cite key="huang2023advancing"></d-cite> <d-cite key="xiong2023effective"></d-cite>. Among these, two prominent approaches have gained significant attention: *Retrieval-Augmented Generation (RAG)* <d-cite key="lewis2020retrieval"></d-cite> <d-cite key="xu2023retrieval"></d-cite> <d-cite key="jiang2024longrag"></d-cite> and *Fine-tuning for Longer Context Windows*<d-cite key="chen2023longlora"></d-cite><d-cite key="peng2023yarn"></d-cite>:

- *RAG* stores the extra knowledge in **external databases** and retrieves relevant knowledge on need. However, it is prone to information loss when conditioned on inaccurate retrievals and may produce hallucinations if the retrieved context contains significant noise.

- *Fine-tuning for longer contexts* extends the context window of a pre-trained model and fine-tunes on long documents for adaptation. This method still stores the extra knowledge **in context**. While effective, the inference can become highly inefficient due to processing the entire lengthy context. Moreover, no matter how much the context window length of an LLM is extended, there will still be scenarios involving texts too long to process in context. 

In this blog, we explore a third way to store the extra knowledge in the lengthy input. Instead of relying on external databases or extended context window, we propose to leverage **in-parameter knowledge** by
directly **fine-tuning the long input into the model parameters** and using the fine-tuned model for inference. 

Our novel framework, **Long Input Fine-Tuning (LIFT)**, is designed to enhance the long-context capabilities of any short-context model. It offers the following advantages:

1. **On-the-fly long-context training.** The framework dynamically adapts model parameters to process lengthy inputs as needed, avoiding the resource-intensive processes of offline fine-tuning for long-context models and inference on the entire long input. 

2. **Unlimited input length.** No matter how long the input is, our framework can fine-tune the knowledge into the parameters by segmenting the input into partially overlapping blocks and performing language modeling on them in parallel. Thus, it eliminates the input length restrictions associated with in-context learning (ICL).

3. **Significant improvement in long-context tasks.** Our evaluations across various benchmarks demonstrate that LIFT greatly benefits tasks such as summarization and complex scenarios like timeline reordering and reading comprehension. Moreover, incorporating supervised fine-tuning prior to applying LIFT further enhances the model's performance on downstream tasks.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/figure1_method.svg" class="img-fluid" title="Figure 1" %}
<div class="caption">Figure 1. Overview of our method compared with existing methods like truncation, RAG, long context fine-tuning.</div>

# 3. What is Long Input Fine-Tuning (LIFT) for Long Context?

As discussed earlier, our method features fine-tuning and inference with only a short-context model, ensuring high efficiency. In this section, we present how we implement LIFT and address the associated challenges.

- In Section 3.1, we introduce our basic setup, which involves training on segments of long input.
- In Section 3.2, we compensate for potential capability loss and enable the model to perform reasoning over long input by incorporating auxiliary tasks (AT) during fine-tuning.
- In Section 3.3, we further refine the model by supervised fine-tuning it on a diverse set of long documents and synthetic tasks, making it familiar with our LIFT paradigm and adapts to new long texts better.

## 3.1. Training with input segments

LLMs access knowledge either from contexts or their parameters. Unlike ICL, we propose storing test-time knowledge in the parameters by fine-tuning the model on the given long input.


We formalize memorizing the input as a language modeling task. Let the input be $$\mathbf{x}=(x_ {1},x_ {2},\dots,x_ {L})$$, where $$L$$ is a very large number. The objective function for the language modeling task is defined as

$$
\mathcal{L}_ {LM}(\mathbf{x};\theta)=\sum_ {i=1}^{L}\log\mathbb{P}(x_ {i}\mid \mathbf{x}_ {1:i-1};\theta),
$$

where $$\theta$$ is the parameters.

However, directly fine-tuning the model on a long text of length $$L$$ incurs a computational complexity of $$\mathcal{O}(L^{2})$$ and becomes infeasible when the base model has a context window shorter than $$L$$. A straightforward approach is to truncate $$\mathbf{x}$$ into non-overlapping short segments, denoted as $$\mathbf{x}_ {l_ {1}:r_ {1}},\dots,\mathbf{x}_ {l_ {K}:r_ {K}}$$, as illustrated in Figure 2. The objective function for the language modeling task with the short segments is expressed as

$$
\mathcal{L}_ {input}(\mathbf{x};\theta)=\sum_ {k=1}^{K}\mathcal{L}_ {LM}(\mathbf{x}_ {l_ {k},r_ {k}};\theta).
$$

Is there any information lost? Yes, **the sequential order of the segments** is lost. Since there is no overlap between the adjacent segments, the model cannot infer the correct order of the segments.

To address this issue, we propose an intuitive solution: introducing overlaps between the adjacent segments, as illustrated in Figure 2. By overlapping the tail of one segment with the head of the next, the model can better retain the sequential order of the context. Ideally, if the model learns to generate the tail of one segment, it can continue to recite the next segment. Formally, we design that

$$
\begin{aligned}
&l_ {1}=1,r_ {K}=L,\\
&\forall i=1,2,\dots,K-1, r_ {i}-l_ {i}+1=\ell,l_ {i+1}=l_ {i}+s.
\end{aligned}
$$

Here $$s$$ controls the length of the overlaps. Empirically, taking $$s=\frac{3}{8}\ell$$ proves sufficient in our experiments, which introduces only constant computational complexity overhead.

## 3.2. Training with auxiliary tasks

Fine-tuning a pretrained LLM for a specific task risks damaging its other capabilities. Similarly, while fine-tuning on the input helps the model memorize the input, it probably degrades other abilities, such as instruction-following. Moreover, effectively memorizing the long input doesn't mean the model can reason based on it.

To mitigate potential capability loss and enable the model to reason based on the long context, we propose synthesizing auxiliary question-answering (QA) tasks, denoted as $$(\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m}$$, based on the long context. The objective function of the auxiliary tasks is defined as

$$
\mathcal{L}_ {AT}((\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m};\theta)=-\sum_ {i=1}^{m}\log\mathbb{P}[\mathbf{a}_ {i}\mid\mathbf{x}_ {i};\theta].
$$

Following the mechanism of mix training <d-cite key="AllenZhu-icml2024-tutorial"></d-cite>, which asserts that LLMs can only learn to perform inference based on $$\mathbf{x}$$ when trained simultaneously on both $$\mathbf{x}$$ and $$(\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m}$$, we propose jointly optimizing the two objective functions, i.e.,

$$
\mathcal{L}(\mathbf{x},(\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m};\theta)=\mathcal{L}_ {input}(\mathbf{x};\theta)+\gamma\cdot\mathcal{L}_ {AT}((\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m};\theta).
$$

There are no strict constraints on the method used to synthesize $$(\mathbf{q}_ {i},\mathbf{a}_ {i})_ {i=1}^{m}$$ based on $$\mathbf{x}$$, except that is should avoid computationally expensive operations on $$\mathbf{x}$$, such as inference over the entire $$\mathbf{x}$$. In our experiments, we extract several short segments from $$\mathbf{x}$$ and use a pretrained LLM to generate QA pairs based on the segments.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/figure2_segmentation.svg" class="img-fluid" title="Figure 2" %}
<div class="caption">Figure 2. Comparison between our segmentation method and the trivial segmentation method.</div>

## 3.3. Further improvement with pre-LIFT Supervised Fine-Tuning

While our framework LIFT is applicable to any model capable of fine-tuning, we suggest that pretrained LLMs may be unfamiliar with our training method, which leads to suboptimal results. We hypothesize that performance on downstream tasks can be enhanced by learning a new set of parameters through multiple rounds of LIFT with auxiliary tasks, a process commonly known as **Supervised Fine-Tuning (SFT)**, which has been shown to be effective for long-context downstream tasks <d-cite key="beltagy2020longformerlongdocumenttransformer"></d-cite><d-cite key="zaheer2021bigbirdtransformerslonger"></d-cite>. Based on this SFT model, we will then apply the normal LIFT process to further fine-tune the model on the given test input.

The SFT process involves training the model on a large corpus of long texts, combined with QA tasks synthesized based on the corpus. **To ensure the model becomes familiar with our LIFT method, the supervised fine-tuning (SFT) tasks are designed to closely resemble those used in our LIFT framework.** Unlike our main approach, where the model is fine-tuned on a single piece of long text, the SFT phase involves fine-tuning on multiple pieces of long text simultaneously, preventing it from overfitting.

Formally, we select the corpus $$(\mathbf{x}^{(i)})_ {i=1}^{N}$$ independent of the test datasets. For each $$\mathbf{x}^{(i)}$$, we synthesize a set of QA tasks $$(\mathbf{q}_ {j}^{(i)},\mathbf{a}_ {j}^{(i)})_ {j=1}^{K}$$. The objective function for SFT is defined as

$$
\mathcal{L}_ {SFT}\Big(\big(\mathbf{x}^{(i)}, (\mathbf{q}_ {j}^{(i)},\mathbf{a}_ {j}^{(i)})_ {j=1}^{K}\big)_ {i=1}^{N};\theta\Big)=\frac{1}{N}\sum_ {i=1}^{N}\left(\mathcal{L}_ {input}(\mathbf{x}^{(i)};\theta)+\gamma\cdot\mathcal{L}_ {AT}((\mathbf{q}_ {j}^{(i)},\mathbf{a}_ {j}^{(i)})_ {j=1}^{K};\theta)\right).
$$

# 4. Experiements and deep analysis

## 4.1. Results on popular long context benchmarks
To evaluate our method, we choose two models for evaluation:
- open-sourced model *LLaMA-3-8B-Instruct*<d-cite key="llama3modelcard"></d-cite> with **8k** context window
- close-sourced model *GPT3.5-turbo*<d-cite key="chen2023robust"></d-cite> with a longer context window **16k**


We compare the following methods:
- ICL with truncation (noted as *ICL*), where we truncate the input by only keeping its beginning and end tokens to maximally fill the context window, and use the original LLM.
- LIFT without ICL (noted as *LIFT_only*), where we use the LIFT LLM without filling any input into the context window.
- LIFT with ICL (noted as *LIFT+ICL*), where we use the LIFT LLM and additionally fill the beginning and end tokens of the input tokens into the context window.

By default, LIFT does not use auxiliary tasks (AT) and SFT and only fine-tunes the model on the input text.


We evaluate our methods on four popular long-context benchmarks *LooGLE* <d-cite key="li2023loogle"></d-cite>, *LongBench* <d-cite key="bai2023longbench"></d-cite>, *Bamboo* <d-cite key="dong2023bamboo"></d-cite> and *QuALITY* <d-cite key="pang2021quality"></d-cite>. They provide a relatively comprehensive evaluation, covering a wide variety of application scenarios. The evaluation metrics are task-specific and consistent with the respective original benchmarks. Among these, the GPT-4 score evaluates the correctness of responses of LLMs given corresponding questions and answers with GPT-4<d-cite key="openai2024gpt4technicalreport"></d-cite>, which is proven to be highly aligned with human evaluations.

### LIFT generally strengthens long context understanding when combined with ICL

For LooGLE, as shown in Table 1, **LIFT+ICL consistently achieves the highest scores** across both LongQA and shortQA tasks for both models, and is particularly effective in the ShortQA task, which doesn't rely on long dependencies. Interestingly, **LIFT_only performs the worst** among all the settings. We observe that LIFT_only struggles to retrieve the original input. While overfitting the model on the input enables the model to precisely memorize the input, it may degrade the original capabilities of the model such as retrieving certain in-parameter knowledge, resulting in even worse performance. In this sense, LIFT and ICL might provide a reference point to each other to help them locate the relevant knowledge better, which combines the strengths of both in-parameter knowledge (full) and in-context knowledge (partial) and results in the best performance.

Compared to GPT-3.5, LLaMA-3 benefits more from LIFT+ICL, showing notable improvement in GPT4_score: from 30.88 (ICL) to 33.42 in LongQA, and from 44.23 to 50.44 in ShortQA. These results highlight that **LIFT significantly improves the performance of ICL only, particularly for models with shorter context windows**.

Notably, all models perform particularly poorly on LongQA, with accuracy falling below 50%. This underscores that modeling long dependencies in extended contexts remains a significant challenge for existing models.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table1_loogle.svg" class="img-fluid" title="Table 1" %}
<div class="caption">Table 1. Performance on LooGLE under different settings</div>

For LongBench, as shown in Table 2, LLaMA-3 exhibits substantial improvement when integrating ICL and LIFT, reaffirming that LIFT is particularly beneficial for models with shorter context lengths. As expected, GPT-3.5, with longer context window and superior ICL capabilities, outperforms LLaMA-3 on all tasks except GovReport.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table2_longbench.svg" class="img-fluid" title="Table 2" %}
<div class="caption">Table 2. Performance on LongBench under different settings</div>

### LIFT shows significant improvement on specific tasks

Table 3 presents detailed experimental results across four LongQA tasks from LooGLE. LLaMA-3 benefits more from LIFT+ICL, with notable improvement in specific tasks, particularly in **Comprehension & Reasoning** (40.88 → 44.83) and **Timeline Reorder** (22.33 → 26.51). These results demonstrate that LIFT enhances ICL by providing a more comprehensive overview of the entire lengthy input by fine-tuning the long input directly into the parameters.

However, LIFT does not yield any improvement in tasks such as Multiple Information Retrieval and even slightly degrades the performance in Computation for both models. This indicates that LIFT may not significantly impact all tasks and could introduce slight noise in some cases. At the same time, the performance variations are closely related to the specific capabilities required by each task and the inherent strengths of the model.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table3_loogle_task.svg" class="img-fluid" title="Table 3" %}
<div class="caption">Table 3. Performance of each task in LongQA for LLaMA-3</div>

In Table 2 on LongBench, LIFT+ICL consistently outperforms both ICL and LIFT_only on Narrativeqa and Qmsum for both models. Notable improvement is observed in Narrativeqa, where performance increases from 20.73 to 25.84. However, the results for Musique and GovReport exhibit different trends between the two models. LLaMA-3 shows a slight improvement on GovReport but experiences a significant drop on Musique, whereas GPT-3.5 demonstrates the opposite pattern.

Interestingly, PassageRetrievalEN exhibits a significant drop when combining LIFT with ICL, suggesting that the LIFT is not effective to specific tasks. This motivates us to fine-tune the model at task level.

## LIFT can be further enhanced with pre-LIFT supervised fine-tuning

Encouraged by the significant improvement observed in the timeline-reorder task from LooGLE, we aim to further enhance the performance of LIFT on similar tasks like sorting and reordering, by incorporating auxiliary tasks (AT) (Section 3.2) and pre-LIFT SFT (Section 3.3). For AT, we generate synthetic QAs according to the input text simliar to the target task and fine-tunes the model on both the input text and the QAs. For SFT, we generate synthetic QAs on independent corpus and fine-tune the model on the corpus and QAs before applying LIFT on specific inputs. 

The results are illustrated in Table 4. There are six models compared:
- ICL and LIFT+ICL are the same as before;
- LIFT+AT+ICL means fine-tuning on both input text and synthetic QAs during the LIFT phase;
- SFT+ICL, SFT+LIFT+ICL and SFT+LIFT+AT+ICL mean using the SFT model rather than the original LLM for the previous three baselines.

Comparing the results of LIFT+ICL and LIFT+AT+ICL, as well as SFT+LIFT+ICL and SFT+LIFT+AT+ICL, we observe that AT brings negligible improvement or even slightly degrades performance for the LIFT phase. A possible explanation is that the number of synthesized samples in our evaluation is insuffcient, potentially causing the model to **overfit these specific examples instead of enhancing the general ability**. However, it's impractical to synthesise a huge number of training samples at test time due to unacceptable computational cost. Striking a balance between efficiency and effectiveness when using AT at test time remains a significant challenge and requires further exploration.

In contrast, we find SFT greatly improves the performance of both ICL and LIFT+ICL, which is reasonable since the tasks used in the SFT process are similar to those at test time. With SFT, LIFT+ICL is still better than ICL only, highlighting the effectiveness of our method.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table4_at_sft.svg" class="img-fluid" title="Table 4" %}
<div class="caption">Table 4. Coordinate score<d-cite key="dong2023bamboo"></d-cite> on specific task in Bamboo, LooGLE, and QuALITY using AT and SFT.</div>

## 4.2. Efficiency evaluations

Benefiting from our truncation strategy (Section 3.1), the computational complexity of our method scales linearly with the input context length (due to fine-tuning). To further evaluate the efficiency of our approach compared to ICL, we measure the time cost of a single Needle-In-A-Haystack (NIAH) task under both methods. In this experiment, the input lengths are controllable and the primary computational cost stems from processing the input context rather than iterative generation.

We plot the GPU time against the input length along with the fitted curves in Figure 3.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/figure3_efficiency.svg" class="img-fluid" title="Figure 3" %}
<div class="caption">Figure 3. GPU time vs. input length for LIFT and ICL. The dashed lines represent the fitted curves, showing linear growth for LIFT and quadratic growth for ICL. The red cross indicates the input length at which ICL runs out of memory.</div>

First, we observe that LIFT is significantly more **memory-efficient** than ICL. Notably, ICL runs out of memory when the input length exceeds 90k tokens on our A100 (80G) system. Upon closer inspection, we find that the cache of hidden states for previous tokens consumes most of the memory in ICL. In contrast, LIFT is capable of **handling arbitrarily long inputs**. Our truncation strategy ensures that LIFT only involves fine-tuning and inference on short text segments, eliminating the need for extensive caching.

Empirically, we find that the time cost of ICL grows quadratically with input length, while our method scales linearly. However, we also observe that the constant factor introduced by fine-tuning in the computational complexity of LIFT is non-negligible. As a result, our method only surpasses ICL in time efficiency when the input length exceeds a certain threshold above 200k tokens. The primary cost of our method arises from the multi-epoch fine-tuning. We hypothesize that by using better parallel fine-tuning techniques and designing tasks that are more aligned with the strengths of LIFT, the efficiency of the LIFT framework can be significantly improved.

# 5. Limitations and future works

1. **Limitations of LIFT without ICL**\\
  While we often employ truncated contexts to simplify inference on lengthy texts, this approach is proven insufficient for tasks that demand precise information extraction from extended contexts, such as the Needle in a Haystack (NIAH) task. Despite the practical value of NIAH is arguable, we still perform the experiments and show the results in Appendix A. For NIAH tasks, LIFT_only is insufficient and ICL using a long context seems indispensable.

2. **More advanced LIFT methods**\\
  We introduce an intuitive strategy, LIFT, for handling long contexts, showcasing its potential to address challenges associated with lengthy inputs. However, pretrained LLMs may not be naturally familiar with LIFT framework. To bridge this gap, we introduce pre-LIFT SFT, but our vision is to generalize the LIFT framework to any pretrained LLM, enhancing its flexibility and adaptability without requiring extensive retraining. This still needs extensive future study.

3. **Strategy to extract parametric knowledge after LIFT**\\
  Through LIFT, embedding the inputs into the model's internal parameters enhances its familiarity with the inputs. However, the effectiveness of downstream tasks still depends on the model's ability to autonomously extract and utilize the parametric knowledge gained during LIFT. Our experiments (Appendix B) reveal that explicitly providing task-relevant knowledge outperforms using LIFT alone. Furthermore, supplying task-relevant knowledge to the model after applying LIFT still significantly improves the performance. This underscores the potential of developing strategies to effectively trigger and leverage LIFT-acquired knowledge for downstream tasks (such as using RAG), making it a promising direction for further research and exploration.

4. **Challenges using LIFT with auxiliary tasks**\\
  Our findings reveal that auxiliary tasks during LIFT offer minimal benefit and can even degrade performance due to overfitting. Additionally, simply fine-tuning the model on long texts does not inherently endow it with robust reasoning capabilities over such texts. These observations underscore the necessity for more effective strategies to harness the in-parameter knowledge of LLMs, enabling them to reason efficiently and accurately over extended contexts.

LIFT is a fascinating concept because humans similarly transform short-term memory into long-term memory, much like LIFT converts in-context knowledge into in-parameter knowledge. While LIFT is far from fully addressing the challenging long-context problem in LLMs, our preliminary results suggest it offers a promising and exciting direction for further research and investment. We encourage the community to explore LIFT with broader training corpora, diverse models, advanced auxiliary task designs, and greater computational resources.

# Appendix

## A. Results on Needle-in-a-Haystack (NIAH)

We present the experimental results in the NIAH<d-cite key="niah"></d-cite> task in Figure 4, as further analysis of the pros and cons of LIFT and directions for future works. The task requires accurate retrieval from the contexts. We adopt a strong long-context model, LLaMA-3.1-8B-Instruct, as the baseline and apply the LIFT framework to the model.

The maximum context length of our test is 100K, which is within the 128K context window of LLaMA-3.1-8B-Instruct. As expected, the baseline achieves nearly perfect performance. However, LIFT slightly degrades the performance and the degradation seems irregular.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/figure4_1_Needle-Long-Baseline.svg" class="img-fluid" title="Figure 4(1)" %}
{% include figure.html path="assets/img/2025-04-28-lift-long-context/figure4_2_Needle-Long-TTT.svg" class="img-fluid" title="Figure 4(2)" %}
<div class="caption">Figure 4. Performance on NIAH: ICL (top) vs. LIFT (bottom)</div>

The reason for the degradation may be that LIFT introduces more noise to the model. While most parts of the context are irrelevant to the answer, LIFT asks the model to memorize all the context. The model is likely to be misled by the large amount of irrelevant information.

As summarized in Section 5, precise memorization can be challenging for LIFT. On the one hand, LIFT can't accurately memorize the context while avoiding overfitting. On the other hand, LIFT is likely to be misled when most information is irrelevant to the answer. Future works may improve the LIFT framework from these two aspects.

## B. LIFT can perform much better with extracted evidence

For a task in LooGLE, the relevant evidences are provided as a sequence of multiple relevant information retrieved from long context for further computation, reorder, reasoning and comprehension to obtain the final answer.

We make further studies on whether extracting relevant evidence can further enhance the long context understanding after LIFT. In Table 5, it highlights the effectiveness of integrating evidences and combining it with LIFT in greatly improving the model's performance, which leaves space for futher enhancement on the strategy of LIFT. While LIFT alone provides modest improvements, the most substantial gains are observed when evidences are integrated into the ICL process, either with or without LIFT.

Table 4 further expands the performance in Table 3 on specific tasks in LongQA in LooGLE. The combination of evidences, LIFT+ICL clearly outperforms the other configurations across all metrics, **highlighting the importance of extracting relevant knowledge from parameters and executing explicit step-by-step reasoning in more complex tasks like long dependency QA**. The incorporation of evidences helps the model ground its inferences resulting in a more refined and contextually accurate response generation.

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table5_loogle_evd.svg" class="img-fluid" title="Table 5" %}
<div class="caption">Table 5. Performance with extracted evidence of LLaMA-3 in LongQA</div>

{% include figure.html path="assets/img/2025-04-28-lift-long-context/table6_loogle_task_evd.svg" class="img-fluid" title="Table 6" %}
<div class="caption">Table 6. Performance with extracted evidence of each task in LongQA for LLaMA-3</div>
