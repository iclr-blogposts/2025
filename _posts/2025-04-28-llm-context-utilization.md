---
layout: distill
title: Measuring Context Utilization on Recent Open-Source Long Context LLMs
description: This blog post aims to evaluate how well the most capable open-source long context large language models (LLMs) utilize context, using the Needle In A Haystack test. We focus on the task of chapter summarization for recently published books to minimize data contamination while ensuring a challenging test. Our discussion highlights the results of the test conducted on the Llama 3.1 70B Nemotron Instruct model, revealing performance variations across different context lengths and needle placement depths.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-llm-context-utilization.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Why the NIAH Test?
  - name: Reasons for Using Chapter Summarization in NIAH Test
  - name: Experiment Setup
  - name: NIAH Visualization
  - name: In-depth Analysis
  - name: Comparison with GPT-4o-mini at 128k Context Length
  - name: Conclusion
  - name: Book List

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
---

## Introduction

The release of the Llama 3.1 model family <d-cite key="dubey2024llama3herdmodels"></d-cite> has been a game-changer for the open-source LLM community. These models support context lengths of up to 128k tokens, on par with the most capable [GPT-4o models](https://openai.com/index/hello-gpt-4o/). Their potential is further enhanced by improved fine-tuning techniques, for instance, the Nemotron reward modelling approach <d-cite key="wang2024helpsteer2preferencecomplementingratingspreferences"></d-cite> powered the fine-tuned 70B model to rank 10th on the [LLM Arena Leaderboard](https://lmarena.ai/) as of November 2024. 

Nevertheless, accepting 128k tokens as input does not mean that the model can utilize them effectively. Many long context LLMs published previously did not perform well on the Needle In A Haystack (NIAH) test, showing significant performance differences across numerous context lengths and document depths when asked to retrieve a piece of information from within the context. In this blog post, we investigate the context utilization capabilities of the Llama 3.1 70B Nemotron Instruct model using an enhanced version of the NIAH test, focusing on chapter summarization for recently published books. Our results show that the 70B model maintains decent performance up to 48k tokens but gradually deteriorates as the context length increases further. At 128k tokens, the model struggles to retrieve the correct chapter from almost all positions, except when the chapter is located at the end of the context. We also test the same setup on the GPT-4o-mini model and observe that the 70B model still exhibits a significant performance gap compared to capable closed-source long context LLMs.

## Why the NIAH Test?

The context length of LLMs has increased dramatically over the years, evolving from 4k tokens in the initial [GPT-3.5 release](https://openai.com/index/chatgpt/) to 2M tokens in [Gemini 1.5 Pro](https://ai.google.dev/gemini-api/docs/long-context). Simultaneously, many benchmarks have been proposed to evaluate whether long context LLMs can effectively handle their advertised context lengths. These benchmarks typically focus on tasks such as document question answering, retrieval and summarization <d-cite key="wang2024novelqa"></d-cite> <d-cite key="zhang-etal-2024-bench"></d-cite>. It is undeniable that these benchmarks pose a significant challenge even to the most capable models. For instance, GPT-4o scores only 53.5 out of 100 on the Loong benchmark <d-cite key="wang-etal-2024-leave"></d-cite>.

Although existing benchmarks have attempted to include questions requiring aggregation of information distributed across the context, they are still insufficient for conducting fine-grained analysis of context utilization. Specifically, we aim to quantify which parts of the context the model focuses on when answering a query, helping us understand whether models can truly identify the key information needed for an accurate response. Fortunately, the Needle In A Haystack (NIAH) test provides an effective framework for such analysis. The simplest version of the test places a unique piece of information within a long, irrelevant context. After the context is presented to the model, a question is asked about this piece of information. A correct response indicates that the model successfully retrieved the key information; an incorrect response suggests otherwise.

Despite its simplicity, many previous long context LLMs struggled with this test. For instance, GPT-4-1106 128K achieved only 87.1% overall accuracy <d-cite key="fu2024dataengineeringscalinglanguage"></d-cite>. While long context LLMs continue to improve in context utilization, more complex versions of the NIAH test have been developed to further stress-test these models. These versions include placing multiple pieces of key information within the context <d-cite key="li2024needlebenchllmsretrievalreasoning"></d-cite>, increasinng task difficulty <d-cite key="roberts2024needlethreadingllmsfollow"></d-cite>, etc. These more complex tests revealed that the most recent and capable LLMs still have shortfalls in context utilization. 

Context utilization remains a significant challenge yet to be effectively addressed, even for the most capable LLMs. In this blog post, we examine one of the best long context LLMs from the open-source community to evaluate how it compares to the most advanced closed-source models in terms of context utilization. Using the challenging task of chapter summarization for recently published books, we aim to uncover the limitations of the 70B Nemotron Instruct model, [nvidia/Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF). By analyzing the results of this test and comparing them to those of more advanced closed-source models, we hope to highlight key areas for improvement in the development of more robust open-source long context LLMs.

## Reasons for Using Chapter Summarization in NIAH Test

Existing NIAH tests typically involve retrieving and answering questions about key information embedded in the context. Instead of adhering to this traditional paradigm, we adopt the task of summarizing chapters from books published in 2024 as the backbone for our NIAH test. Compared to previous approaches, our method offers the following advantages:

1. **Mitigating data contamination.** Llama 3.1 models have a knowledge cutoff of December 2023, meaning their pre-training data likely includes most publicly available texts on the internet dated before 2024. Consequently, using texts from before 2024 introduces a risk of data contamination, making it difficult to determine whether the model is utilizing the provided context or simply recalling pre-training knowledge. While finding perfectly clean data is challenging, we aim to minimize this risk by exclusively using books published in 2024 to build our test set.

2. **Direct visualization of key information retrieval.** The purpose of NIAH tests is to evaluate whether a model can retrieve relevant information from the context. Existing NIAH tests primarily focused on question answering tasks with short answer lengths. However, when an answer is incorrect, it is unclear whether the model failed to retrieve the key information entirely or if it successfully identified the information but failed to answer the question correctly. The latter could be due to the complexity of the question or interference from the surrounding context. 

In our chapter summarization task, we can directly compare the relevance of the model's summary to the chapter text to determine whether the summary is entirely, partially, or not at all derived from the chapter text. Since summaries are much longer than short answers, it becomes more straightforward to identify whether the model has retrieved the key information. If the summary includes characters or events present in the chapter, it can be considered a successful retrieval, and vice versa. Additionally, this approach provides insights into the model's ability to summarize long texts. By using summarization metrics, we can evaluate the quality of the summaries, either by comparing them to reference summaries or directly against the chapter texts.

## Experiment Setup

Finding comprehensive, high-quality chapter summaries for recent publications can be challenging. To ensure the evaluation is conducted with high-quality data, we use chapter summaries collected from [SuperSummary](https://www.supersummary.com/) consisting of 10 books published in 2024 (Full book list provided [below](#book-list)). All chapter summaries are manually curated by domain experts <d-footnote>https://www.supersummary.com/ai-powered-content-disclaimer/</d-footnote>. We sample 100 chapter summaries with their corresponding chapter texts as our test set, with an average chapter length of 3,032 tokens and an average summary length of 183 tokens. Following the evaluation format used in LLMTest_NeedleInAHaystack <d-cite key="kamradt2023needle"></d-cite>, we set the context length to range from 16k to 128k tokens, with a step size of 16k, and the needle position to range from 0% to 100%, with a step size of 10%. This yields a total of 8,800 test cases. For needle prompt engineering, we place an instruction before the chapter text and include a 6-letter identifier that is unique within the entire context, ensuring the model can distinguish the chapter text from the rest of the context. An example of the needle prompt is shown below:

```txt
Below is a chapter from a book with ID: fbnmkd, marked by the <chapter> tags.
<chapter>
{}
</chapter>
```

In terms of evaluation metrics, we first assess the relevance between the 70B model's summary and the chapter text using the GPT-4o model. We ask the model to classify the relevance into three categories: 1) Irrelevant, where the summary is completely unrelated to the chapter text; 2) Partially relevant, where the summary contains some information from the chapter text but also includes irrelevant details; 3) Fully relevant, where the summary closely follows the chapter text. Additionally, we evaluate the 70B model's ability to generate summaries based on the chapter text using automatic summarization metrics, including AlignScore for factual consistency with the chapter text and BERTScore for semantic similarity with reference summaries. Finally, we also evaluate the GPT-4o-mini model on the full 128k context length tests to compare the performance of the 70B model to that of a capable closed-source model with the same context length.

## NIAH Visualization

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/relevance.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 1:</b> Relevance scores of the 70B model across different context lengths and needle positions..
</div>

Figure 1 shows the relevance scores of the 70B model across different context lengths and needle positions. We observe that the model maintains a relevance score close to 1 at context lengths up to 48k tokens but gradually deteriorates as the context length increases further. In terms of document depth, performance starts to decline rapidly at depths between 10% and 30% when the context length reaches 64k tokens, ultimately dropping to near 0 at depths between 0-50% and 80% for the full 128k context length. 

Traditionally, when models are asked to summarize documents, the output summaries tend to exhibit "lead bias" that focuses more on the first few segments of the entire context <d-cite key="ravaut-etal-2024-context"></d-cite>. Existing work that directly measures context utilization also found the "lost in the middle" problem, where models perform better at retrieving key information from the beginning and the final parts of the context compared to the middle <d-cite key="liu-etal-2024-lost"></d-cite>. In comparison to previous work, our results demonstrate a "lost in the beginning" pattern, where the model performs particularly poorly when the key information is placed in the initial parts of the context. Performance improves in the later parts of the context, especially toward the end, where the relevance score approaches 1 for all context lengths. 

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/alignscore.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 2:</b> (Top) AlignScore results (Bottom) BERTScore results.
</div>

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/bertscore.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

In terms of summarization metrics (Figure 2), we observe a strong correlation with the relevance scores shown in Figure 1. This is unsurprising, as the summary quality is unlikely to be good if the model cannot even identify the source text. However, we can see that the range of both scores is much smaller than the relevance scores. While relevance scores range from 0 to 1, both BERTScore and AlignScore are roughly in the range of 0.4 to 0.7.

## In-depth Analysis

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/context_proportion.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 3:</b> Proportions of irrelevant, partially relevant and fully relevant summaries at different context lengths.
</div>

To gain a better understanding of the performance differences across various context lengths, we examine the proportions of irrelevant, partially relevant, and fully relevant summaries at different context lengths. From Figure 3, we can see that the model maintains robust performance (>94% correct responses) up to 48k tokens, suggesting strong capability in handling moderate-length contexts. However, this performance begins to decline abruptly as the context length increases, with notable drops around 48k-64k tokens where relevant cases decrease rapidly. The decline becomes particularly pronounced after 96k tokens, where correct responses drop below 50%, ultimately reaching just 19% at 128k tokens. This pattern, coupled with the corresponding exponential rise in incorrect responses at longer context lengths, indicates a systematic degradation in the model's ability to identify key information as the context length increases.

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/depth_proportion.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 4:</b> Proportions of irrelevant, partially relevant and fully relevant summaries at different document depths.
</div>

Conducting the same analysis on document depth, we observe that the distribution is more uniform compared to the distribution of context length. The model is able to generate over 60% of relevant summaries at all depths except for the 10%-30% range. We also observe an steady increase in performance towards deeper depths starting from 30%, where the proportion of fully relevant summaries climbs from 49.8% at 30% depth to 97.8% at 100% depth. Interestingly, the model has a 28.4% error rate at the beginning of the document, which is smaller than the error rates at depths of 10%-50%, but larger than those at the remaining depths. This distribution can be explained by the "lost in the beginning" pattern observed in Figure 1, but also resembles a "lost in the middle" pattern skewed toward the beginning of the document. Finally, the proportion of partially relevant summaries is evenly distributed across all depths (<5%), in contrast to the distribution over context length in Figure 3, where the majority of partially relevant summaries are found at high context lengths. This balanced distribution of uncertain responses suggests that the model maintains consistent calibration regardless of document depth, avoiding confusion between irrelevant context and key information.

## Comparison with GPT-4o-mini at 128k Context Length

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/gpt4o_70B_comp.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 5:</b> Metric comparison between the 70B Nemotron model and the GPT-4o-mini model at 128k context length.
</div>

All analysis up to this point has been based on the nvidia/Llama-3.1-Nemotron-70B-Instruct model only. To understand how it compares to the most capable closed-source models, we run the full 128k context length test on the GPT-4o-mini model to see if any performance gap exists. Figure 5 shows the comparison between the two models across all metrics. We can immediately observe that the 70B model experiences a drastic drop in relevance scores. While GPT-4o-mini achieves near perfect relevance scores at several depths, the 70B model is only able to match the performance at the end of the context. Although the GPT-4o-mini model shows a similar pattern to the the 70B model, with performance at 10-30% depths lagging behind other depths, it still maintains a relevance score above 0.8, while the 70B model drops to nearly 0 at the same depths. The curve of GPT-4o-mini is also more stable compared to the 70B model, where the latter oscillates wildly starting from the 60% depth onward.

In terms of summarization metrics, we notice the following:

1. The gaps between the 70B model and GPT-4o-mini are much smaller compared to relevance scores. For example, AlignScore for GPT-4o-mini remains around 0.7 consistently across all depths, while the 70B model can drop below 0.4 at 0% depth and exceed 0.6 at 100% depth. This is expected, as the summarization task primarily evaluates the ability to condense information effectively, which is less dependent on direct relevance between the summary and the source text. Nevertheless, the shape of the summarization metric curves resembles the relevance curves, albeit with smaller variances, suggesting that some correlation exists between the summarization metrics and relevance scores.

2. At 80%-100% depths (shaded area in Figure 5), the 70B model quickly catches up to GPT-4o-mini, even slightly outperforming it on BERTScore at 100% depth. The relatively small performance gap in summarization metrics here suggests that the 70B model's ability to generate summaries is close to that of the GPT-4o-mini model, implying that context utilization is the primary reason for the more significant performance gaps between the two models at earlier depths.

|           | BERTScore | AlignScore |
|----------------|-----------|------------|
| nvidia/Llama-3.1-Nemotron-70B-Instruct   | 0.6219    | 0.6699     |
| GPT-4o-mini    | 0.6206    | 0.7019     |

<div class="l-gutter caption" style="width: 150%">
  <b>Table 1:</b> BERTScore and AlignScore of the 70B model and the GPT-4o-mini model on vanilla chapter summarization (i.e. without irrelevant context).
</div>

Looking at the results of vanilla chapter summarization shown in Table 1, we indeed observe that the summarization metrics of both models are quite close when asked to summarize chapters based solely on the chapter text. The similarity in summarization metrics between the results in Table 1 and the 100% depth results in Figure 5 further indicates that the influence of document depth to summary quality is the smallest at the end of the context.

## Conclusion

In this blog post, we evaluate the Llama 3.1 70B Nemotron Instruct model using the chapter summarization version of the NIAH test. Our results show that the 70B model still needs to improve context utilization, as performance deteriorates quickly once the context length exceeds 64k tokens. We also observe a "lost in the beginning" pattern in terms of document depth, where the model performs particularly poorly when key information is placed in the initial parts of the context, specifically at depths between 10-30% for long context lengths. Comparing to GPT-4o-mini, we find that the 70B model still has a significant gap to close in terms of accurately identifying key information in the context.

Nevertheless, the 70B model demonstrates a promising ability to generate quality summaries even when the key information is located at the very end of the context, achieving comparable performance to GPT-4o-mini at 100% depth. Similarly, in the vanilla chapter summarization task—where no irrelevant context is introduced—the 70B model performs reasonably close to GPT-4o-mini. This suggests that the primary limitation of the 70B model is not its reasoning ability but its capacity to leverage all parts of the context equally effectively. Finally, we hope these findings highlight the challenges of context utilization and inspire further research into developing more robust open-source long context LLMs in the future.

## Book List

The list of books used in the experiment is provided below. We will not release the source chapter texts nor the reference chapter summaries due to copyright restrictions.

| Book                                              | Author              | Published Date   |
|---------------------------------------------------|---------------------|------------------|
| All Fours                                         | Miranda July        | May 14, 2024     |
| When the Moon Hatched                             | Sarah A. Parker     | January 13, 2024 |
| The Unwedding                                     | Ally Condie         | June 4, 2024     |
| Sociopath: A Memoir                               | Patric Gagne        | April 2, 2024    |
| Age of Revolutions: Progress and Backlash from 1600 to the Present | Fareed Zakaria      | March 26, 2024    |
| Reckless                                          | Lauren Roberts      | July 2, 2024     |
| The Women                                         | Kristen Hannah      | February 6, 2024 |
| You Like It Darker                                | Stephen King        | May 21, 2024     |
| Only If You’re Lucky                              | Stacy Willingham    | January 16, 2024 |
| Knife: Meditations After an Attempted Murder     | Salman Rushdie      | April 16, 2024   |
