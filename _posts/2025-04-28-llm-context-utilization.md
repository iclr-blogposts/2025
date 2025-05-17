---
layout: distill
title: "Open-Source vs Close-Source: The Context Utilization Challenge"
description: This blog post aims to evaluate how well the most capable open-source long context large language models (LLMs) utilize context, using the Needle In A Haystack test. We adopt the task of chapter summarization for recently published books to minimize data contamination while ensuring a challenging test. Our results show that open-source models still have room to improve in context utilization compared to close-source models.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Litu Ou
    url: https://www.linkedin.com/in/litu-ou-77baa61a6/
    affiliation:
      name: University of Edinburgh

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
  - name: In-depth Analysis for Llama
  - name: Comparison with GPT-4o-mini at 128k Context Length
  - name: Conclusion
  - name: Acknowledgements
  - name: Book List

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
---

## Introduction

The release of the Llama 3.1 <d-cite key="dubey2024llama3herdmodels"></d-cite> and Qwen 2.5 model family <d-cite key="qwen2025qwen25technicalreport"></d-cite> has been a game-changer for the open-source LLM community. These models support context lengths of up to 128k tokens, on par with the most capable [GPT-4o models](https://openai.com/index/hello-gpt-4o/). Nevertheless, accepting 128k tokens as input does not mean that the model can utilize them effectively. Many long context LLMs published previously did not perform well on the Needle In A Haystack (NIAH) test, showing significant performance differences across numerous context lengths and document depths when asked to retrieve a piece of information from within the context. In this blog post, we investigate the context utilization capabilities of long context open-source models using an enhanced version of the NIAH test, focusing on chapter summarization for recently published books. Our results show that the [Llama-3.1-Nemotron-70B-Instruct model](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) maintains decent performance up to 48k tokens but drastically deteriorates as the context length increases further. Compared to Llama, The Qwen-2.5-72B-Instruct model demonstrates a more stable performance overall, but still lags behind the GPT-4o-mini model at 128k context length. Our analysis suggests that the primary limitation of the ~70B models is not generation quality, but rather their ability to utilize all parts of the context effectively.

## Why the NIAH Test?

The context length of LLMs has increased dramatically over the years, evolving from 4k tokens in the initial [GPT-3.5 release](https://openai.com/index/chatgpt/) to 2M tokens in [Gemini 1.5 Pro](https://ai.google.dev/gemini-api/docs/long-context). Simultaneously, many benchmarks have been proposed to evaluate whether long context LLMs can effectively handle their advertised context lengths. These benchmarks typically focus on tasks such as document question answering, retrieval and summarization <d-cite key="wang2024novelqa"></d-cite> <d-cite key="zhang-etal-2024-bench"></d-cite>. It is undeniable that these benchmarks pose a significant challenge even to the most capable models. For instance, GPT-4o scores only 53.5 out of 100 on the Loong benchmark <d-cite key="wang-etal-2024-leave"></d-cite>.

Although existing benchmarks have attempted to include questions requiring aggregation of information distributed across the context, they are still insufficient for conducting fine-grained analysis of context utilization. Specifically, we aim to quantify which parts of the context the model focuses on when answering a query, helping us understand whether models can truly identify the key information needed for an accurate response. Fortunately, the Needle In A Haystack (NIAH) test provides an effective framework for such analysis. The simplest version of the test places a unique piece of information within a long, irrelevant context. After the context is presented to the model, a question is asked about this piece of information. A correct response indicates that the model successfully retrieved the key information; an incorrect response suggests otherwise.

Despite its simplicity, many previous long context LLMs struggled with this test. For instance, GPT-4-1106 128K achieved only 87.1% overall accuracy <d-cite key="fu2024dataengineeringscalinglanguage"></d-cite>. While long context LLMs continue to improve in context utilization, more complex versions of the NIAH test have been developed to further stress-test these models. These versions include placing multiple pieces of key information within the context <d-cite key="li2024needlebenchllmsretrievalreasoning"></d-cite>, increasing task difficulty <d-cite key="roberts2024needlethreadingllmsfollow"></d-cite>, etc. These more complex tests revealed that the most recent and capable LLMs still have shortfalls in context utilization. 

Context utilization remains a significant challenge yet to be effectively addressed, even for the most capable LLMs. In this blog post, we examine two of the best long context LLMs from the open-source community to evaluate how it compares to the most advanced closed-source models in terms of context utilization. Using the challenging task of chapter summarization for recently published books, we aim to uncover the limitations of the 70B Nemotron Instruct model, [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF), and the 72B Qwen Instruct model, [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct). By analyzing the results of this test and comparing them to those of more advanced closed-source models, we hope to highlight key areas for improvement in the development of more robust open-source long context LLMs.

## Reasons for Using Chapter Summarization in NIAH Test

Existing NIAH tests typically involve retrieving and answering questions about key information embedded in the context. Instead of adhering to this traditional paradigm, we adopt the task of summarizing chapters from books published in 2024 as the backbone for our NIAH test. Compared to previous approaches, our method offers the following advantages:

1. **Mitigating data contamination.** Llama 3.1 models have a knowledge cutoff of December 2023, meaning their pre-training data likely includes most publicly available texts on the internet dated before 2024. Consequently, using texts from before 2024 introduces a risk of data contamination, making it difficult to determine whether the model is utilizing the provided context or simply recalling pre-training knowledge. While finding perfectly clean data is challenging, we aim to minimize this risk by exclusively using books published in 2024 to build our test set.

2. **Direct visualization of key information retrieval.** The purpose of NIAH tests is to evaluate whether a model can retrieve relevant information from the context. Existing NIAH tests primarily focused on question answering tasks with short answer lengths. However, when an answer is incorrect, it is unclear whether the model failed to retrieve the key information entirely or if it successfully identified the information but failed to answer the question correctly. The latter could be due to the complexity of the question or interference from the surrounding context. 

In our chapter summarization task, we can directly compare the relevance of the model's summary to the chapter text to determine whether the summary is entirely, partially, or not at all derived from the chapter text. Since summaries are much longer than short answers, it becomes more straightforward to identify whether the model has retrieved the key information. If the summary includes characters or events present in the chapter, it can be considered a successful retrieval, and vice versa. Additionally, this approach provides insights into the model's ability to summarize long texts. By using summarization metrics, we can evaluate the quality of the summaries, either by comparing them to reference summaries or directly against the chapter texts.

## Experiment Setup

Finding comprehensive, high-quality chapter summaries for recent publications can be challenging. To ensure the evaluation is conducted with high-quality data, we use chapter summaries collected from [SuperSummary](https://www.supersummary.com/) consisting of 10 books published in 2024 (Full book list provided [below](#book-list)). All chapter summaries are manually curated by domain experts <d-footnote>https://www.supersummary.com/ai-powered-content-disclaimer/</d-footnote>. We sample 100 chapter summaries with their corresponding chapter texts as our test set, with an average chapter length of 3,032 tokens and an average summary length of 183 tokens. Following the evaluation format used in LLMTest_NeedleInAHaystack <d-cite key="kamradt2023needle"></d-cite>, we set the context length to range from 16k to 128k tokens, with a step size of 16k, and the needle position to range from 0% to 100%, with a step size of 10%. This yields a total of 8,800 test cases. 

For prompt engineering of the NIAH test, we first construct the needle prompt which identifies the chapter text. We design the needle prompt to have an instruction placed before the chapter text with a 6-letter identifier that uniquely identifies the chapter text within the entire context. To fill the rest of the context with irrelevant information, we use Paul Graham Essays as the filler texts following previous approaches <d-cite key="kamradt2023needle"></d-cite>. After the context is constructed, a summary generation instruction is appended at the end of the context. The whole prompt looks like the following:

```txt
... Random filler texts ...
... Random filler texts ...
... Random filler texts ...

Below is a chapter from a book with ID: fbnmkd, marked by the <chapter> tags.
<chapter>
{}
</chapter>

... Random filler texts ...
... Random filler texts ...
... Random filler texts ...
Read the document and answer: Summarize the chapter from the book with ID: eDrFhW. The summary should be a coherent paragraph containing all key information about the chapter.
```

In terms of evaluation metrics, we primarily assess the relevance between the ~70B models' summary and the chapter text using the GPT-4o model. We ask the model to classify the relevance into three categories: 1) Irrelevant, where the summary is completely unrelated to the chapter text; 2) Partially relevant, where the summary contains some information from the chapter text but also includes irrelevant details; 3) Fully relevant, where the summary closely follows the chapter text. Additionally, we also evaluate the ~70B models' ability to generate chapter summaries based solely on the chapter text using automatic summarization metrics. Specifically, we adopt the following two metrics:

1. BERTScore <d-cite key="rogers2019bertscore"></d-cite>. This metric measures the similarity between the generated summary and the reference summary using BERT embeddings. We use "microsoft/deberta-xlarge-mnli" <d-cite key="he2021debertadecodingenhancedbertdisentangled"></d-cite> as the backbone embedding model and compute the F1 score.
2. AlignScore <d-cite key="kamradt2023needle"></d-cite>. This metric measures whether the generated summary aligns with the contents of the chapter text. We use the [AlignScore-large model](https://github.com/yuh-zha/AlignScore?tab=readme-ov-file#checkpoints) as the backbone natural language inference model.

Finally, we also evaluate the GPT-4o-mini model on the full 128k context length tests to compare the performance of the ~70B models to that of a capable closed-source model with the same context length.

## NIAH Visualization

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/llama_relevance.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 1:</b> Relevance scores of the Llama-3.1-Nemotron-70B-Instruct model across different context lengths and needle positions.
</div>

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/qwen_relevance.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 2:</b> Relevance scores of the Qwen2.5-72B-Instruct model across different context lengths and needle positions..
</div>

Figures 1 and 2 illustrate the relevance scores of the Llama-3.1-Nemotron-70B-Instruct and Qwen2.5-72B-Instruct models across varying context lengths and needle positions, respectively. For the Llama model, we observe that the relevance score remains close to 1 at context lengths up to 48k tokens, but gradually deteriorates as the context length increases further. Regarding document depth, performance begins to decline rapidly at depths between 10% and 30% when the context length reaches 64k tokens, ultimately dropping to nearly 0 at depths between 0-50% and 80% for the full 128k context length. Compared to the Llama model, the Qwen model exhibits more stable performance across all context lengths and depths. However, we still observe degraded performance approaching a 128k context length. At 128k tokens, the average relevance score is 79.3% for Qwen, whereas the average relevance score across all context lengths is 81.47%, suggesting that the Qwen model still has considerable room for improvement in context utilization.

Traditionally, when models are asked to summarize documents, the output summaries tend to exhibit "lead bias" that focuses more on the first few segments of the entire context <d-cite key="ravaut-etal-2024-context"></d-cite>. Existing work that directly measures context utilization also found the "lost in the middle" problem, where models perform better at retrieving key information from the beginning and the final parts of the context compared to the middle <d-cite key="liu-etal-2024-lost"></d-cite>. In comparison to previous work, our results demonstrate a "lost in the beginning" pattern, where the model performs particularly poorly when the key information is placed in the initial parts of the context. Although the Qwen model also demonstrates significantly worse performance at 80-90% depths towards 128k context length, both ~70B models give degraded results when the chapter text is placed between 10-50% depths.

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/example.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 3:</b> Example summaries with their corresponding relevance scores for both ~70B models (Left) and the reference summary (Right). The chapter text is placed at 70% document depth with 128k context length. Irrelevant parts of the summaries are highlighted in red and the relevant parts of the summaries are highlighted in green.
</div>

To examine the characteristics of summaries containing irrelevant information, Figure 3 presents example summaries generated by both ~70B models, alongside the reference summary. The Qwen model successfully identifies key characters and some key events in the first half of its summary but shifts to a different topic in the second half, discussing content extracted from the filler texts. While the Qwen model partially identifies the chapter content, the Llama model fails to do so, producing a summary that is entirely unrelated to the chapter text. These examples demonstrate that both ~70B models struggle to accurately isolate the chapter text, despite the use of unique <chapter> tags intended to differentiate it from the filler texts.

## In-depth Analysis for Llama

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/context_proportion.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 4:</b> Proportions of irrelevant, partially relevant and fully relevant summaries at different context lengths.
</div>

To gain a better understanding of the performance differences across various context lengths, we examine the proportions of irrelevant, partially relevant, and fully relevant summaries at different context lengths. From Figure 3, we can see that the model maintains robust performance (>94% correct responses) up to 48k tokens, suggesting strong capability in handling moderate-length contexts. However, this performance begins to decline abruptly as the context length increases, with notable drops around 48k-64k tokens where relevant cases decrease rapidly. The decline becomes particularly pronounced after 96k tokens, where correct responses drop below 50%, ultimately reaching just 19% at 128k tokens. This pattern, coupled with the corresponding exponential rise in incorrect responses at longer context lengths, indicates a systematic degradation in the model's ability to identify key information as the context length increases.

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/depth_proportion.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 5:</b> Proportions of irrelevant, partially relevant and fully relevant summaries at different document depths.
</div>

Conducting the same analysis on document depth, we observe that the distribution is more uniform compared to the distribution of context length. The model is able to generate over 60% of relevant summaries at all depths except for the 10%-30% range. We also observe an steady increase in performance towards deeper depths starting from 30%, where the proportion of fully relevant summaries climbs from 49.8% at 30% depth to 97.8% at 100% depth. Interestingly, the model has a 28.4% error rate at the beginning of the document, which is smaller than the error rates at depths of 10%-50%, but larger than those at the remaining depths. This distribution can be explained by the "lost in the beginning" pattern observed in Figure 1, but also resembles a "lost in the middle" pattern skewed toward the beginning of the document. Finally, the proportion of partially relevant summaries is evenly distributed across all depths (<5%), in contrast to the distribution over context length in Figure 3, where the majority of partially relevant summaries are found at high context lengths. This balanced distribution of uncertain responses suggests that the model maintains consistent calibration regardless of document depth, avoiding confusion between irrelevant context and key information.

## Comparison with GPT-4o-mini at 128k Context Length

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/relevance_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 6:</b> Relevance score comparison between Llama-3.1-Nemotron-70B-Instruct, Qwen2.5-72B-Instruct and GPT-4o-mini at 128k context length.
</div>

All analysis up to this point has been based on the ~70B models only. To understand how it compares to capable closed-source models, we run the full 128k context length test on the GPT-4o-mini model to see if any performance gap exists. Figure 5 shows the comparison between the two models across all metrics. We can immediately observe that the Llama-3.1-Nemotron-70B-Instruct model experiences a drastic drop in relevance scores at various depths. While GPT-4o-mini achieves near-perfect relevance scores across most depths, Llama-3.1 struggles significantly, only recovering towards the end of the context. Qwen2.5-72B-Instruct exhibits a performance trend that falls between the two models. Although it does not maintain the same high stability as GPT-4o-mini, it generally performs better than Llama-3.1, particularly in the earlier depths. However, Qwen2.5 also shows a significant dip in performance around the 30% depth, where its score briefly drops below 0.4 before recovering. It is worth mentioning that all three models show a similar pattern of performance degradation at depths between 10-30% and 70-90%, potentially indicating a common phenomenon in context utilization.

<div class="row justify-content-center">
    <div style="height: 70%">
        {% include figure.html path="assets/img/2025-04-28-llm-context-utilization/trend.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="l-gutter caption" style="width: 150%">
  <b>Figure 7:</b> Performance variation of the ~70B models and the GPT-4o-mini model at BERTScore and AlignScore. In each cell, the score at the top represents the vanilla chapter summarization result, while the score at the bottom represents the average NIAH result for all document depths at 128k context length.
</div>

Thus far, we have demonstrated that open-source models lag behind the closed-source GPT-4o-mini model in context utilization. However, another crucial factor that impacts summary quality is the model's natural language generation (NLG) capability. If open-source models possess inherently weaker NLG abilities, the argument that context utilization is their primary limitation would be weakened. To isolate the influence of NLG ability on summary quality, we evaluated the performance of the 70B model and the GPT-4o-mini model on vanilla chapter summarization (i.e., generating chapter summaries based solely on the chapter text) and compared the results with those from the NIAH tests. This comparison allows us to assess the impact of introducing irrelevant context on summary quality.

A comparison of vanilla chapter summarization results with the NIAH tests is presented in Figure 7. The results indicate that all three models exhibit similar performance in vanilla chapter summarization. The largest discrepancy lies in the AlignScore between Llama and GPT-4o-mini, but this difference is only 0.032. Critically, however, when irrelevant context is introduced in the NIAH tests, the performance of all three models declines significantly. The most pronounced change is observed in the Llama model, where the AlignScore decreases by 31.4%. The Qwen model is less affected than Llama, experiencing drops below 10% for both metrics. GPT-4o-mini is the least affected, with slightly better performance on BERTScore compared to Qwen, while even showing a 0.9% increase in AlignScore. Overall, these results are consistent with the findings presented in Figure 6, reinforcing the conclusion that context utilization has a substantial influence on summary quality. We also note that the 0.9% increase in AlignScore for GPT-4o-mini is unexpected as the NIAH test is designed to be more difficult than the vanilla chapter summarization task. This suggests that automatic summarization metrics are not sensitive enough to detect content mismatches, which is why we use the relevance score to directly measure the ability of the model to identify key information in the context.

## Conclusion

In this blog post, we evaluate open-source long-context LLMs, including Llama-3.1-Nemotron-70B-Instruct and Qwen2.5-72B-Instruct, using the chapter summarization version of the NIAH test. Our results indicate that ~70B models still have room for significant improvement in context utilization, with the Llama model exhibiting more severe performance degradation compared to the Qwen model. We also observe a "lost in the beginning" pattern with respect to document depth, where model performance is particularly poor when key information is placed in the initial sections of the context, specifically at depths between 10-50% for extended context lengths. Compared to GPT-4o-mini, the tested open-source models exhibit a significant performance gap in accurately identifying key information within the context.

However, the ~70B models demonstrate a promising ability to generate summaries with quality comparable to that of GPT-4o-mini when no irrelevant context is introduced. This suggests that the primary limitation of the tested open-source models is not their natural language generation capabilities, but rather their capacity to leverage all parts of the context with equal effectiveness. We hope these findings highlight the challenges of context utilization and inspire further research aimed at developing more robust open-source long-context LLMs in the future.

## Acknowledgements

This work has made use of the resources provided by the Edinburgh Compute and Data Facility (ECDF) (http://www.ecdf.ed.ac.uk/), and supported by the Edinburgh International Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh. We also would like to thank Prof. Mirella Lapata for her helpful comments and suggestions for improving the camera-ready version of this blog post.

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
