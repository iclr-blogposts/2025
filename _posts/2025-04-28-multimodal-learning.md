---
layout: distill
title: >-
  Multi-modal Learning: A Look Back and the Road Ahead
description: "Advancements in language models has spurred an increasing interest in multi-modal AI — models that 
process and understand information across multiple forms of data, such as text, images and audio. 
While the goal is to emulate human-like ability to handle diverse information, a key question is: do human-defined 
modalities align with machine perception? If not, how does this misalignment affect AI performance? 
In this blog, we examine these questions by reflecting on the progress made by the community 
in developing multi-modal benchmarks and architectures, highlighting their limitations. By reevaluating our definitions and assumptions, 
we propose ways to better handle multi-modal data by building models that 
analyze and combine modality contributions both independently and jointly with other modalities."
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Divyam Madaan
    url: "https://dmadaan.com/"
    affiliations:
      name: New York University
  - name: Sumit Chopra
    url: "https://www.spchopra.net/"
    affiliations:
      name: New York University, New York University Grossman School of Medicine
  - name: Kyunghyun Cho
    url: "https://kyunghyuncho.me/"
    affiliations:
      name: New York University, Genentech, CIFAR


# must be the exact same name as your blogpost
bibliography: 2025-04-28-multimodal-learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: Why is modality grouping information important?
  - name: Reflection on the Progress so Far
    subsections:
      - name: Are benchmarks enough?
      - name: Are complex multi-modal architectures enough?
  - name: Towards Meaningful Progress in Multi-modal Learning
    subsections:
      - name: Analysis of the strengths of individual and combination of modalities
      - name: Building better multi-modal models
  - name: Takeaway

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
  .box-note, .box-warning, .box-error, .box-important {
    padding: 15px 15px 15px 10px;
    margin: 20px 20px 20px 5px;
    border: 1px solid #eee;
    border-left-width: 5px;
    border-radius: 5px 3px 3px 5px;
  }
  d-article .box-note {
    background-color: #eee;
    border-left-color: #2980b9;
  }
  d-article .box-warning {
    background-color: #fdf5d4;
    border-left-color: #f1c40f;
  }
  d-article .box-error {
    background-color: #f4dddb;
    border-left-color: #c0392b;
  }
  d-article .box-important {
    background-color: #d4f4dd;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article .box-note {
    background-color: #333333;
    border-left-color: #2980b9;
  }
  html[data-theme='dark'] d-article .box-warning {
    background-color: #3f3f00;
    border-left-color: #f1c40f;
  }
  html[data-theme='dark'] d-article .box-error {
    background-color: #300000;
    border-left-color: #c0392b;
  }
  html[data-theme='dark'] d-article .box-important {
    background-color: #003300;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article aside {
    color: var(--global-text-color) !important;
  }
  html[data-theme='dark'] d-article blockquote {
    color: var(--global-text-color) !important;
  }
  html[data-theme='dark'] d-article summary {
    color: var(--global-text-color) !important;
  }
  d-article aside * {
    color: var(--global-text-color) !important;
  }
  d-article p {
    text-align: justify;
    text-justify: inter-word;
    -ms-hyphens: auto;
    -moz-hyphens: auto;
    -webkit-hyphens: auto;
    hyphens: auto;
  }
  d-article aside {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
    font-size: 90%;
  }
  d-article aside p:first-child {
      margin-top: 0;
  }
  d-article details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
  }
  d-article summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
    display: list-item;
  }
  d-article details[open] {
    padding: .5em;
  }
  d-article figure {
    padding: 1em 1em 0;
  }
  d-article details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
  }
  html[data-theme='dark'] d-article blockquote {
    border-left-color: #f1c40f;
  }
  @media (min-width: 768px) {
  .l-gutter:has(> aside) {
      height: 0px;
    }
  }
  @media (min-width: 1025px) {
    d-article d-contents {
      height: 0px;
    }
  }
---

## Introduction

Humans constantly use multiple senses to interact with the world around us. We use **vision** to see, **olfaction** to smell, **audition** to hear, and we communicate through **speech**. Similarly, with recent multi-modal artificial intelligence (AI) advancements, we now see [articles](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/) announcing “ChatGPT can **see, hear and speak”**. But there’s a fundamental question underlying this progress:

<aside class="box-note l-body" markdown="1">
 Does the definition of a "modality" align between humans and AI?
</aside>

To unpack this question, we show examples that illustrate the basic ambiguity of defining modalities for machine learning.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
    {% include figure.html path="assets/img/2025-04-28-multimodal-learning/modality_types.png" class="img-fluid" width=width %}
  </div>
  <div class="caption">
A dog image with caption represented with RGB channels and texture as four image modalities, alongside tokenization strategies as modalities for text.
  </div>
  </div>
</div>

**Example 1: Images**

Images are commonly considered as a single modality for both humans and AI. For humans, an image is a unified representation of visual information. However, machine learning models can perceive images in various ways :

- **Color channels:** RGB images comprise of Red, Green, and Blue (R, G, B) channels, each of which could be viewed as a separate component or modality.
- **Derived Representations**: We can generate secondary images highlighting specific features<d-cite key="zeng2024rgb"></d-cite>, such as material properties like shininess or texture.
- **Image Patches:** The image can be divided into smaller sections or patches of different sizes<d-cite key="dosovitskiy2021an"></d-cite>, capturing different parts of the image.

These decompositions raise an important question: 

<aside class="box-note l-body" markdown="1">
Should each decomposition be considered as a modality, or are they simply variations of the image modality?
</aside> 

**Example 2: Text**

Text is commonly treated as a single modality in language models, however, the method of tokenization—how text is broken into atomic units—can vary widely:

- **Word-Level Tokenization:** Sentences are split into individual words. For example, the sentence "The dog is on the grass" would be tokenized as (`the` `dog` `is` `on` `the` `grass`)
- **Character-Level Tokenization:** Each individual character is treated as a token, turning the same sentence into (`t` `h` `e`  `d` `o` `g`  `i` `s` `o` `n` `t` `h` `e` `g` `r` `a` `s` `s`)

This leads us to ask:

<aside class="box-note l-body" markdown="1">
Should each tokenization be considered as a different modality or a different view of the text modality?
</aside> 


Although the term “multi-view” is sometimes used to describe these variations, the line between "multi-view" and "multi-modal" is often blurred, adding to the confusion. This is an important question because often different tokenization strategies perform differently on different tasks<d-cite key="dang2024tokenization"></d-cite>. While BPE and UnigramLM i.e. world-level embeddings are commonly used for LLMs, character level encoding works better than world-level tokenization for multilingual translation<d-cite key="lee2017fully"></d-cite>.

**Example 3: Medical data**

Consider diagnosing skin lesion using both image data and tabular features, such as patient’s age, demographic information and characteristics of the lesion along with the anatomical region. Each modality independently and jointly contributes to detecting lesion<d-cite key="imrie2024automated"></d-cite>. This prompts the question,

<aside class="box-note l-body" markdown="1">
Are images and tabular data separate modalities, or can they be unified under "patient" modality?
</aside>

### Why is modality grouping information important?
The definition and grouping of modalities is important because it affects how we design models to process and integrate different types of data. Often the objective of prior studies is to ensure that models capture interactions between all mdoalities for the downstream task. This goal has led to two parallel lines of work. One approach focuses on developing new benchmarks to capture this interaction. These benchmarks often exhibit uni-modal biases, leading to subsequent iterations or new benchmarks intended to better evaluate multi-modal capabilities. The other approach emphasizes building complex architectures designed to learn interactions between modalities more effectively.

In this blog post, we examine the community’s progress in both of these directions and why they have fallen short of meaningful impact. We then propose ways to analyze and capture the relative importance of individual modalities and their interaction for the downstream task. 

## Reflection on the Progress so Far
Over the past decade, numerous multi-modal  benchmarks and model architectures have been proposed to evaluate and enhance the multi-modal learning capabilities of AI models. Yet, we continue to struggle with making meaningful progress due to benchmarks not accurately representing real-world scenarios and models failing to capture the essential multi-modal dependencies effectively. We reflect on the progress made in these two areas and discuss why these approaches have not been sufficient in obtaining the desired results. 

### Are benchmarks enough?

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
    {% include figure.html path="assets/img/2025-04-28-multimodal-learning/benchmarks.png" class="img-fluid" width=width %}
  </div>
  <div class="caption">
     List of benchmarks for multi-modal learning from a recent work<d-cite key="tong2024cambrian"></d-cite>.
  </div>
  </div>
</div>

Numerous benchmarks have been developed for multi-modal learning (see Figure above). Many were initially designed to evaluate whether models could effectively capture interactions between modalities for downstream tasks. The early benchmarks often exhibited a reliance on uni-modal biases, allowing models to perform well even when ignoring certain modalities<d-cite key="wang2020makes, tong2024cambrian"></d-cite>. Although subsequent iterations of these benchmarks made an attempt to address these issues, these updates often fail to fully resolve the underlying problems. Instead, they highlight persistent challenges that reflect the complexity of real-world scenarios where multi-modal models continue to struggle.

For instance, consider visual question answering (VQA)<d-cite key="antol2015vqa"></d-cite>, which has undergone multiple updates since its inital release in 2015:

1. **VQAv2 (2016)**<d-cite key="goyal2017making"></d-cite>: Attempted to mitigate language bias by providing distinct answers for two different image-question pairs. Despite the subsequent updates, VQAv2 continues to be a benchmark in research and evaluations for recent models like Gemini<d-cite key="team2023gemini"></d-cite>, ChatGPT and LLaVA<d-cite key="liu2024visual"></d-cite>.
2. **VQA-CP (2018)**<d-cite key="agrawal2018don"></d-cite>: Adjusted answer distributions between training and test sets to reduce prior biases.
3. **VQA-CE (2021)**<d-cite key="dancette2021beyond"></d-cite>: Emphasized multi-modal spurious dependencies among image, text, and answer to better capture multi-modal interactions.
4. **VQA-VS (2022)**<d-cite key="si2022language"></d-cite>: Broadened the scope to explore both uni-modal and multi-modal shortcuts in a more comprehensive manner.

Even after a decade of refinements, VQA benchmarks continue to grapple with biases and limitations, raising concerns about the direction of constructing more and more benchmarks. While many benchmarks aim to evaluate different capabilities of models, they often result in only incremental improvements. Similar challenges are evident in non-VQA benchmarks such as Natural Language Visual Reasoning (NLVR)<d-cite key="suhr2019corpus"></d-cite>, action recognition using RGB frames combined with optical flow and audio<d-cite key="wang2020makes"></d-cite>, 3D object classification employing front and rear views as two modalities<d-cite key="du2023uni"></d-cite>, and many others. 

These examples highlight a critical limitation:

<aside class="box-error l-body" markdown="1">
Designing a "balanced" benchmark to avoid dependence on a single modality fails to capture the true complexity of real-world scenarios.
</aside>

### Are complex multi-modal architectures enough?

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
    {% include figure.html path="assets/img/2025-04-28-multimodal-learning/modality_fusion.png" class="img-fluid" width=width %}
  </div>
  <div class="caption">
      Conventional multi-modal methods use different fusion methods to tackle the underlyung task effectively.
  </div>
  </div>
</div>

In the task of VQA, as shown in the Figure above, many methods would process the input and text modalities to have same dimensions and then use one of the many conventional “multi-modal” algorithms:

1. **Early fusion**: These methods concatenate modality features followed by joint processing of features. This involves using a unified encoder with shared parameters across all modalities. This approach is common in early multi-modal learning based methods<d-cite key="baltruvsaitis2018multimodal, barnum2020benefits, gadzicki2020early, hessel2020does"></d-cite>
to recent transformer based methods<d-cite key="likhosherstov2021polyvit, liang2023highmodality"></d-cite>.
<!-- supervised and [self-supervised methods](https://arxiv.org/pdf/2304.01008).  -->
2. **Intermediate fusion**<d-cite key="anderson2018bottom,joze2020mmtm, wang2020deep, wu22characterizing"></d-cite>: These methods fuse specific layers, rather than sharing all the parameters. 
<!-- The underlying objective remains the same — to capture the interaction between modalities for the downstream task -->
3. **Late fusion**<d-cite key="morvant2014majority, kielaclark2015multi, shutova2016black, li2021align, singh2022flava"></d-cite>: Thse methods learn separate encoder representations for each modality followed by fusion. The fusion often uses additional layers to capture the interaction between these modalities on top of the individual representations. 

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    <div style="background-color: white; padding: 10px; border-radius: 5px;">
    {% include figure.html path="assets/img/2025-04-28-multimodal-learning/eyes_shut.png" class="img-fluid" width=width %}
  </div>
 <div class="caption">
      Examples of scenarios where GPT-4V fails to accurately answer questions about the provided images<d-cite key="tong2024eyes"></d-cite>.
  </div>
  </div>
</div>

Despite a decade of advancements in developing these complex architectures, many multi-modal models continue to struggle with effectively integrating vision and text, often disregarding one modality in the process (see Figure above). For example, **while humans consistently achieve around 95% accuracy on VQA, recent AI models such as GPT-4V and Gemini only reach about 40%, with others like LLaVA-1.5, Bing Chat, mini-GPT4, and Bard performing even worse—sometimes falling below random performance levels**<d-cite key="tong2024eyes"></d-cite>. 

Some studies<d-cite key="tong2024cambrian, si2022language"></d-cite> attribute this persistent performance gap to inherent imbalances or design flaws within the benchmarks themselves, while others<d-cite key="wang2020makes, du2023uni"></d-cite> argue that the issue lies with models failing to effectively learn from the interaction of modalities. This debate has prompted successive iterations of benchmarks and models aimed at addressing these challenges. Several survey papers<d-cite key="baltruvsaitis2018multimodal, liang2022foundations"></d-cite> provide a comprehensive overview of recent developments in the field.


<aside class="box-error l-body" markdown="1">
Even with hundreds of benchmarks and increasingly complex architectures, we still have a limited understanding of the specific limitations inherent in earlier versions, and lack a clear strategy for tackling future benchmarks.
</aside>

## Towards Meaningful Progress in Multi-modal Learning

To drive meaningful progress in multi-modal learning, we need to move away from simply creating more benchmarks or building increasingly complex architectures. While these efforts have advanced the field incrementally, they haven't tackled its fundamental challenges. Instead, we propose approaching the field from a two stage perspective below:


### Analysis of the strengths of individual and combination of modalities

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <div style="background-color: white; padding: 10px; border-radius: 5px;">
            {% include figure.html path="assets/img/2025-04-28-multimodal-learning/analyze_benchmarks.png" class="img-fluid" %}
        </div>
        <div class="caption">
         Relative inter- and intra-modality dependency strengths across multiple benchmarks.
        </div>
    </div>
</div>

For any dataset or benchmark, we recommend to start with the assumption of what constitutes a modality in the context of the desired task. This definition should not be limited to human-perceived notions of modality but should critically evaluate and challenge these assumptions. The goal however is not merely to label an input as multi-modal; rather, it is to assess whether such labeling provides meaningful advantages for model performance or understanding.

To answer that question, we recommend a thorough examination of the dependencies for each defined modality, both individually and jointly with other modalities for the target task.  These dependencies are categorized as ***intra-modality dependencies***, which represent interactions between individual modalities and the target label and ***inter-modality dependencies***, which captures interaction between modalities and label. 

Several studies have evaluated benchmarks involving images and text as two modalities, and we illustrate how these dependencies differed across benchmarks based on prior studies<d-cite key="wang2020makes, du2023uni, tong2024cambrian, madaan2024jointly"></d-cite> in the Figure above. For datasets such as ***SQA-I, MMMU, fastMRI, MathVista, UCF101, intra-modality dependencies are more important***, as removing one modality all together does not affect model performance. Conversely, for datasets like ***MMB, SEED-I, GQA, ChartQA and so on, both inter- and intra-modality dependencies are important***, as both uni-modal and inter-modality models obtain more than random performance. For benchmarks such as **MME, OCRBench, NLVR2 and others, inter-dependencies are more important***;* removing one modality results in random performance; underscoring the importance of modality interactions. 

For all these benchmarks, specific architectural choices—such as the type of fusion method or the backbone architecture for vision and language models—exhibit minimal impact on performance, as noted in prior studies<d-cite key="wang2020makes, du2023uni, tong2024cambrian, madaan2024jointly"></d-cite>. Thus

<aside class="box-important l-body" markdown="1">
The emphasis should be on analyzing how each modality contributes to task performance, both independently and jointly with other modalities. 
</aside>

This analysis provides a better understanding of the importance of each modality and their interaction in the corresponding dataset. Such understanding enables us to prioritize and weight these contributions when constructing multi-modal models, as elaborated in the next section.

### Building better multi-modal models


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <div style="background-color: white; padding: 10px; border-radius: 5px;">
            {% include figure.html path="assets/img/2025-04-28-multimodal-learning/inter_and_intra.png" class="img-fluid" style="width: 120% !important; max-width: none; height: auto;" %}
        </div>
        <div class="caption">
          Product of experts model combining the inter-modality and intra-modality models<d-cite key="madaan2024jointly"></d-cite>. 
        </div>
    </div>
</div>

With a clear understanding of each modality’s contribution, we can prioritize the dependencies that are the most important, or simply build a “product of experts” approach to combine these dependencies (see Figure above). Particularly, for two modalities $$\{x_1, x_2\}$$ and label $y$, the output can be expressed as the product of two sets of models: one capturing the importance of individual modalities for the label (intra-modality predictors) and the other focusing on the importance of their interactions for the label (inter-modality predictor) as follows:


{% highlight python linenos %}
def forward(x_1, x_2):
  """
  Forward pass for multi-modal classification using Product of Experts (PoE).
  Combines predictions from intra-modality models and inter-modality model.

  Args:
    x_1 (Tensor): Input from modality 1 
    x_2 (Tensor): Input from modality 2

  Returns:
    Tensor: Log-probabilities over output classes, shape [batch_size, num_classes]
  """
  # Intra-modality predictors (separate models for each modality)
  intra_output_1 = self.intra_model_1(x_1)
  intra_output_2 = self.intra_model_2(x_2)

  # Inter-modality predictor (fusion model over both modalities)
  inter_input = torch.cat([x_1, x_2], dim=-1)
  inter_output = self.inter_model(inter_input)

  # Convert outputs to log-probabilities
  log_probs_intra_1 = torch.log_softmax(intra_output_1, dim=-1)
  log_probs_intra_2 = torch.log_softmax(intra_output_2, dim=-1)
  log_probs_inter = torch.log_softmax(inter_output, dim=-1)

  # Product of Experts: add log-probabilities from each expert
  combined_log_probs = log_probs_intra_1 + log_probs_intra_2 + log_probs_inter

  # Normalize
  log_normalizer = torch.logsumexp(combined_log_probs, dim=-1, keepdim=True)
  log_probs = combined_log_probs - log_normalizer

  return log_probs

{% endhighlight %}

The code above combines the output log probabilities in an additive way and has been shown to work effectively across various healthcare, language, and vision benchmarks, even when the relative strength of these dependencies is not known<d-cite key="madaan2024jointly"></d-cite>. Such an approach does not fundamentally alter the multi-modal learning problem; instead, it offers a structured way to balance individual and joint modality contributions. By explicitly modeling the importance of individual modalities, this approach mitigates uni-modal biases.

<aside class="box-important l-body" markdown="1">
For multi-modal learning, we must model and combine both inter- and intra-modality dependencies.
</aside>

This comes with the trade-off of increased parameter requirements, which could impact efficiency. We believe future research should focus on optimizing this framework to reduce its computational cost. Progress in this direction is important, as current trends often attempt to address these challenges by either expanding datasets or increasing architectural complexity as highlighted above—approaches that have not led us to efficient or scalable solutions. 

## Takeaway
Current approaches to multimodal learning tend to overemphasize the interaction between modalities for downstream tasks, resulting in benchmarks and architectures narrowly focused on modeling these interactions. In real-world scenarios, however, the strength of these interactions are often unkown. To build more effective multimodal models, we need to shift our focus toward holistically understandinging the independent contributions of each modality as well as their joint impact on the target task.

## Acknowledgement
This work was supported by the Institute of Information & Communications Technology Planning
& Evaluation (IITP) with a grant funded by the Ministry of Science and ICT (MSIT) of the Republic
of Korea in connection with the Global AI Frontier Lab International Collaborative Research,
Samsung Advanced Institute of Technology (under the project Next Generation Deep Learning: From Pattern
Recognition to AI), National Science Foundation (NSF) award No. 1922658, Center for Advanced Imaging Innovation and Research (CAI2R), National Center for Biomedical Imaging and Bioengineering operated by NYU Langone Health, and National Institute of Biomedical Imaging and Bioengineering through award number P41EB017183.
