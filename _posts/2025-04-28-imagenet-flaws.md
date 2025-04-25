---
layout: distill
title: Flaws of ImageNet, Computer Vision's Favorite Dataset
description: Since its release, ImageNet-1k has been a gold standard for evaluating model performance. It has served as the foundation of numerous other datasets and it has been widely used for pretraining. <br/> As models have improved, issues related to label correctness have become increasingly apparent. In this blog post, we analyze the issues, including incorrect labels, overlapping or ambiguous class definitions, training-evaluation domain shifts, and image duplicates. The solutions for some problems are straightforward. For others, we hope to start a broader conversation about how to improve this influential dataset to better serve future research.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Nikita Kisel
    url:
    affiliations:
      name: Visual Recognition Group, Czech Technical University in Prague
  - name: Illia Volkov
    url:
    affiliations:
      name: Visual Recognition Group, Czech Technical University in Prague
  - name: Kateřina Hanzelková
    url:
    affiliations:
      name: Faculty of Science, Charles University, Czech Republic, Prague
  - name: Klara Janouskova
    url:
    affiliations:
      name: Visual Recognition Group, Czech Technical University in Prague
  - name: Jiri Matas
    url:
    affiliations:
      name: Visual Recognition Group, Czech Technical University in Prague

bibliography: 2025-04-28-imagenet-flaws.bib

toc:
  - name: Introduction to ImageNet
  - name: Known ImageNet-1k Issues
    subsections:
    - name: Bringing the Errors Together
  - name: Dataset Construction Issues
    subsections:
    - name: The Cat Problem
    - name: The Laptop Problem
    - name: Exploring VLM Results
    - name: Key Takeaways
  - name: Distribution Shift Between Training and Validation Sets
  - name: Images Requiring Multiple Labels
  - name: ILSVRC Class Selection
    subsections:
    - name: Taxonomy of class selection problems
  - name: Addressing Duplicates
    subsections:
    - name: "Exact Duplicate Search: Pixel-Level Comparisons"
    - name: Near Duplicate Detection Method
  - name: Prompting Vision-Language Models
  - name: "Fixing ImageNet Labels: A Case Study"
    subsections:
    - name: The Weasel Problem
    - name: The Ferret Problem
    - name: Results of Relabeling
  - name: Conclusions
  - name: Outlook

---

<!-- Light/dark modes styles -->
<style>
  /* Light mode styles */
  details {
    background-color: #f5f5f5;
    border-radius: 3px;
    padding: 15px;
  }

  summary {
    color: black;
  }

  figcaption {
    color: black;
  }

  .llm-cite {
    margin-bottom: 10px;
    border-left: 4px solid lightgrey;
    border-radius: 3px;
    padding: 10px;
    font-style: italic;
    background-color: #EEEEEE;
  }

  .weasel-caption {
    color: black;
  }

  .theme-image {
    height: 27rem;
    background-size: contain; /* Ensures the entire SVG fits */
    background-repeat: no-repeat;
    background-position: center;
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/mustelidae.svg' | relative_url }}");
  }

  .weasel-image {
    height: 30rem;
    margin-bottom: 25px;
    background-size: contain; /* Ensures the entire SVG fits */
    background-repeat: no-repeat;
    background-position: center;
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/weasel_fam_distr.svg' | relative_url }}");
  }

  .related-papers-image {
    height: 17rem;
    background-size: contain; /* Ensures the entire SVG fits */
    background-repeat: no-repeat;
    background-position: center;
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/related_papers.svg' | relative_url }}");
  }

  .multiple-papers-image {
    height: 15rem;
    background-size: contain; /* Ensures the entire SVG fits */
    background-repeat: no-repeat;
    background-position: center;
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/multiple_papers.svg' | relative_url }}");
  }

  @media (min-width: 800px) {
     .weasel-image {
        height: 22rem;
        }
     }

  @media (min-width: 1200px) {
     .weasel-image {
        height: 30rem;
        }
    }

  .iframe-dark {
    display: none;
  }

  /* Dark mode styles */
  html[data-theme="dark"] details {
    background-color: #424346;
  }

  html[data-theme="dark"] summary {
    color: white;
  }

  html[data-theme="dark"] figcaption {
    color: white;
  }

  html[data-theme="dark"] .llm-cite {
    background-color: #424346;
  }

  html[data-theme="dark"] .weasel-caption {
    color: white;
  }

  html[data-theme="dark"] .theme-image {
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/mustelidae_dark.svg' | relative_url }}");
  }

  html[data-theme="dark"] .weasel-image {
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/weasel_fam_distr_dark.svg' | relative_url }}");
  }

  html[data-theme="dark"] .iframe-dark {
    display: block;
  }

  html[data-theme="dark"] .iframe-light {
    display: none;
  }

  html[data-theme="dark"] .related-papers-image {
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/related_papers_dark.svg' | relative_url }}");
  }

  html[data-theme="dark"] .multiple-papers-image {
    background-image: url("{{ 'assets/img/2025-04-28-imagenet-flaws/multiple_papers_dark.svg' | relative_url }}");
  }

</style>

<span style="font-size:17px; font-weight:800">Disclaimer:</span>
By undertaking this work, we have no intention to diminish the significant contributions of ImageNet, whose value remains undeniable. It was, at the time of its publication, far ahead of all existing datasets. Given ImageNet-1k's continued broad use, especially in model evaluation, fixing the issues may help the field move forward. With current tools, we believe it is possible to improve ImageNet-1k without huge manual effort. 

    
## Introduction to ImageNet-1k

<div style="display: flex; gap: 15px; flex-wrap:wrap; justify-content:center;">
<figure style="margin: 0; text-align: left;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/otter.png' | relative_url }}" height="200px">
    <figcaption style="font-size: 13px;">(a) <span style="color:green; font-weight:bold;">"otter" ✓ </span></figcaption>
  </figure>
    <figure style="margin: 0; text-align: left;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tiger_shark.png' | relative_url }}" height="220px">
        <figcaption style="font-size: 13px;">(b) <span style="color:red; font-weight:bold;">"tiger shark" ×</span></figcaption>
        <figcaption style="font-size: 13px;">&emsp;&nbsp; <span style="color:green; font-weight:bold;">"grey whale" ✓</span></figcaption>
  </figure>

  <figure style="margin: 0; text-align: left;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/parallel_bars.png' | relative_url }}" height="220px">
      <figcaption style="font-size: 13px;">(c) <span style="color:red; font-weight:bold;">"parallel bars" ×</span></figcaption>
      <figcaption style="font-size: 13px;">&emsp;&nbsp; <span style="color:green; font-weight:bold;">"horizontal bar" ✓</span></figcaption>
  </figure>
  <figure style="margin: 0; text-align: left;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/wine_bottle.png' | relative_url }}" height="220px">
      <figcaption style="font-size: 13px;">(d) <span style="color:green; font-weight:bold;">"wine bottle" ✓</span></figcaption>
      <figcaption style="font-size: 13px;">&nbsp; <span style="color:green; font-weight:bold;">+ "goblet" ✓</span></figcaption>
      <figcaption style="font-size: 13px;">&nbsp; <span style="color:green; font-weight:bold;">+ "red wine" ✓</span></figcaption>
  </figure>
  <figure style="margin: 0; text-align: left;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/toilet_seat.png' | relative_url }}" height="220px">
      <figcaption style="font-size: 13px;">(e) <span style="color:red; font-weight:bold;">"toilet seat" ×</span></figcaption>
  </figure>

  <div style="display: flex; flex-direction: column; gap: 10px;">
    <figure style="margin: 0; text-align: left;">
      <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/race_car.png' | relative_url }}" height="105px">
        <figcaption style="font-size: 13px;">(f) <span style="color:green; font-weight:bold;">"race car" ✓</span></figcaption>
    </figure>
    <figure style="margin: 0; text-align: left;">
      <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/race_sports_car.png' | relative_url }}" height="105px">
        <figcaption style="font-size: 13px;">(g) <span style="color:red; font-weight:bold;">"sports car" ×</span></figcaption>
        <figcaption style="font-size: 13px;">&emsp;&nbsp; <span style="color:green; font-weight:bold;">"race car" ✓</span> </figcaption>
    </figure>
  </div>
</div>

<div style="margin: 20px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Figure 1. Examples of ImageNet issues. The top label is the ground truth; colors indicate <span style="color:green"> correct ✓</span>; and <span style="color:red">incorrect × </span> labels. When the ground truth is wrong, the correct label is below. Labels of objects from other ImageNet classes present in the image are marked "+". (a), (f) <em style="color:grey;">Correctly labeled images</em>. (b), (c), (e), (g) <em style="color:grey;">Incorrectly labeled images</em>. (d) <em style="color:grey;">Images with multiple objects</em>. (e) <em style="color:grey;">Ambiguous images</em>. (f), (g) <em style="color:grey;">Pixel-wise duplicates</em> (the top image is from the training, the bottom from the validation set). </div>

#### Brief History
The concept of **ImageNet**<d-cite key="2"/> was introduced in 2009. It was to become the first large-scale labeled image dataset, marking a transformative moment in computer vision. In its construction, the authors followed the structure of WordNet<d-cite key="3"/> — a lexical database that organizes words into a semantic hierarchy. 
With over 3 million labeled images from more than 5,000 categories in 2009, it far surpassed previous commonly used datasets, which contained only tens of thousands of images (e.g., LabelMe with 37,000 images, PASCAL VOC with 30,000 images, CalTech101 with 9,000 images).
This unprecedented scale allowed ImageNet to capture a diverse range of real-world objects and scenes. As of November 2024, the ImageNet<d-cite key="2"/> paper has since been cited over 76,000 times according to Google Scholar, and more than 34,000 times in IEEE Xplore, underscoring its profound impact on computer vision. Its influence is so significant that even in other fields of machine learning, the expression
"ImageNet moment" is used to describe groundbreaking developments.

Introduced in 2012, **ImageNet-1k**<d-cite key="1"/> was created specifically for the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). It contains 1,000 diverse classes of animals, plants, foods, tools, and environments (e.g., lakesides, valleys). It includes a training set of over 1 million images from the original ImageNet, along with validation and test sets of 50,000 and 100,000 images, respectively. The evaluation subsets were formed by a process following the original dataset creation. It is important to note that, since the test set labels are not publicly available, the validation set is typically used for model evaluation.

#### Problems
We were aware that ImageNet-1k had issues, which is common in real-world datasets. However, while analyzing model errors for another project, we were surprised by the size of the effects of the problems related to the ground truth. We decided to investigate them in greater detail. The initial goal looked simple: correct the labels to reduce noise — a task we assumed would be straightforward. However, upon further examination, we discovered that the issues were complex and far more deeply rooted.

The easy-to-solve issues, such as incorrect image labels, redundant classes, and duplicate images, can be addressed by relabeling, merging classes, and cleaning up duplicates. However, there are other issues, such as a distribution shift between the training and evaluation subsets, where we see no fully backward-compatible solution. The domain shift could even be seen as a feature rather than a bug since in real deployment, test and training data are never i.i.d. (independent and identically distributed).
The inclusion of images depicting objects from multiple ImageNet classes, called multilabel images, poses another problem with no straightforward solution. Their removal is a possibility when ImageNet-1k is used for training. However, their prevalence, which we estimate to be in the 15 to 21% range in the validation set, probably rules out adopting this solution. Other modifications of the evaluation protocol might be preferable, e.g. considering any predicted label from the multilabel ground-truth set as correct. Referred to as ReaL accuracy <d-cite key="5"/>, it has already been reported alongside the standard accuracy. 


## Known ImageNet-1k Issues

Prior studies identified and analyzed many ImageNet issues, but they each deal only with a specific concern and usually focus on the validation set. The topic that received the most attention is annotation errors, which distort the evaluation of model accuracy <d-cite key="4,6,5,7,9"/>. 
Overlapping class definitions were reported in Vasudevan et al.<d-cite key="8"/>. Duplicate images were mentioned in several papers <d-cite key="8, 11, 16"/>.

<details>
    <summary style="font-weight:800">Prior Work in Detail</summary>

<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">Imagenet Multilabel<d-cite key="4"/>:</span>
the authors reannotated a class-balanced subset of images, covering 40% of the validation set. The main focus was on assigning multiple labels to capture the presence of several objects from different ImageNet classes. The work highlighted the extent of the multilabel problem.
</div>

<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">Contextualizing Progress on Benchmarks<d-cite key="6"/>:</span> 
the authors reannotated 20% of validation images from each class with predictions from an ensemble of deep models, which were verified by non-expert annotators.
Their analysis revealed over 21% of images were multilabel. The work also lists groups of most commonly co-occurring classes, like "race car" and "car wheel" (see Figure 9).
</div>
    
<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">Imagenet Real<d-cite key="5"/>:</span> aimed at correcting all labels in the validation set. The original label was deemed correct if it agreed with the predictions of all 6 deep models selected by the authors. This was the case for approximately 50% of images. For the remaining images, labels were manually reviewed by five human annotators. 
About 15% of the images are reported as multilabel and about 6% as ambiguous (which were left unlabeled).
<div style="margin-top: 15px">
The training set was checked using BiT-L<d-cite key="16"/> model predictions, without human involvement. In 10-fold cross-validation, approximately 90% of the training images had labels consistent with model predictions.
</div>
</div>
    
<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">Label Errors<d-cite key="7"/>:</span> used
Confident Learning framework<d-cite key="18"/> and five Amazon Mechanical Turk (MTurk) workers for re-annotation. Images lacking workers' consensus — about 11% — were marked as "unclear".
 The label error on the validation set is estimated to be about 20%.
</div>
    
<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">Re-labelling ImageNet<d-cite key="9"/>:</span>
authors used the EfficientNet-L2<d-cite key="12"/> model trained on the JFT-300M dataset to reannotate the training set and to convert ImageNet’s single-label annotations into multilabel annotations with localized information.
</div>
    
<div style="margin:15px 0px 15px 0;">
<span style="font-weight:800">When Does Dough Become a Bagel?<d-cite key="8"/></span>
The study identifies duplicate images in the validation and training sets.
Interestingly, each image in a group of duplicates is found to have a different label. This indicates duplicates were removed only within the same class, incorrectly assuming that duplicate images cannot have different labels. The work emphasizes the importance of addressing not only duplicates but also near-duplicate images, e.g., similar photos from the same photoshoot.
</div>

<div class="related-papers-image"></div>
    
<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Figure 2. Error analyses <d-cite key="4,5,6,7"/> of the ImageNet validation set labels. <span style="display:inline-block;width:10px;height:10px;background-color:#A3D8A0;"></span> Single label image, original label is correct. <span style="display:inline-block;width:10px;height:10px;background-color:#4C8C99;"></span> Single label image, original label is incorrect, full agreement on correction. <span style="display:inline-block;width:10px;height:10px;background-color:#8BBEE8;"></span> Multilabel images. <span style="display:inline-block;width:10px;height:10px;background-color:#F8D76E;"></span> Single label image, inconsistent label corrections. <span style="display:inline-block;width:10px;height:10px;background-color:#E55353;"></span> Ambiguous, no agreement on the label.</div> 

<div>
The significant differences in the estimates for the percentages of some types of errors have multiple reasons, the major ones are the percentage of images seen, methodology used by human annotators, and the models used for label error prediction.
</div>
    
</details>


### Bringing the Errors Together

After downloading the available corrected labels from prior studies and analyzing them, we discovered that approximately 57.2% of the images in the validation set had been reviewed by multiple studies.
Further examination revealed that only 33% of the entire validation set has identical labels across all the studies. This finding reminds us that error-correction processes are not error-free. We analyzed the images with consistent labels, and for nearly 94% of them, the original labels were correct. The remaining 6% consists of ambiguous and multilabel images, as well as images where annotators fully agreed on a label different from the original.

<details>
    <summary style="font-weight:800">Results Evaluation Note</summary>
    
<div style="margin:15px 0px 0px 0;">
It should be noted that for the <span style="font-weight:800">Label Errors<d-cite key="7"/></span> paper (see Prior Work in Detail), the .json file containing MTurk decisions was used and evaluated with a modification from the original methodology. Instead of using the majority consensus (3+ out of 5 workers), only unanimous decisions of all 5 workers were considered.
</div>
</details>

<div class="multiple-papers-image"></div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 3. Top: The number of images checked by one <span style="display:inline-block;width:10px;height:10px;background-color:#F8D76E;"></span>, two <span style="display:inline-block;width:10px;height:10px;background-color:#4C8C99;"></span>, three <span style="display:inline-block;width:10px;height:10px;background-color:#8BBEE8;"></span> and four <span style="display:inline-block;width:10px;height:10px;background-color:#A3D8A0;"></span> papers. <br> Bottom: Images checked by more than one paper where the annotators agreed <span style="display:inline-block;width:10px;height:10px;background-color:#F2A477;"></span> and disagreed <span style="display:inline-block;width:10px;height:10px;background-color:#B784C6;"></span>.
</div>

This analysis highlights significant inconsistencies in the labeling of reannotated images. The discrepancies between the studies arise from each following its own methodology, leading to varying interpretations of the class definitions and different approaches to resolving issues encountered during annotation.

## Dataset Construction Issues

Let us first examine the two-step process used to construct ImageNet: 

1. **Image collection.** Images were scraped from the web and organized according to the WordNet hierarchy, which will be revisited later. Both automated and manual methods were involved: automated tools initially assigned labels based on metadata available (textual description, category) at the image source, often [flickr](https://flickr.com). 
2. **Annotation process**. The preliminary labels were then reviewed by MTurk workers, who were only asked to confirm whether the object with the given label was present in the image. Notably, no alternative labels were suggested, such as other similar classes. The MTurkers were shown the target synset definition and a link to Wikipedia.


**ImageNet-1k** consists of three sets: a *training set* with over 1 million images, a *validation set* with 50,000 images, and a *test set* with 100,000 images. The training set is drawn from ImageNet. Images for the evaluation subsets were obtained through a process that tried to replicate the one used for the ImageNet. The new data were collected up to three years later than training data and then they were randomly split between the validation and test sets.

 ImageNet links each image category to a **WordNet** noun synset. WordNet is a comprehensive lexical database of English. It organizes nouns, verbs, adjectives, and adverbs into synonym sets, or synsets, each representing a unique concept (see the official [website](https://wordnet.princeton.edu/)).

Each **synset** should consist of one or more terms referred to as "synonyms", for example, *"church, church building"*. However, they are not synonyms in all cases. For example, consider the synset *"diaper, nappy, napkin"*. The first two terms are synonyms, the third one is not. Moreover, there are cases where the same term is in more than one synset, e.g. there are two synsets named *"crane"* — one defining a bird and the second a machine. In ImageNet-1k, they are separate classes. Think about the consequences for zero-shot classification with vision-language models (VLMs) like CLIP<d-cite key="15"/>.

We will demonstrate some issues related to dataset construction with a few examples.

### The Cat Problem


<div style="display: flex; align-items: center; justify-content: center; row-gap: 15px; column-gap: 15px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tiger_cat_1.png' | relative_url }}" style="height: 140px;" />
        <p>(a)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tiger_cat_2.png' | relative_url }}" style="height: 140px; " />
        <p>(b)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tiger_cat_3.png' | relative_url }}" style="height: 140px;" />
        <p>(c)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tiger_cat_4.png' | relative_url }}" style="height: 140px;" />
        <p>(d)</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 4. Images from the <em style="color:grey;">"tiger cat"</em> class. (a) is a tiger cat. (b) is a cat, but the coat pattern is not clearly visible, it might be a <em style="color:grey;">"tabby, tabby cat"</em>, another ImageNet-1k class. (c) is a wild cat from the <em style="color:grey;">Leopardus</em> genus, not a domestic cat. (d) is a tiger.
</div>

Looking at the ***"tiger cat"*** images above, you might think "This seems to be a really diverse dataset...". Let us have a closer look. The *"tiger cat"* class is defined in WordNet as **"a cat having a striped coat"**, which aligns precisely with (a).

To understand the issue with (b), we must know that ImageNet-1k also includes a ***"tabby, tabby cat"*** class, defined as **"a cat with a grey or tawny coat mottled with black"**. In common usage, tabby cat refers broadly to any domestic cat with a striped, spotted, or swirled coat pattern, all of which must include an "M" marking on their forehead (which can be seen in this image). Most dictionaries agree that all tiger cats are tabbies, but not all tabby cats have the tiger pattern. However, even if we look at image (b) through the lens of WordNet definitions, it shows a grey cat, but its coat is not clearly visible. Moreover, the term "mottled coat" in the tabby cat definition can be somewhat confusing, as some dictionaries consider stripes to be a type of mottling. So, how do we determine which type of cat this is? 

We find modern large language models (LLMs) to be more accurate when handling such questions, so we asked them whether these two definitions overlap:

<div class="llm-cite">
    Yes, the "tabby, tabby cat" definition and the "tiger cat" definition overlap. While the first definition is broader in its description of coloration and pattern, the second one specifies the striped aspect, which is a common characteristic of the broader "mottled" description in the first.
</div>

-- *ChatGPT-4o* 

<div class="llm-cite">Yes - a tabby cat definition ("a cat with a grey or tawny coat mottled with black") overlaps with the tiger cat definition ("a cat having a striped coat") since "mottled with black" typically manifests as stripes in domestic cats.</div>

-- *Claude 3.5 Sonnet*


<div class="llm-cite">Yes, mottled patterns can sometimes include stripes, so the "tabby, tabby cat" definition can occasionally overlap with the "tiger cat" definition.</div>

-- *Microsoft Copilot*

This raises the question: *If WordNet definitions are not precise enough, what is the ultimate source for correct image labels? ChatGPT, Wikipedia, GBIF?*

We are not wildlife experts, but we may say that either an oncilla, ocelot, or margay is in \(c). While this might seem like harmless noise, common in such large datasets, these animals do appear more than once in the training set. In everyday language, *"tiger cat"* is more commonly used to refer to these wild cats than to striped domestic ones; however, these usages coexist simultaneously. 


We have already mentioned the WordNet definition of the *"tiger cat"* synset; WordNet also contains a ***"tiger cat, Felis tigrina"*** synset, defined as **"a medium-sized wildcat of Central America and South America having a dark-striped coat"**.
All three cat species we have mentioned as possible labels for \(c) fit the definition. Consistently annotating *"tiger cat"* images given such confusing background information is difficult for experts, and probably impossible for MTurkers.

Obviously, (d) is a tiger, which has a class, ***"tiger, Panthera tigris"***. Tigers make up a significant portion of the "tiger cat" class in both the training and validation sets.
Distinguishing between a tabby and a tiger is not a challenging recognition task and yet the non-expert annotators made many mistakes. This highlights the need to think carefully about the expertise of and guidelines for the annotators.


### The Laptop Problem

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/laptop.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"laptop, laptop computer"</em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/notebook.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"notebook, notebook computer"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Figure 5. The terms laptop and notebook are now used interchangeably. Black cat, unlike <em style="color:grey;">"tabby cat"</em> and <em style="color:grey;">"tiger cat"</em>, is not an ImageNet-1k class, which makes it a single-label image.</div>

Consider two synsets: *"laptop, laptop computer"* and *"notebook, notebook computer"*. Their respective WordNet definitions are "a portable computer small enough to use in your lap" and "a small compact portable computer". In this case, the definitions overlap, with the former being a subset of the latter. We again asked modern LLMs about the difference between laptop and notebook: 

<div class="llm-cite">Previously, "laptops" referred to portable computers with larger screens and more powerful hardware, while "notebooks" were slimmer, lighter, and less powerful. Today, the terms are used interchangeably, as advancements in technology have blurred the distinctions.</div>

-- *ChatGPT-4o*

This raises a question whether there is any other solution besides merging these two classes into a single class with 2600 training images or changing the evaluation protocol so that laptop -- notebook swaps are not penalized.

### Exploring VLM Results

We expected the issues described above to have an impact on the results of Vision Language Models. To test the hypothesis, we selected a zero-shot open-source model, OpenCLIP (ViT-H-14-378), and examined its predictions on the training set for the classes discussed above. The confusion matrices below show the discrepancies between the original and predicted labels.

<div style="display:flex; align-items:center; justify-content: center; margin:20px; font-size:14px;">
    <table style="border-collapse: collapse; text-align: center; border: 0px;">
      <tr>
        <th style="border: 0px; border-top: 0px solid black; padding: 8px;"></th>
        <th colspan="4" style="border: 0px; border-top: 0px solid black; border-bottom: 1px solid grey; padding: 8px; text-align: center; font-weight:900">Predicted Label (OpenCLIP) →</th>
      </tr>
      <tr>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:900; text-align: center;">Original Label ↓</th>
          <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600"><em>tabby, tabby cat</em></th>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600"><em>tiger cat</em></th>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600"><em>tiger, Panthera tigris</em></th>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600">Other classes</th>
      </tr>
      <tr>
        <th style="border: 0px solid black; padding: 8px; text-align: left; font-weight:600"><em>tabby, tabby cat</em></th>
        <td style="border: 0px solid black; padding: 8px;">76.4%</td>
        <td style="border: 0px solid black; padding: 8px;">8.5%</td>
        <td style="border: 0px solid black; padding: 8px;">0%</td>
                 <td style="border: 0px solid black; padding: 8px;">15.1%</td> 
      </tr>
      <tr>
        <th style="border: 0px solid black; padding: 8px; text-align: left; font-weight:600"><em>tiger cat</em></th>
        <td style="border: 0px solid black; padding: 8px;">57.2%</td>
        <td style="border: 0px solid black; padding: 8px;">6.9%</td>
        <td style="border: 0px solid black; padding: 8px;">23.8%</td>
                  <td style="border: 0px solid black; padding: 8px;">12.1%</td>
      </tr>
      <tr>
        <th style="border: 0px; border-bottom: 3px solid black; padding: 8px; text-align: left; font-weight:600"><em>tiger, Panthera tigris</em></th>
        <td style="border: 0px; border-bottom: 3px solid black; padding: 8px;">0.1%</td>
        <td style="border: 0px; border-bottom: 3px solid black; padding: 8px;">0%</td>
        <td style="border: 0px; border-bottom: 3px solid black; padding: 8px;">99.2%</td>
                  <td style="border: 0px; border-bottom: 3px solid black; padding: 8px;">0.7%</td>
      </tr>
    </table>
</div>   

<div style="margin: 20px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Table 1. OpenCLIP predictions for classes related to '<a style="color:grey;" href="#the-cat-problem">The Cat Problem</a>'.</div>

Note that the differences can be both due to OpenCLIP's errors and wrong ground truth labels. Nearly a quarter of the images in *"tiger cat"* class are predicted to be tigers, which we trust to be an estimate of the percentage of tigers in the training data of the class. Only 6.9% of the images are predicted as *"tiger cat"*, highlighting the conceptual overlap with *"tabby, tabby cat"*.

<div style="display:flex; align-items:center; justify-content: center; margin:20px; font-size:14px;">
    <table style="border-collapse: collapse; text-align: center; border: 0px;">
      <tr>
        <th style="border: 0px; border-top: 0px solid black; padding: 8px;"></th>
        <th colspan="3" style="border: 0px; border-top: 0px solid black; border-bottom: 1px solid grey; padding: 8px; text-align: center; font-weight:900">Predicted Label (OpenCLIP) →</th>
      </tr>
      <tr>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; text-align: center; font-weight:900">Original Label ↓</th>
          <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600"><em>laptop,<br/>laptop computer</em></th>
        <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600"><em>notebook,<br/> notebook computer</em></th>
                <th style="border: 0px; border-bottom: 2px solid grey; padding: 8px; font-weight:600">Other classes</th>
      </tr>
      <tr>
          <th style="border: 0px solid black; padding: 8px; text-align: left; font-weight:600"><em>laptop, laptop computer </em></th>
        <td style="border: 0px solid black; padding: 8px;">35.8%</td>
        <td style="border: 0px solid black; padding: 8px;">44.6%</td>
            <td style="border: 0px solid black; padding: 8px;">19.6%</td>
      </tr>
      <tr>
          <th style="border: 0px solid black; border-bottom: 3px solid black; padding: 8px; text-align: left; font-weight:600"><em>notebook, notebook computer</em></th>
        <td style="border: 0px solid black; border-bottom: 3px solid black; padding: 8px;">17.2%</td>
        <td style="border: 0px solid black; border-bottom: 3px solid black; padding: 8px;">65.8%</td>
            <td style="border: 0px solid black; border-bottom: 3px solid black; padding: 8px;">17%</td>
      </tr>
    </table>
</div>   

<div style="margin: 20px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Table 2. 
OpenCLIP predictions for classes related to <a style="color:grey;" href="#the-laptop-problem">'The Laptop Problem'</a>. </div>


Approximately 80% of the images in these classes were predicted as either a notebook or a laptop, with an error not far from random guessing. The remaining 20% were assigned to other labels. This will be discussed in the [section on multilabels](#images-deserving-multiple-labels).

### Key Takeaways 

The examples demonstrate that the incorrect labels are not caused just by random noise, but they are also a consequence of the dataset's construction process. WordNet is not error-free and fully consistent; the issues propagate to ImageNet. Also, some meanings shift over time, which is a problem in the era of VLMs. Perhaps WordNet and ImageNet should co-evolve.


Relying solely on MTurkers and using Wikipedia (a source that may be continuously edited by non-experts and may lack precise definitions) not only led to the inclusion of noisy labels but also sometimes distorted the very concepts that the classes were intended to represent. For example, the *"sunglasses, dark glasses, shades"* and *"sunglass"* classes represent the same object — sunglasses. While this is accurate for the "sunglasses", the "sunglass" class is defined in WordNet as "a convex lens that focuses the rays of the sun; used to start a fire".
This definition was lost during the dataset construction process, resulting in two classes representing the same concept.


## Distribution Shift Between Training and Validation Sets 

<div style="display: flex; align-items: center; justify-content: center; row-gap: 15px; column-gap: 15px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/canoe.png' | relative_url }}" style="height: 130px;" />
        <p>(a)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/kayak.png' | relative_url }}" style="height: 130px; " />
        <p>(b)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/canoe_paddle.png' | relative_url }}" style="height: 130px;" />
        <p>(c)</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/kayak_paddle.png' | relative_url }}" style="height: 130px;" />
        <p>(d)</p>
    </div>
</div>



<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 6. A distribution shift between the training and validation sets. (a) <em style="color:grey;">"canoe"</em> in the training set (actual canoe). (b) <em style="color:grey;">"canoe"</em> in the validation set (a kayak). (c) <em style="color:grey;">"paddle, boat paddle"</em> depicting a canoe. (d) <em style="color:grey;">"paddle, boat paddle"</em> depicting a kayak.
</div>


As mentioned earlier, for ImageNet-1k, additional images were collected using the same strategy as the original ImageNet. However, even with the same process, issues arose.

For example, in the training set, the *"canoe"* class mainly consists of images of canoes, but it also includes many images of kayaks and other types of boats. In contrast, **the *"canoe"* class in the validation set only contains images of kayaks, with no canoes at all**.

To clarify, the difference is not only in the boat shapes, with a kayak being flatter, but also in how they are paddled. A canoe is typically paddled in a kneeling position (though seated paddling is common) with a short single-bladed paddle, while a kayak is paddled from a seated position with a long double-bladed paddle. Interestingly, "paddle, boat paddle" is a separate class in ImageNet-1k.

<div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; align-items: center;">
    <div style="text-align: center; flex: 0 0 30%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/planetaria_1.png' | relative_url }}" style="height: 140px;" />
    </div>
    <div style="text-align: center; flex: 0 0 30%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/planetaria_2.png' | relative_url }}" style="height: 140px;" />
    </div>
    <div style="text-align: center; flex: 0 0 30%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/planetaria_3.png' | relative_url }}" style="height: 140px;" />
    </div>
    <div style="text-align: center; flex: 0 0 30%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/planetaria_4.png' | relative_url }}" style="height: 140px;" />
    </div>
    <div style="text-align: center; flex: 0 0 30%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/planetaria_5.png' | relative_url }}" style="height: 140px;" />
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 7. Images from the <em style="color:grey;">"planetarium"</em> class in the ImageNet validation set. 68% of validation images in this class depict this particular building.
</div>

The *"planetarium"* class exhibits another issue. In the validation set, 68% of the images (34 out of 50) feature the same planetarium in Buenos Aires. The building appears many times in the training set too, but the range of planetaria is much broader. Perhaps it is a beautiful location, and the authors enjoyed featuring it, but it is not i.i.d.

Here, the solution is fairly straightforward - collecting a more representative set of images.
Nevertheless, this breaks backward compatibility, rendering old and new results incomparable.

## Images Requiring Multiple Labels

We illustrate the problem of multilabel images with an extreme example. What do you think is the correct label for the following image?

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/computer_mouse.png' | relative_url }}" style="height: 400px; margin: 30px 30px 5px 30px;" />
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 8. An image from the <em style="color:grey;">"mouse, computer mouse"</em> class. Objects from at least 11 other ImageNet-1k classes are visible.
</div>

The following objects in the image have a class in ImageNet-1k:

<div style="display: flex; flex-wrap: wrap; margin-bottom: 20px;">
  <div style="flex: 1; min-width: 33.33%; padding-right: 20px;">
    - Computer keyboard<br>
    - Space bar<br>
    - Monitor<br>
    - Screen<br>
  </div>
  <div style="flex: 1; min-width: 33.33%; padding-right: 20px;">
    - Notebook<br>
    - Laptop<br>
    - Desktop computer<br>
    - Desk<br>
  </div>
  <div style="flex: 1; min-width: 33.33%;">
    - iPod<br>
    - Website<br>
    - Printer<br>
  </div>
</div>

One could argue that in the presence of multiple objects from distinct classes, the dominant should be labeled, but here and elsewhere, it is not clear what the dominant object is.

As with the *"canoe"* and *"paddle"* classes in the [section about domain shift](#distribution-shift-between-training-and-validation-sets), some objects naturally appear together in photos. In everyday usage, desktop computers are accompanied by a computer keyboard and a monitor (all of which are ImageNet-1k classes). The difference between a monitor and a screen, another pair of questionable ImageNet-1k classes, is an issue in its own right. Additionally, desktop computers are generally placed on desks (also a class), so these two objects often appear together in images.
Many cases of multilabel image issues stemming from frequently co-occurring classes exist.

The issue runs deeper, and the authors' claim that there is no overlap and no parent-child relationship between classes appears to be incorrect. 
Consider the example of a "spacebar" and a "computer keyboard". The "space bar" may not always be part of a "computer keyboard", but most "keyboards" do have a "space bar".

Let us look at another example.
    
<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/car_wheel.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"car wheel"</em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/sports_car.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"sports car, sport car"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 9. Classes <em style="color:grey;">"car wheel"</em> and<em style="color:grey;">"sports car, sport car"</em>. There are not many cars without a wheel.
</div>

A wheel can exist without a car, but a car — except for some rare cases, say in a scrapyard — cannot exist without a wheel. When an image contains both, and many do, it is unclear which label should take priority. Even if MTurkers were familiar with all 1000 ImageNet-1k classes, assigning a single accurate label would still be challenging.

As mentioned in the section about dataset construction, MTurkers were asked to determine whether an image contains an object that matches a given definition. Such a process of annotation may not be inherently problematic. However, when paired with the problematic class selection, our next topic, it is.

## ILSVRC Class Selection

The ImageNet-1k classes were chosen as a subset of the larger ImageNet dataset. One reason the dataset is so widely used is that it is perceived to reflect the diversity of the real world. 
Yet, the class distribution is skewed. Does having more than 10% of the dataset represent dog breeds truly capture the human experience as a whole, or is it more reflective of dog owners' perspective? Similarly, is having a separate class for *"iPod"* — rather than a broader category like *"music player"* — a durable representation of the world?

### Taxonomy of class selection problems

We categorize the problems with class selection as follows:

**Class Is a Subset or a Special Case of Another Class**
 - *"Indian elephant, Elephas maximus"* & *"African elephant, Loxodonta africana"* are also *"tusker"*
 - *"bathtub, bathing tub, bath, tub"* is also a *"tub, vat"*

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/bathtub.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"bathtub, bathing tub, bath, tub” </em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/tub.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"tub, vat”</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 10. An example of <em style="color:grey;">"bathtub, bathing tub, bath, tub”</em> and <em style="color:grey;">"tub, vat”</em>.
</div>

**Class Is a Part of Another Class Object**
   - *"space bar"* is a part of *"computer keyboard, keypad"*
   - *"car wheel"* is a part of any vehicle class (*"racer, race car, racing car"*, *"sports car, sport car"*, *"minivan"*, etc.)

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/spacebar.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"space bar"</em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/computer_keyboard.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"computer keyboard, keypad"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 11. An example of <em style="color:grey;">"space bar"</em> and <em style="color:grey;">"computer keyboard, keypad"</em>.
</div>

**Near Synonyms as Understood by Non-experts**
   - *"laptop, laptop computer"* & *"notebook, notebook computer"*
   - *"sunglasses, dark glasses, shades"* & *"sunglass"*

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/sunglasses.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"sunglasses, dark glasses, shades"</em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/sunglass.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"sunglass"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 12. An example of <em style="color:grey;">"sunglasses, dark glasses, shades"</em> and <em style="color:grey;">"sunglass"</em>.
</div>

**Mostly Occur Together**
   - *"sea anemone"* & *"anemone fish"*
   - *"microphone, mike"* & *"stage"*

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/sea_anemone.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"sea anemone"</em></p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/anemone_fish.png' | relative_url }}" style="height: 190px; margin-top: 30px;" />
        <p><em>"anemone fish"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 13. An example of <em style="color:grey;">"sea anemone"</em> and <em style="color:grey;">"anemone fish"</em>.
</div>

We ran hierarchical clustering on the confusion matrix of EfficientNet-L2 <d-cite key="9"/>. Next, we manually reviewed the clusters and selected those that fit the definition of one of the categories above. Ultimately, 48 clusters were selected, containing 151 classes in total (2-10 per cluster). In some cases, clusters belong to more than one of the problematic categories.

<div class="l-page">
  <iframe class="iframe-light" src="{{ 'assets/html/2025-04-28-imagenet-flaws/confusion_matrix.html' | relative_url }}" frameborder='0' scrolling='no' height="550px" width="100%"></iframe>
  <iframe class="iframe-dark" src="{{ 'assets/html/2025-04-28-imagenet-flaws/confusion_matrix_dark.html' | relative_url }}" frameborder='0' scrolling='no' height="550px" width="100%"></iframe>
</div>

<div style="margin: 0px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 14. An interactive confusion matrix for all ImageNet-1k classes, based on predictions made by the EfficientNet-L2 model, organized by clusters. The top-1 predicted classes match the ground truth for all classes except <em style="color:grey;">'projectile, missile,'</em> which is predicted as <em style="color:grey;">'missile'</em> for 56.85% of its images.
</div>


The full list of problematic categories is [here](https://github.com/klarajanouskova/ImageNet/tree/main/classes/problem_groups). 
OpenCLIP accuracies for both problematic and non-problematic groups of classes are given in **Table 3**.

| Dataset    | Overall Accuracy | Problematic Classes Accuracy | Non-problematic Classes Accuracy |
|------------|------------------|------------------------------|----------------------------------|
| Validation | 84.61%          | 73.47%                       | 86.59%                           |
| Training   | 86.11%          | 75.44%                       | 88.02%                          |

<div style="margin: 0px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Table 3. OpenCLIP accuracy on ImageNet-1k. </div>

The classes from problematic groups have a significantly lower OpenCLIP accuracy.

## Addressing Duplicates

<div style="display: flex; gap: 30px; align-items: stretch; min-height: 240px;">
    <!-- Left column - first wolf centered vertically -->
    <div style="flex: 0 0 auto; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <div>
            <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/val_lynx.png' | relative_url }}" style="height: 107px;" />
            <p style="font-size: 13px;"> <span style="color:red; font-weight:bold;">"lynx, ..." ×</span></p>
        </div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 15px;">
        <div style="display: flex; gap: 15px; text-align: center;">
            <div>
                <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/timber_wolf.png' | relative_url }}" style="height: 107px;" />
                <p style="font-size: 13px;"> <span style="color:green; font-weight:bold;">"timber wolf, ..." ✓</span></p>
            </div>
            <div>
                <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/dingo.png' | relative_url }}" style="height: 107px;" />
                <p style="font-size: 13px;"> <span style="color:red; font-weight:bold;">"dingo, ..." ×</span></p>
            </div>
            <div>
                <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/white_wolf.png' | relative_url }}" style="height: 107px;" />
                <p style="font-size: 13px;"> <span style="color:red; font-weight:bold;">"white wolf, ..." ×</span></p>
            </div>
        </div>
        <div style="display: flex; gap: 15px; text-align: center; justify-content: center;">
            <div>
                <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/red_wolf.png' | relative_url }}" style="height: 107px;" />
                <p style="font-size: 13px;"> <span style="color:red; font-weight:bold;">"red wolf, ..." ×</span></p>
            </div>
            <div>
                <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/coyote.png' | relative_url }}" style="height: 107px;" />
                <p style="font-size: 13px;"> <span style="color:red; font-weight:bold;">"coyote, ..." ×</span></p>
            </div>
        </div>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 15. An extreme example of duplicate images in ImageNet-1k. The leftmost image is from the validation set, while the others are from the training set. Caption colors indicate <span style="color:green"> correct ✓</span>; and <span style="color:red">incorrect × </span> labels. 
</div>

Of all prior studies, **When Does Dough Become a Bagel?<d-cite key="8"/>** examined the issue of duplicates most extensively. The paper identified 797 validation images that also appear in the training set, with some images occurring multiple times. They also highlighted the broader problem of near duplicates in ImageNet-1k (e.g. images from the same photoshoot). 
However, no statistics were provided since near duplicates are significantly more difficult to detect than identical images.

The *"planetarium"* class mentioned earlier is a great example. It contains many near-duplicate images, as was noted in the [section focused on distribution shift](#distribution-shift-between-training-and-validation-sets). Specifically, 68% of the validation images feature the same building in Buenos Aires. This observation naturally led us to investigate the issue of image duplicates more comprehensively.

Our analysis focuses on three types of duplicate sets:
1. **Cross-duplicates** between the validation and training sets (identified in earlier research).
2. **Duplicates within the validation set** (new findings).
3. **Duplicates within the training set** (new findings).

The search began with the duplicate candidate detection process. We then categorized duplicates into 2 groups: **exact duplicates** and **near duplicates**, and the results are surprising...

<details>
  <summary style="font-weight:800">Candidate Detection Details</summary>

<div style="margin:15px 0px 15px 0; font-weight:800">
Initial Approach: DINOv2 Over CLIP
</div>

<div style="margin:15px 0px 15px 0;">
We computed image embeddings using the DINOv2<d-cite key="13"/> model. We initially considered using CLIP for this, but its results were not satisfactory. Once the embeddings were generated, we applied the <em>K-Nearest Neighbors (K-NN) algorithm</em> to detect possible duplicates based on their similarity in the embedding space.
</div>

<div style="margin:15px 0px 15px 0; font-weight:800">
Duplicate Candidate Detection Method
</div>

<div style="margin:15px 0px 15px 0;">
The algorithm checks how close the embeddings of two images are. If the distance between them is less than a certain threshold (confidence level), we marked them as possible duplicates.
</div>
    
<div style="margin:15px 0px 15px 0;">
Let us break it down:
</div>
    
<ol>
<li>Each image $I_i$ has an embedding, $\mathbf{e}(I_i)$, in the feature space.</li>
<li>The K-NN algorithm finds the 5 closest neighbors for each image.</li>
<li>$d(\mathbf{e}(I_i), \mathbf{e}(I_j))$ represents the cosine distance between the embeddings of two images.</li>
<li>$\tau$ is a predefined confidence threshold chosen high enough to ensure that no true positives are lost.</li>
</ol>

<div style="margin:15px 0px 0px 0px;">
Mathematically, the condition for possible duplicates is:
<div style="margin:10px 0px 10px 0;">
$$ d(\mathbf{e}(I_i), \mathbf{e}(I_j)) \leq \tau $$
</div>
</div>
</details>

### Exact Duplicate Search: Pixel-Level Comparisons

#### Key Findings

- In the **validation set**, 29 duplicate pairs were found. Each image in a pair belonged to a different ImageNet class.
- In the **training set**, 5,836 distinct images fall into duplicate groups, with 2 to 4 images per group. Most of these groups (with 5,782 distinct images) contained images assigned to different classes.
This highlights that cross-class deduplication was not performed during the dataset creation.
- For the **cross-validation-training search**, we confirm that 797 images in the validation set had duplicates in the training set. All these duplicate groups also consisted of images assigned to different ImageNet classes which is in agreement with previous studies<d-cite key="8"/>.

#### Bonus
- In the **test set**, 89 duplicate pairs were found. 
    
    Since labels for the test set are not publicly available, we cannot determine whether the images in each pair have the same label or not. However, given that the test and validation sets were created simultaneously by splitting the collected evaluation data, we can infer that the situation is likely similar to the validation set. This suggests that each image in a pair belongs to a different class, which sets a lower bound on accuracy on the test set.
   
After finding exact duplicates, we removed them and recalculated the accuracies of two models: OpenCLIP and an ImageNet-pretrained CNN EfficientNetV2. We conducted three experiments. First, we removed all duplicate pairs in the validation set (Table 4, x Val). Next, we removed all duplicate images in the validation set that were also present in the training set (Table 4, x Cross). Finally, we combined these two methods to remove all exact duplicates (Table 4, x Val+Cross). In summary, our approach led to a 0.7% accuracy increase for the zero-shot model and a 1% accuracy increase for the pretrained CNN. We remind the reader that all exact duplicates have different labels and their erroneous classification is very likely; the improvement is thus expected.
<div style="margin-top: 20px;"></div>
    
| Model | Standard | × Val | × Cross | × Val+Cross |
| --- | --- | --- | --- | --- |
| OpenCLIP | 84.61 | 84.67 | 85.27 | 85.32 |
| EfficientNetV2 | 85.56 | 85.62 | 86.51 | 86.57 |

<div style="margin: 20px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Table 4. OpenCLIP and EfficientNet accuracies on the whole ImageNet-1k (standard) and without (×) different kinds of <em style="color:grey;">exact duplicates</em>.</div>


### Near Duplicate Detection Method

The initial automatic search for duplicates was followed by a careful manual review of duplicate candidate images. After the review, each image was classified into one of the following near-duplicate groups.

**Image Augmentations**: images that result from various transformations applied to the original image, such as cropping, resizing, blurring, adding text, rotating, mirroring, or changing colors. An example is shown below.
    
<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: auto auto; grid-template-areas: 'left right' 'center center'; row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="grid-area: left; text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/mouse.png' | relative_url }}" style="height: 160px; margin-top: 30px;" />
        <p>(a) <em>"computer mouse"</em> from the validation set</p>
    </div>
    <div style="grid-area: right; text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/mousetrap.png' | relative_url }}" style="height: 160px; margin-top: 30px;" style="height: 160px; margin-top: 30px;" />
        <p>(b) <em>"mousetrap"</em> from the training set</p>
    </div>
    <div style="grid-area: center; text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/difference.png' | relative_url }}" style="height: 160px; margin-top: 30px;" style="height: 160px; margin-top: 30px;" />
        <p>(c) pixels that differ in (a) and (b)</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 16. Near duplicates - Image Augmentation.
</div>

**Similar View**: images of the same object taken from slightly different angles at different times. An example is depicted below.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/same_view_1.png' | relative_url }}" style="height: 160px; margin-top: 30px;" style="height: 190px; margin-top: 30px;" />
        <p><em>"dam, dike, dyke"</em> from the validation set</p>
    </div>
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/same_view_2.png' | relative_url }}" style="height: 160px; margin-top: 30px;" style="height: 190px; margin-top: 30px;" />
        <p><em>"dam, dike, dyke"</em> from the training set</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 17. Near duplicates - Similar View.
</div>

#### Key Findings

- In the **validation set**, 26 near-duplicate groups were found, involving 69 images in total. All duplicates in a group had consistent labels.
- For the **cross validation-training search**, we discovered that 269 images from the validation set matched 400 training images.
    
We continued evaluating models with near duplicates removed. First, we removed all near duplicate groups in the validation set (Table 5, x Val). Next, we removed validation images that appeared in the training set (Table 5, x Cross), and then we removed both (Table 5, x Val + Cross). Lastly, we removed all exact duplicates and near duplicates from the validation set (Table 5, x All). As shown in **Table 5**, removing near duplicates had minimal impact on accuracy, as these images were mostly consistently assigned the same label within each duplicate group.
    
| Model | Standard | × Val | × Cross | × Val+Cross | × All |
| --- | --- | --- | --- | --- | --- |
| OpenCLIP | 84.61 | 84.60 | 84.63 | 84.62 | 85.32 |
| EfficientNetV2 | 85.56 | 85.54 | 85.59 | 85.59 | 86.59 |

<div style="margin: 20px 0px 20px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">Table 5. OpenCLIP and EfficientNet accuracies on the whole ImageNet-1k (Standard) and without (x) different kinds of duplicates. </div>


## Prompting Vision-Language Models
CLIP zero-shot classification is based on the distance of the image embeddings to the text embeddings representing each class. A natural approach is to create class embeddings based on the WordNet synset names. However, there are problems, such as overlapping synsets or WordNet synsets whose meanings evolve over time. For example, the term maillot appears in both *"maillot"* and *"maillot, tank suit"*. The first synset definition is "tights for dancers or gymnasts", while the second one is "a woman's one-piece bathing suit". 


#### Class Text Prompt Modifications: An Empirical Study

To illustrate the impact of class names in zero-shot recognition, we developed a new set of ["modified" class names](https://github.com/klarajanouskova/ImageNet/blob/main/classes/modified_classnames.txt), building on [OpenAI’s version](https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/notebooks/Prompt_Engineering_for_ImageNet.ipynb) of the ImageNet-1k class names. In the notebook, the authors suggest further work with class names is necessary. 
In the experiments, we decided to use OpenCLIP, an open-source implementation that outperforms the original CLIP model.

**Table 6** shows recognition accuracy for the five classes with the most significant gain when using OpenAI class names vs. the original ImageNet names. The changes primarily address CLIP's need for a broader context. For instance, in ImageNet, *"sorrel"* refers to a horse coloring, while in common usage, we’re used to hearing it refer to a [plant](https://en.wikipedia.org/wiki/Sorrel). This can be a problem for VLMs due to the lack of context, which in turn the new class name *"common sorrel horse"* provides.

<table style="margin-top: 20px;">
  <thead>
    <tr>
      <th colspan="2" style="text-align:center;">ImageNet Class Name (WordNet)</th>
      <th colspan="2" style="text-align:center;">OpenAI Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><em>"sorrel"</em></td>
      <td>0%</td>
      <td>98%</td>
      <td><em>"common sorrel horse"</em></td>
    </tr>
    <tr>
      <td><em>"bluetick"</em></td>
      <td>0%</td>
      <td>78%</td>
      <td><em>"Bluetick Coonhound"</em></td>
    </tr>
    <tr>
      <td><em>"redbone"</em></td>
      <td>0%</td>
      <td>78%</td>
      <td><em>"Redbone Coonhound"</em></td>
    </tr>
    <tr>
      <td><em>"rock beauty, Holocanthus tricolor"</em></td>
      <td>22%</td>
      <td>96%</td>
      <td><em>"rock beauty fish"</em></td>
    </tr>
    <tr>
      <td><em>"notebook, notebook computer"</em></td>
      <td>16%</td>
      <td>66%</td>
      <td><em>"notebook computer"</em></td>
    </tr>
  </tbody>
</table>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Table 6. OpenCLIP zero-shot recognition accuracy with ImageNet (left) and OpenAI text prompts (right).
</div>

**Table 7** demonstrates the improvement of our modifications w.r.t. OpenAI's class names. Notably, renaming *"coffee maker"* to *"coffeepot"* not only increased accuracy within this class but also positively impacted the class *"espresso machine"*, where no changes were made. 

<table style="margin-top: 20px;">
  <thead>
    <tr>
      <th colspan="2" style="text-align:center;">OpenAI Class Name</th>
      <th colspan="2" style="text-align:center;">"Modified" Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><em>"canoe"</em></td>
      <td>48%</td>
      <td>100%</td>
      <td><em>"kayak"</em></td>
    </tr>
    <tr>
      <td><em>"vespa"</em></td>
      <td>42%</td>
      <td>82%</td>
      <td><em>"motor scooter"</em></td>
    </tr>
    <tr>
      <td><em>"coffeemaker"</em></td>
      <td>48%</td>
      <td>84%</td>
      <td><em>"coffeepot"</em></td>
    </tr>
    <tr>
      <td><em>"sailboat"</em></td>
      <td>76%</td>
      <td>100%</td>
      <td><em>"yawl (boat)"</em></td>
    </tr>
    <tr>
      <td><em>"espresso machine"</em></td>
      <td>50%</td>
      <td>72%</td>
      <td><em>"espresso machine"</em></td>
    </tr>
  </tbody>
</table>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Table 7. OpenCLIP zero-shot recognition accuracy with OpenAI (left) and text prompts "modified" by us (right). The "canoe" ImageNet-1k class achieves 100% accuracy if prompted by 'kayak'. This is not surprising, given that <em style="color:grey;">all images in the "canoe" validation set depict kayaks</em>. There is no 'kayak' class in ImageNet-1k.
</div>

Our modifications were found by trial and error, which suggests that there is a large space for possible improvement in VLM text prompting.

## Fixing ImageNet Labels: A Case Study

Do you know the precise difference between a weasel, mink, polecat, black-footed ferret, domestic ferret, otter, badger, tayra, and marten? Most likely not. We use these animal species to illustrate the complexity of image labeling in ImageNet-1k.

With the help of an expert, images from the following 4 classes are considered for re-annotation:
<ol style="margin-top: 0;">
  <li style="margin-bottom: 5px;"><i>"weasel"</i></li>
  <li style="margin-bottom: 5px;"><i>"mink"</i></li>
  <li style="margin-bottom: 5px;"><i>"polecat, fitch, foulmart, foumart, Mustela putorius"</i></li>
  <li style="margin-bottom: 5px;"><i>"black-footed ferret, ferret, Mustela nigripes"</i></li>
</ol>

<div style="display: flex; flex-wrap:wrap; gap: 10px; align-items: center; justify-content;center; ">
    <div style="text-align: center; flex: 1 1 48%; box-sizing: border-box;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/weasel.png' | relative_url }}" style="height: 190px; " />
        <p><em class="weasel-caption">"weasel"</em></p>
    </div>
<div style="text-align: center; box-sizing: border-box; flex: 1 1 48%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/mink.png' | relative_url }}" style="height: 190px;" />
    <p><em class="weasel-caption">"mink"</em></p>
    </div>
    <div style="text-align: center; box-sizing: border-box; flex: 1 1 48%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/polecat.png' | relative_url }}" style="height: 190px; " />
        <p><em class="weasel-caption">"polecat, fitch, foulmart, foumart, Mustela putorius"</em></p>
    </div>

<div style="text-align: center; box-sizing: border-box; flex: 1 1 48%;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/ferret.png' | relative_url }}" style="height: 190px; " />
    <p><em class="weasel-caption">"black-footed ferret, ferret, Mustela nigripes"</em></p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 18. Correctly labeled ImageNet-1k images from the <em>"weasel"</em>, <em>"mink"</em>, <em>"polecat"</em>, and <em>"black-footed ferret"</em> classes.       
The ground truth labels for these categories are often wrong, mainly confusing these species. 
</div>

These classes have a high percentage of incorrect ground truth labels, both in the training and validation sets. Most of the errors are caused by confusion between the four classes but the sets also contain images depicting animals from other ImageNet-1k classes, such as otter or badger, as well as images from classes not in ImageNet-1k, e.g. vole or tayra. But that is not the sole issue.

### The Weasel Problem

Let us look at the *"weasel"* class definitions:
- **WordNet**: small carnivorous mammal with short legs and elongated body and neck. 
- **[Wikipedia](https://en.wikipedia.org/wiki/Weasel)**: the English word weasel was originally applied to one species of the genus, the European form of the least weasel (*Mustela nivalis*). This usage is retained in British English, where the name is also extended to cover several other small species of the genus. However, in technical discourse and in American usage, the term weasel can refer to any member of the genus, the genus as a whole, and even to members of the related genus *Neogale*. 
- **[Webster](https://www.merriam-webster.com/dictionary/weasel)**: any of various small slender active carnivorous mammals (genus *Mustela* of the family *Mustelidae*, the weasel family) that can prey on animals (such as rabbits) larger than themselves, are mostly brown with white or yellowish underparts, and in northern forms turn white in winter.

The definition of the *"weasel"* synset in WordNet is broad - it potentially encompasses all the other mentioned classes. Moreover, the interpretation of the term weasel varies, between UK and US English, further complicating its consistent application. In US English, the term weasel often refers to the whole *Mustelidae*, also called 'the weasel family'. All of the following - weasel, mink, European polecat, and black-footed ferret - belong to the weasel family, as understood by US English.
One possible solution is to define the *"weasel"* class more precisely as the subgenus *Mustela*, which contains the 'least weasel' and other very similar species, which would lead only to the removal of a few images.

<div class="theme-image"></div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 19. A simplified branching diagram showing the <em style="color:grey;">Mustelidae</em> family, also referred to as 'the weasel family', or simply 'weasels' in US English. Some of ImageNet classes are: <em style="color:grey;">"weasel"</em>, <em style="color:grey;">"polecat"</em>, <em style="color:grey;">"black-footed ferret"</em> and <em style="color:grey;">"mink"</em>. All of these belong to the weasel family, pointing at yet another issue with class names. Images of these classes also depict species outside of ImageNet, such as tayras or voles, or even different ImageNet classes, such as otter or badger. The diagram shows evolutionary relationships between selected species from the ImageNet dataset; the circles mark higher taxonomic units.
</div>


### The Ferret Problem

Another complication arises with the *"black-footed ferret"* and *"polecat"* classes:
- **WordNet synset name:** *"black-footed ferret, ferret, Mustela nigripes"*   
- **ferret, [Webster](https://www.merriam-webster.com/dictionary/ferret):** 'a domesticated usually albino, brownish, or silver-gray animal (*Mustela furo* synonym *Mustela putorius furo*) that is descended from the European polecat'.
- **black-footed ferret, [Wikipedia](https://en.wikipedia.org/wiki/Black-footed_ferret):** 'the black-footed ferret is roughly the size of a mink and is similar in appearance to the European polecat and the Asian steppe polecat'.

The synset *"black-footed ferret, ferret, Mustela nigripes"* includes both the term 'black-footed ferret' and 'ferret'. The latter refers to a domesticated variety of the European polecat.
Consequently, the term 'ferret' is ambiguous; it may be understood both as a synonym for the black-footed ferret or as the domesticated polecat. Additionally, the domestic ferret and European polecat are nearly indistinguishable to non-specialists; even experts may face difficulties because these species can interbreed. 
There is also a potential for contextual bias in labeling, as ferrets are commonly found in domestic environments or in the presence of humans.


<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/eu_polecat.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>European polecat</p>
    </div>
<div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/dom_polecat.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>domestic ferret</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;"> Figure 20. Training set images from the ImageNet-1k <em style="color:grey;">"polecat"</em> class. Distinguishing between European polecat and domestic ferret is challenging due to the similarity in appearance. European polecats tend to be more muscular than domestic ferrets, and have overall darker fur and a well-defined white face mask.
</div>

To make matters worse, in the **validation set for the class *"black-footed ferret"*, only one image depicts this species!** A solution to this problem thus requires not only removal, or transfer to the correct class, of the incorrectly labeled images, but also collection of new data.

The term polecat presents a similar ambiguity w.r.t. the term ferret, as it is currently included in two synsets. One synset refers to skunk (family *Mephitidae*), while the other to *Mustela putorius*, the European polecat. These are in line with the definitions of the word polecat [here]( https://www.merriam-webster.com/dictionary/polecat).

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/eu_polecat_2.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>European polecat</p>
    </div>
<div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/skunk.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>skunk</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;"> Figure 21. European polecat and skunk. These species share the term 'polecat' in their synsets but belong to different families (see Figure 19).
</div>

To solve the 'ferret' issues, redefinition of classes might be needed, e.g.: 
1. Introduction of a distinct class for ferret, specifically denoting the domesticated form of the European polecat.
2. Reclassification of the term polecat so that it no longer appears in the synset for skunk; instead, this term should be used to represent a broader category encompassing both the European polecat and American polecat (also referred to as the black-footed ferret), as well as other species, such as the marbled, steppe, and striped polecats.
3. Create a class that encompasses both polecats and ferrets.


### Results of Relabeling

After relabeling the weasel family classes, we found that only the *"mink"* class had more than 50% of the original labels correct.
The percentage of the **correctly** labeled images in ImageNet-1k was: 


| Class Name                     | Percentage Correctly Labeled |
|--------------------------------|------------------------------|
| Weasel                         | 44%                          |
| Mink                           | 68%                          |
| Polecat                        | 32%                          |
| Black-footed ferret            | 2%                           |

The misclassified images either show an animal from the aforementioned classes or from a different ImageNet class (such as otter or badger). There are also images of animals outside of ImageNet-1k classes, while some images are ambiguous, see Figure 22.

Images of animals that do not belong to any ImageNet class are assigned to the 'non-ImageNet' label in the graph shown in Figure 23. This category includes animals such as vole, tayra, marten, and Chinese ferret-badger. Although 'domestic ferret' is also a non-ImageNet label, it is shown separately because of its large representation in the sets.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); row-gap: 15px; column-gap: 35px; justify-items: center;">
    <div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/obscured.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>characteristic features obscured</p>
    </div>
<div style="text-align: center;">
        <img src="{{ 'assets/img/2025-04-28-imagenet-flaws/distance.png' | relative_url }}" style="height: 190px; width: 300px; margin-top: 30px;" />
        <p>too far for identification</p>
    </div>
</div>

<div style="margin: 15px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;"> Figure 22. Images from the validation set <em style="color:grey;">"mink"</em> class, which our expert labeled as ambiguous. 
Left: the characteristic features are obscured; most likely a mink, but possibly a species from genus <em style="color:grey;">'Martes'</em>.
Right: an animal from genus <em style="color:grey;">'Mustela'</em>; possibly a mink or another dark-furred species. 
</div>

The 'ambiguous' label is used for images that are blurry, have the characteristic features of the species obscured, show the species from too great a distance, or have other flaws that prevent unequivocal identification of the species.

Let us take a closer look at the four examined classes.

<div class="weasel-image"></div>
<div style="margin: 0px 0px 25px 0px; color:grey; font-size:14px; font-weight:600; line-height: 1.3;">
    Figure 23. Validation set relabeling results for the <em style="color:grey;">"weasel"</em>, <em style="color:grey;">"mink"</em>, <em style="color:grey;">"polecat, fitch, foulmart, foumart, Mustela putorius"</em>, and <em style="color:grey;">"black-footed ferret, ferret, Mustela nigripes"</em> classes.
</div>

The weasel class contains a wide variety of misclassified images. This includes minks (6%), polecats (16%), domestic ferrets (10%), otters (6%), and badgers (6%). The high rate of misclassification may be due to the unclear definition of this class, as all of these species belong to the weasel family discussed earlier.

The *"mink"* class is predominantly correctly labeled but a substantial portion (30%) of images is ambiguous; meaning they are low quality or the subject is obscured. These images should preferably be removed or assigned multiple possible labels (single object but ambiguous).

The *"polecat"* class has a significant percentage (40%) of images depicting domestic ferrets. That is not surprising as distinguishing between polecats and domestic ferrets is particularly challenging.

Finally, the *"black-footed ferret"* class contains only one image of this species, while the majority (80%) of the images depict domestic ferrets.

Luccioni and Rolnick (2022)<d-cite key="19"/> analyzed the classes representing wildlife species and the misrepresentation of biodiversity within the ImageNet-1k dataset. Their findings reveal a substantial proportion of incorrect labels across these classes. Notably, they examined the class *"black-footed ferret"* and reported results consistent with those observed in our relabeling process.

## Conclusions
The blogpost deals with the problems of the ImageNet-1k dataset, both pertaining to image data selection and the ground truth labels. Most of the issues have been known to exist, like the presence of: incorrectly labeled images, multilabel images with no obvious correct label, overlapping class definitions, and the presence of intra-set duplicates and near-duplicate instances.

The analysis here is more in-depth provides new quantitative information about the phenomena and presents new findings, e.g. about the distribution shift between the training and validation sets, intra-set duplicates, and near-duplicate instances within both training and validation sets. We also introduce new, improved class names for zero-shot recognition with VLMs, as well as a refined taxonomy of problematic classes—such as hierarchical and duplicate classes—while introducing new ones, e.g. “part-of” and co-occurrence relationships.

For many problems, we provide exemplars, such as the expert-led case study of closely related *weasel family* species, which highlights the complexity of fine-grained classification. The training-validation domain shift is demonstrated on the *canoe & kayak* and *planetaria* classes. We discuss the causes of the problems, some rooted in ImageNet’s dataset collection process, particularly the reliance on untrained Mechanical Turk workers, and some originating from ambiguous WordNet definitions, exemplified by the *tiger cat* class.

At a technical level, we consolidate and publish prior ImageNet-1k label error corrections, resulting in the [“clean validation set”](https://github.com/klarajanouskova/ImageNet/blob/main/eval_corrections/verify_images/results/clean_validation.csv) with consistent image labels from previous studies, and provide the improved VLM zero-shot prompts for all ImageNet classes.

## Outlook
The blog mainly focuses on a precise description of the issues and their scale, i.e., what fraction of the image-label pairs it affects. We hope that one of the outcomes of publishing this analysis will be a broader exchange of ideas on what is and what is not worth fixing. Every solution will involve trade-offs, at least between correctness and backward compatibility. The wide use of ImageNet-1k makes it difficult to assess the full impact of any change; possibly we are heading towards multiple evaluation protocols and ground-truth versions.

The activity opened many questions. First and foremost: "Is this worth it?" and "Will the community benefit from (much) more accurate ImageNet-1k re-labeling and class definitions?". For the validation set, which is used for performance evaluation of a wide range of models, the answer seems a clear "yes". For the training set, we see benefits too. For instance, it seems that learning a fine-grained model from very noisy data is very challenging. The answers to the questions above depend on the effort needed to re-label the images and to clean the class definitions. Our experience is that current tools, both state-of-the-art classifiers and zero-shot VLM models, reduce the need for manual effort significantly. 

The effort to find precise, unambiguous definitions of the ImageNet-1k classes lead us to the use of VLMs and LLMs. The LLM responses were accurate and informative if prompted properly, even warning about common causes of confusion. Our experience suggests that LLMs are very suitable for both annotator training and decision support.

We hired an expert annotator in order to obtain precise annotation of the weasel-like animal classes analyzed in the case study. Expert annotators help identify subtle nuances and complexities within the dataset that might be easily overlooked by non-specialists. On the other hand, their understanding of certain terms might not coincide with common usage. We might need parameterizable VLM models, e.g., for professional and technical use as well as for the vernacular. Prior work<d-cite key="7"/> used MTurkers to find incorrect labels. However, we found that they missed many problems. Their errors are not independent, and attempts to remove them by e.g. majority voting or consensus, failed to detect them. When it comes to highly accurate labels, an expert is worth not a thousand, but any number of MTurkers.

In many areas of computer vision, models reached accuracy comparable to the so-called ground truth, losing the compass pointing to better performance. As we have seen, improving ground truth quality is not a simple task of checking and re-checking, but touches on some core issues of both vision and language modeling. This blog is a small step towards resetting the compass for ImageNet-1k.
