---
layout: distill
title: Understanding Model Calibration - A gentle introduction and visual exploration of calibration and the expected calibration error (ECE)
description: To be considered reliable, a model must be calibrated so that its confidence in each decision closely reflects its true outcome. In this blogpost we'll take a look at the most commonly used definition for calibration and then dive into a frequently used evaluation measure for model calibration. We'll then cover some of the drawbacks of this measure and how these surfaced the need for additional notions of calibration, which require their own new evaluation measures. This post is not intended to be an in-depth dissection of all works on calibration, nor does it focus on how to calibrate models. Instead, it is meant to provide a gentle introduction to the different notions and their evaluation measures as well as to re-highlight some  issues with a measure that is still widely used to evaluate calibration.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
- name: Maja Pavlovic
  url: "https://majapavlo.github.io"
  affiliations:
    name: Queen Mary University London

# must be the exact same name as your blogpost
bibliography: 2025-04-28-calibration.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What is Calibration?
    subsections:
    - name: (Confidence) Calibration
  - name: Evaluating Calibration - Expected Calibration Error (ECE)
    subsections:
    - name: ECE - Visual Step by Step Example
  - name: Most frequently mentioned Drawbacks of ECE
    subsections:
    - name: Pathologies - Low ECE ≠ high accuracy
    - name: Binning Approach
    - name: Only maximum probabilities considered
        #   subsections:
        #   - name: Multi-class Calibration
        #   - name: Class-wise Calibration
        #   - name: Human Uncertainty Calibration
  - name: Final Thoughts

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

## What is Calibration?
__Calibration__ makes sure that a model's estimated probabilities match real-world likelihoods. For example, if a weather forecasting model predicts a 70% chance of rain on several days, then roughly 70% of those days should actually be rainy for the model to be considered well calibrated <d-cite key="dawid1982well, degroot1983comparison"></d-cite>. This makes model predictions more _reliable_ and _trustworthy_, which makes calibration relevant for many applications across various domains.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f1_reliability_diagram.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 1 | Reliability Diagram
</div>

Now, what __calibration__ means more precisely depends on the specific definition being considered. 
We will have a look at the most common notion in machine learning (ML) formalised in <d-cite key="guo2017calibration"></d-cite> and 
termed __*confidence calibration*__ in <d-cite key="kull2019beyond"></d-cite>. But first, let's define a bit of formal notation for this blog. 
In this blogpost we consider a classification task with $$K$$ possible classes, with labels $$Y \in \{1, ..., K\}$$ and a classification model $$\hat{p} : \mathscr{X} \rightarrow \Delta^K $$, that takes inputs in $$\mathscr{X}$$ (e.g. an image or text) and returns a probability vector as its output. $$\Delta^K $$ refers to the *K*-simplex, which just means that the elements of the output vector must sum to 1 
and that each estimated probability in the vector is between 0 & 1. These individual probabilities (_or confidences_) indicate how likely 
an input belongs to each of the $$K$$ classes.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f2_notation_updated.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 2 | Notation - input example sourced from <d-cite key="uma2021learning"></d-cite>
</div>

### (Confidence) Calibration
A model is considered confidence-calibrated if, for all confidences $$c,$$ the model is correct $$c$$ proportion of the time:

$$  \mathbb{P} (Y  = \text{arg max}(\hat{p}(X)) \; | \; \text{max}(\hat{p}(X))=c ) = c \;\;\:\:  \forall c \in [0, 1] \; ,$$

<div class="caption">
where $(X,Y)$ is a datapoint and $\hat{p} : \mathscr{X} \rightarrow \Delta^K$ returns a probability vector as its output
</div>


This definition of calibration, ensures that the model's final predictions align with their observed accuracy at that 
confidence level <d-cite key="guo2017calibration"></d-cite>. The left chart below visualises the perfectly calibrated outcome (green diagonal line) for all confidences using  a binned reliability diagram <d-cite key="guo2017calibration"></d-cite>. On the right hand side it shows two examples for a specific confidence level across 10 samples.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f3_confidence_calibration.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 3 | Confidence Calibration
</div>

For simplification, we assume that we only have 3 classes as in image 2 and we zoom into confidence $$c=0.7$$, see image above. Let's assume we have 10 inputs here whose most confident prediction (*max*) equals $$0.7$$. If the model correctly classifies 7 out of 10 predictions (*true*), it is considered calibrated at confidence level $$0.7$$. For the model to be fully calibrated this has to hold across all confidence levels from 0 to 1. At the same level $$c=0.7$$, a model would be considered miscalibrated if it makes only 4 correct predictions.

## Evaluating Calibration - Expected Calibration Error (ECE)
One widely used evaluation measure for confidence calibration is the Expected Calibration Error (ECE) <d-cite key="naeini2015obtaining, guo2017calibration"></d-cite>. ECE measures how well a model's estimated probabilities match the observed probabilities by taking a weighted average over the absolute difference between average accuracy (*acc*) and average confidence (*conf*). The measure involves splitting all $$n$$ datapoints into M equally spaced bins:

$$  ECE = \sum_{m=1}^M \frac{\mathopen| B_m \mathclose|}{n} \mathopen| acc(B_m) - conf(B_m) \mathclose| , $$

where $$B$$ is used for representing "bins" and $$m$$ for the bin number, while *acc* and *conf* are:

$$ \small{ acc(B_m) = \frac{1}{ \mathopen| B_m \mathclose|} \sum_{i\in B_m} \mathbb{1} (\hat{y}_i = y_i ) \;\: \text{&} \;\: conf(B_m) = \frac{1}{ \mathopen| B_m \mathclose|} \sum_{i\in B_m} \hat{p}(x_i) } $$

$$\hat{y}_i$$ is the model's predicted class (*arg max*) for sample $$i$$ and $$y_i$$ is the true label for sample $$i$$. $$\mathbb{1}$$ is an indicator function, meaning when the predicted label $$\hat{y}_i$$ equals the true label $$y_i$$ it evaluates to 1, otherwise 0.
Let's  look at an example, which will clarify *acc*, *conf* and the whole binning approach in a visual step-by-step manner.

### ECE - Visual Step by Step Example

In the image below, we can see that we have $$9$$ samples indexed by $$i$$ with estimated probabilities $$\hat{p}(x_i)$$ (simplified as $$\hat{p}_i$$) for class __cat (C)__, __dog (D)__ or __toad (T)__. The final column shows the true class $${y}_i$$ and the penultimate column contains the predicted class $$\hat{y}_i$$.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f4_ece_table1.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Table 1 | ECE - Toy Example
</div>

Only the maximum probabilities, which determine the predicted label are used in ECE <d-cite key="guo2017calibration"></d-cite>. Therefore, we will only bin samples based on the maximum probability across classes (*see left table in below image*). To keep the example simple we split the data into 5 __equally spaced__ bins $$M=5$$. If we now look at each sample's maximum estimated probability, we can group it into one of the 5 bins (*see right side of image below*).

<div class="l-body-outset">
        {% include figure.html path="assets/img/2025-04-28-calibration/f5_ece_table2.png" class="img-fluid rounded " %}
</div>
<div class="caption">
    Table 2 & Binning Diagram
</div>

We still need to determine if the predicted class is correct or not to be able to determine the average accuracy per bin. If the model predicts the class correctly (i.e. $$y_i =\hat{y}_i$$), the prediction is highlighted in green; incorrect predictions are marked in red:

<div class="l-body-outset">
        {% include figure.html path="assets/img/2025-04-28-calibration/f6_ece_table3.png" class="img-fluid rounded" %}
</div>
<div class="caption">
    Table 3 & Binning Diagram
</div>

We now have visualised all the information needed for ECE and will briefly run through how to calculate the values for bin 5 ($$B_5$$). The other bins then simply follow the same process, see below.

<div class="l-page-outset">
        {% include figure.html path="assets/img/2025-04-28-calibration/f7_ece_table4.png" class="img-fluid rounded" %}
</div>
<div class="caption">
    Table 4 & Example for bin 5
</div>

We can get the empirical probability of a sample falling into $$B_5$$ , by assessing how many out of all $$9$$ samples fall into $$B_5$$, see $$\mathbf{(\;1\;)}$$. We then get the average accuracy for $$B_5$$, see $$\mathbf{(\;2\;)}$$ and lastly the average estimated probability for $$B_5$$, see $$\mathbf{(\;3\;)}$$. Repeat this for all bins and in our small example of $$9$$ samples we end up with an ECE of $$0.10445$$. A perfectly calibrated model would have an ECE of 0.

#### Expected Calibration Error Drawbacks
The images of binning above provide a visual guide of how ECE could result in very different values if we used more bins or perhaps binned the same number of items instead of using equal bin widths. Such and more drawbacks of ECE have been highlighted by several works early on <d-cite key="kumar2018trainable, nixon2019measuring, gupta2020calibration, zhang2020mix, roelofs2022mitigating, vaicenavicius2019evaluating, widmann2019calibration, futami2024informationtheoretic"></d-cite>. However, despite the known weaknesses ECE is still widely used to evaluate confidence calibration in ML <d-cite key="xiong2023can, yuan2024does, collins2023human, si2023prompting, mukhoti2023deep, gao2024spuq"></d-cite>. This motivated this blogpost, with the idea to highlight the most frequently mentioned drawbacks of ECE visually and to provide a simple clarification on the development of different notions of calibration.

## Most frequently mentioned Drawbacks of ECE

### Pathologies - Low ECE ≠ high accuracy
A model which minimises ECE, does not necessarily have a high accuracy <d-cite key="kumar2018trainable, kull2019beyond, si2022re"></d-cite>. 
For instance, if a model always predicts the majority class with that class's average prevalence as the probability, 
it will have an ECE of 0. This is visualised in the image above, where we have a dataset with 10 samples, 7 of those are cat, 2 dog and only one is a toad. Now if the model always predicts cat with on average 0.7 confidence it would have an ECE of 0. 
There are more of such pathologies <d-cite key="nixon2019measuring"></d-cite>. To not only rely on ECE, some researchers use additional measures such as the Brier score or LogLoss alongside ECE <d-cite key="kumar2018trainable, kull2019beyond"></d-cite>.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f8_pathologies.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 4 | Pathologies Example
</div>

### Binning Approach

One of the most frequently mentioned issues with ECE is its sensitivity to the change in binning <d-cite key="kumar2018trainable, nixon2019measuring, gupta2020calibration, zhang2020mix, roelofs2022mitigating, famiglini2023towards"></d-cite>. 
This is sometimes referred to as the __*Bias-Variance trade-off*__ <d-cite key="nixon2019measuring, zhang2020mix"></d-cite>: 
Fewer bins reduce variance but increase bias, while more bins lead to sparsely populated bins increasing variance. 
If we look back to our ECE example with 9 samples and change the bins from 5 to 10 here too, 
we end up with the following:

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f9_binning_1.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 5 | More Bins
</div>

We can see that bin *8* and *9* each contain only a single sample and also that half the bins now contain 
no samples. The above is only a toy example, however since modern models tend to have higher confidence values 
samples often end up in the last few bins <d-cite key="naeini2015obtaining, zhang2020mix"></d-cite>, which 
means they get all the weight in ECE, while the average error for the empty bins contributes 0 to ECE.

To mitigate these issues of fixed bin widths some authors <d-cite key="nixon2019measuring, roelofs2022mitigating"></d-cite> 
have proposed a more adaptive binning approach.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f10_binning_2.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 6 | Adaptive Bins
</div>

Binning-based evaluation with bins containing an equal number of samples are shown to have 
*lower bias* than a fixed binning approach such as ECE <d-cite key="roelofs2022mitigating"></d-cite>. 
This leads <d-cite key="roelofs2022mitigating"></d-cite> to urge against using equal width binning and to
suggest the use of an alternative: ECEsweep, which maximizes the number of equal-mass bins while ensuring 
the calibration function remains monotonic <d-cite key="roelofs2022mitigating"></d-cite>. 
The Adaptive Calibration Error (ACE) and Threshold Adaptive calibration Error (TACE) are two 
other variations of ECE that use flexible binning <d-cite key="nixon2019measuring"></d-cite>.
However, some find it sensitive to the choice of bins and thresholds, leading to inconsistencies in 
ranking different models <d-cite key="ashukha2020pitfalls"></d-cite>. 
Two other approaches aim to eliminate binning altogether: MacroCE does this by averaging over 
instance-level calibration errors of correct and wrong predictions <d-cite key="si2022re"></d-cite> and 
the KDE-based ECE does so by replacing the bins with non-parametric density estimators, 
specifically kernel density estimation (KDE) <d-cite key="zhang2020mix"></d-cite>.

### Only maximum probabilities considered
Another frequently mentioned drawback of ECE is that it only considers the maximum estimated probabilities <d-cite key="nixon2019measuring, ashukha2020pitfalls, vaicenavicius2019evaluating, widmann2019calibration, kull2019beyond"></d-cite>. The idea that more than just the maximum confidence should be calibrated, is best illustrated with a simple example <d-cite key="vaicenavicius2019evaluating"></d-cite>:


<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f11_max_probs_only_updated.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 7 | input example sourced from <d-cite key="schwirten2024ambiguous"></d-cite>
</div>

Let's say we trained two different models and now both need to determine if the same input image contains a *person*, an *animal* or *no creature*. The two models output vectors with slightly different estimated probabilities, but both have the same maximum confidence for "*no creature*". Since ECE only looks at these top values it would consider these two outputs to be the same. Yet, when we think of real-world applications we might want our self-driving car to act differently in one situation over the other <d-cite key="vaicenavicius2019evaluating"></d-cite>. This restriction to the maximum confidence prompted various authors <d-cite key="vaicenavicius2019evaluating, kull2019beyond, widmann2019calibration"></d-cite> to reconsider the definition of calibration. 
The existing concept of calibration as "confidence calibration" (coined in <d-cite key="kull2019beyond"></d-cite>) makes a distinction between two additional interpretations of confidence: __multi-class__ and __class-wise calibration__.

<br />

#### Multi-class Calibration
A model is considered multi-class calibrated if, for any prediction vector $$q=(q_1,...,q_K) \in \Delta^K $$​, the class proportions among 
all values of $$X$$ for which a model outputs the same prediction $$\hat{p}(X)=q$$ match the values in the prediction vector $$q$$.

$$  \mathbb{P} (Y  = k \; | \; \hat{p}(X)=q  ) = q_k \;\;\;\:\:\:\:  \forall k \in \{1,...,K\}, \; \forall q \in \Delta^K $$

<div class="caption">
where $(X,Y)$ is a datapoint and $\hat{p} : \mathscr{X} \rightarrow \Delta^K$ returns a probability vector as its output
</div>

What does this mean in simple terms? Instead of $$c$$ we now calibrate against a vector $$q$$, with K classes. Let's look at an example below:

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f12_multi-class.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 8 | Multi-class Calibration
</div>

On the left we have the space of all possible prediction vectors. Let's zoom into one such vector that our  model predicted and say the model has 10 instances  for which it predicted the vector $$q$$=[0.1,0.2,0.7]. Now in order for it to be multi-class calibrated, the  distribution of the true (*actual*) class needs to match the prediction vector $$q$$. The image above shows a calibrated example with [0.1,0.2,0.7] and a not calibrated case with [0.1,0.5,0.4].

<br />

#### Class-wise Calibration
A model is considered class-wise calibrated if, 
for each class k, all inputs that share an estimated probability $$\hat{p}_k(X)$$
<!-- for each class probability each estimated probability aligns  -->
align with the true frequency of class k when considered on its own:

$$ \mathbb{P} (Y  = k \; | \; \hat{p}_k(X)= q_k  ) = q_k \;\;\;\;\;\;  \forall k \in \{1,...,K\}$$

<div class="caption">
where $(X,Y)$ is a datapoint; $q \in \Delta^K $ and $\hat{p} : \mathscr{X} \rightarrow \Delta^K$ returns a probability vector as its output
</div>

Class-wise calibration is a __*weaker*__ definition than __multi-class__ calibration as it considers each class probability in __*isolation*__ rather than needing the full vector to align. The image below illustrates this by zooming into a probability estimate for class 1 specifically: $$q_1=0.1$$. Yet again, we assume we have 10 instances for which the model predicted a probability estimate of 0.1 for class 1. We then look at the true class frequency amongst all classes with $$q_1=0.1$$. If the empirical frequency matches $$q_1$$ it is calibrated.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f13_class-wise.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 9 | Class-wise Calibration
</div>

To evaluate such different notions of calibration, 
some updates are made to ECE to calculate a class-wise error. One idea is to calculate the ECE for each class and then take the average <d-cite key="nixon2019measuring, kull2019beyond"></d-cite>. Another idea is to swap the L1-distance used in ECE with the L2 and use several ECE metrics to more effectively assess the overall level of calibration <d-cite key="famiglini2023towards"></d-cite>. 
Others, introduce the use of the KS-test for class-wise calibration <d-cite key="gupta2020calibration"></d-cite> and <d-cite key="vaicenavicius2019evaluating"></d-cite> also 
suggest using statistical hypothesis tests instead of ECE based approaches.
<!-- by separately binning predictions for each class probability and then calculating the error and averaging across bins -->
And other researchers develop a hypothesis test framework \[TCal\] to detect whether a model is significantly mis-calibrated <d-cite key="donghwan2023tcal"></d-cite> and <d-cite key="sun2024confidenceintervalell2expected"></d-cite> build on this by developing confidence intervals for the L2-ECE.

All the approaches mentioned above __share a key assumption: ground-truth labels are available__. Within this gold-standard mindset a prediction is either true or false. However, annotators  might unresolvably and justifiably disagree on the real label <d-cite key="aroyo2015truth, uma2021learning"></d-cite>. Let's look at a simple example below:

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f14_one_hot.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 10 | One-Hot-Vector
</div>

We have the same image as in our entry example and can see that the chosen label differs between annotators. A common approach to resolving such issues in the labelling process is to use some form of aggregation <d-cite key="paun2022statistical, artstein2008inter"></d-cite>. Let's say that in our example the majority vote is selected, so we end up evaluating how well our model is calibrated against such  'ground truth'. One might think, the image is small and pixelated; of course humans will not be certain about their choice. However,  rather than being an exception such disagreements are widespread <d-cite key="aroyo2024dices, sanders2022ambiguous, schwirten2024ambiguous"></d-cite>. So, when there is a lot of human disagreement in a dataset it might not be a good idea to calibrate against an aggregated 'gold' label <d-cite key="baan2022stop"></d-cite>. Instead of gold labels more and more researchers are using soft or smooth labels which are more representative of the human uncertainty <d-cite key="peterson2019cifar, sanders2022ambiguous, collins2023human"></d-cite>, see example below.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f15_soft_label.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 11 | Soft-Label
</div>

In the same example as above, instead of aggregating the annotator votes we could simply use their frequencies to create a distribution $$P_{vote}$$ over the labels instead, which is then our new $$y_i$$. This shift towards training models on collective annotator views, rather than relying on a single source-of-truth motivates another definition of calibration: calibrating the model against human uncertainty <d-cite key="baan2022stop"></d-cite>.

<br />

#### Human Uncertainty Calibration
A model is considered human-uncertainty calibrated if, for each specific sample $$x$$, the predicted probability for each class k matches the '*actual*' probability $$P_{vote}$$ of that class being correct.

$$ \mathbb{P}_{vote} (Y  = k \; | \; X = x ) = \hat{p}_k(x) \;\;\;\;\; \forall k \in \{1,...,K\}$$

<div class="caption">
where $(X,Y)$ is a datapoint and $\hat{p} : \mathscr{X} \rightarrow \Delta^K$ returns a probability vector as its output
</div>

This interpretation of calibration aligns the model's prediction with human uncertainty, which means each prediction made by the model is individually reliable and matches human-level uncertainty for that instance. Let's have a look at an example below:

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f16_human_uncertainty.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 12 | Human Uncertainty Calibration
</div>

We have our sample data (*left*) and zoom into a single sample $$x$$ with index $$i=1$$. 
The model's predicted probability vector for this sample is [0.1,0.2,0.7]. 
If the human labelled distribution $$y_i$$ matches this predicted vector then this sample is considered calibrated.


This definition of calibration is more granular and strict than the previous ones as it applies 
directly at the level of individual predictions rather than being averaged or assessed over a set 
of samples. It also relies heavily on having  an accurate estimate of the human judgement distribution, 
which requires a large number of annotations per item. Datasets with such properties of annotations are 
gradually becoming more available <d-cite key="aroyo2024dices, nie2020learn"></d-cite>.

To evaluate human uncertainty calibration three new measures are introduced in <d-cite key="baan2022stop"></d-cite> : __the Human Entropy Calibration Error *(EntCE)*, the Human Ranking Calibration Score *(RankCS)* and the Human Distribution Calibration Error *(DistCE)*__.

$$EntCE(x_i)= H(y_i) - H(\hat{p}_i), $$

where $$H(.)$$ signifies entropy.

__EntCE__ aims to capture the agreement between the model's uncertainty $$H(\hat{p}_i)$$ and the human uncertainty $$H(y_i)$$
for a sample $$i$$. However, entropy is invariant to the permutations of the probability values; in other words it doesn't change when you rearrange the probability values. This is visualised in the image below:

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f17_entce.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 13 | EntCE drawbacks
</div>

On the left, we can see the human label distribution $$y_i$$
<!-- $$\mathbb{P}_{vote}$$ -->
, on the right are two different model predictions for that same sample. All three distributions would have the same entropy, so comparing them would result in 0 EntCE. While this is not ideal for comparing distributions, entropy is still helpful in assessing the noise level of label distributions.

$$RankCS = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1} (argsort(y_i) = argsort(\hat{p}_i)), $$   
<!-- $$RankCS = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1} (argsort(\hat{p}_i) = argsort(\mathbb{P}_{vote\; i}) ), $$ -->
<!-- LOOK AT ORIGINAL FORMULA FROM PAPER! -->

where argsort simply returns the indices that would sort an array.

So, __RankCS__  checks if the sorted order of estimated probabilities $$\hat{p}_i$$ matches the sorted order of $$y_i$$ for each sample. 
If they match for a particular sample $$i$$ one can count it as 1; if not, it can be counted as 0, which is then used to average over all samples N. <d-footnote>In the paper it is stated more generally: If the argsorts match, it means the ranking is aligned, contributing to the overall RankCS score.</d-footnote>

Since this approach uses ranking it doesn't care about the actual size of the probability values. The two predictions below, while not the same in class probabilies would have the same ranking. This is helpful in assessing the overall ranking capability of models and looks beyond just the maximum confidence. At the same time though, it doesn't fully capture human uncertainty calibration as it ignores the actual probability values.

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/f18_rankcs.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 14 | RankCS drawbacks
</div>



$$ DistCE(x_i) = \mathbf{TVD}(y_i, \hat{p}_i)$$

__DistCE__ has been proposed as an additional evaluation for this notion of calibration. It simply uses the total variation distance $$(TVD)$$ between the two distributions, which aims to reflect how much they diverge from one another. _DistCE_ and _EntCE_  capture instance level information. So to get a feeling for the full dataset one can simply take the average expected value over the absolute value of each measure: $$E[\mid DistCE \mid]$$ and $$E[\mid EntCE \mid]$$. 
Perhaps future efforts will introduce further measures that combine the benefits of ranking and noise estimation for this notion of calibration.


______


<!-- ### Visual Overview of all Definitions -->

<div class="row mt-2">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-calibration/calibration_overview.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Image 15 | Visual Summary: Definitions at a Glance
</div>

## Final Thoughts

We have run through the most common definition of calibration, the shortcomings of ECE and additional notions of calibration: multi-class, class-wise & human-uncertainty calibration. 
We also touched on some of the newly proposed evaluation measures and their shortcomings.
Despite several works arguing against the use of ECE for evaluating calibration, it remains widely used. 
The aim of this blogpost is to draw attention to these works and their alternative approaches. 
Determining which notion of calibration best fits a specific context and how to evaluate it should avoid misleading results. 
Maybe, however, ECE is simply so easy, intuitive and just good enough for most applications that it is here to stay?
