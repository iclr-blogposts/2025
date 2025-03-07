---
layout: distill
title: Do vision models perceive objects like toddlers ?
description: Despite recent advances in artificial vision systems, humans are still more 
  data-efficient at learning strong visual representations. Psychophysical experiments 
  suggest 
  that toddlers develop fundamental visual properties between the ages of one and 
  three, which affect their perceptual system for the rest of their life. 
  They begin to recognize impoverished variants of daily objects, pay more 
  attention to the shape of 
  an object to categorize it, prefer objects in specific orientations and progressively 
  generalize over the configural arrangement of objects' parts. This post examines 
  whether these four visual properties also emerge in off-the-shelf machine learning (ML) 
  vision models. We reproduce and complement previous studies by 
  comparing toddlers and a large set of diverse pre-trained vision models for each visual property. This way, we 
  unveil the interplay between these visual properties and highlight the main 
  differences between ML models and toddlers. Code is available at (https://github.
  com/Aubret/BabyML).
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Arthur Aubret
    url: "https://aubret.github.io/"
    affiliations:
      name: Frankfurt Institute for Advanced Studies
  - name: Jochen Triesch
    url: "https://www.fias.science/en/life-and-neurosciences/research-groups/jochen-triesch/"
    affiliations:
      name: Frankfurt Institute for Advanced Studies

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-toddlers-vs-vismodels.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: Motivation
      - name: Scope of the study
      - name: TL;DR
  - name: Experiments
    subsections:
      - name: Caricature recognition
      - name: Shape bias
      - name: Side-view bias
      - name: Configural sensitivity
  - name: Conclusion
  - name: Acknowledgments
  - name: Models

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

## Motivation
State-of-the-art machine learning vision models learn visual representation by 
learning from millions/billions of independently and identically distributed diverse 
images.
In comparison, toddlers almost always play with/observe the same objects in the same 
playground/home/daycare environments. 
They likely have not seen 10% of the different platypuses or sea snakes on ImageNet. 
In sum, toddlers likely experience a 
diversity of objects which is several orders of magnitude lower than current models. 
Unlike current machine learning (ML) vision models, they also do not have access to a 
massive amount of category
labels (or aligned language utterances) or adversarial samples. Despite using 
different learning mechanisms, 
they develop strong semantic representations that are robust to image distortions, viewpoints, machine-adversarial
samples and different styles (drawings, silhouettes...) <d-cite key="wichmann2023deep, 
huber2023developmental"></d-cite>.

What perceptual mechanisms underpin such a robust visual system ? Psychophysics 
experiments demonstrate that toddlers progressively reach fundamental visual 
milestones and develop
visual biases during their second and third year. Specifically, toddlers become able 
to extract the category of simplified objects <d-cite key="smith2003learning"></d-cite> and learn to 
attend to the shapes of novel objects when judging their 
similarity based on their *kind* <d-cite key="diesendruck2003specific"></d-cite>. Simultaneously, 
they also manipulate object to clearly exhibit their main axis of elongation <d-cite 
key="pereira2014main"></d-cite>, 
and later, begin to semantically identify objects based on
the configural arrangement between their parts <d-cite key="augustine2011parts"></d-cite>.


In this blog post, we investigate the extent to which off-the-shelf ML models also 
exhibit 
these properties.
Comparing 
vision models specifically to toddlers (not adults <d-cite key="bowers2023deep,
wichmann2023deep"></d-cite>) allows us to identify which 
developmental milestones are not (yet) achieved by current vision models. We extend previous studies to give a global 
picture of the potential emergence of toddler-like 
visual properties in current pre-trained ML models. We include diverse models based on 
supervised learning, semi-weak supervision, robust 
out-of-distribution image 
recognition, self-supervised learning with instance discrimination 
(ID) and masked auto-encoders (MAE), adversarial training, 
vision-language models and egocentric visual training. We also try both convolutional architectures (CNN) and vision transformers (ViT). We 
refer to the end of the blog post for a complete list 
of models and why we chose them. Our objective is to 1) complement and extend prior 
experiments 
comparing ML to humans to (in)validate prior claims; 2) understand 
the (dis)similarities between the visual properties of ML models and toddlers; 3) clarify the interplay between
different toddler-based visual properties in ML. We will also publicly release our 
code on github upon acceptance as an easy-to-use toolbox for assessing new models.


## Scope of the study

We divide our study into four parts, each corresponding to a specific visual
property emerging in toddlers and remaining for the rest of their life. In the following,
we describe each and clarify the scope of
our experiments with respect to prior work. Note that we do not aim to faithfully 
reproduce evaluation protocols from developmental studies, as they are often based on 
language and we do not want to limit our experiments to vision-language models. 
Instead, we use 
alternatives that aim to assess the presence of a visual property relative to toddlers.
Note that we still include vision-language models.


{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/overview.png"
class="img-fluid" %}
<div class="caption">
Figure 1. Illustration of emerging visual properties in toddlers. 
A: Caricature recognition. 
B: Shape bias.
C: Preferred views. 
D: Sensitivity to spatial arrangement of features. 
Green rectangles in A, B indicate what toddlers have learned to judge as more similar. 
Blue rectangle in C indicates preference for side views. 
Red rectangle in D means that humans learn to consider the two stimuli with rearranged features as dissimilar.
</div>

#### Caricature recognition

At around 18 months, toddlers become able to recognize
impoverished variants of daily-life objects called caricatures. These caricatures are
constructed from few simple shapes <d-cite key="smith2003learning"></d-cite> (e.g. 
Figure 1, A. The
recognition of these caricatures is related to the development of abstract 
representations and may boost category generalization <d-cite 
key="son2008simplicity"></d-cite>. To the best of our
knowledge, the only study investigating ML models for caricature recognition focuses on
vision-language models <d-cite key="sheybani2024modelvsbaby"></d-cite>. We modify
their experimental protocol to compare a wider range of pre-trained models.

#### Shape bias

When asked to match objects based on their *kind* (but also names etc.), 
toddlers group together novel objects that share the same shape, rather than objects that 
share the same color 
or texture <d-cite key="diesendruck2003specific"></d-cite> (cf. Figure 1, B). This 
bias emerges slightly after learning caricature recognition <d-cite 
key="yee2012changes"></d-cite>. This effect has been widely investigated in ML models 
following the introduction of related datasets <d-cite 
key="geirhos2021partial, hermann2020origins"></d-cite>. However, mainstream evaluation
protocols differ from evaluation protocols in the developmental literature <d-cite
key="tartaglini2022developmentally"></d-cite>, which are known to be very sensitive to
experimental details <d-cite key="frigo1998adjective"></d-cite>. Here, we extend
the developmentally-relevant experiments of Tartaglini et al. (2022) to include more
models, as they only evaluated 5 pre-trained models.

#### Side-view bias

From 18-24 months onward, toddlers learn to manipulate objects
to see them 1- in upright positions; 2- with their main axis of 
elongation perpendicular to the line of sight, which we call side view (e.g. Figure 1,
C) <d-cite key="pereira2010early, pereira2014main"></d-cite>. Here, we focus on this 
side-view bias, which is a perceptual bias (not reflecting motor or grasping biases)<d-cite key="james2014some"></d-cite>. A more fine-grained 
look at toddlers' behavior shows that they visually inspect these views while 
keeping the object almost static <d-cite 
key="pereira2014main,lisboa2014looking"></d-cite>. This suggests that these 
side-views are somehow perceptually *special*. However, the origins of this bias remain 
unclear. To the best of our knowledge, we are the first to investigate if these views are also 
special for ML models.

#### Configural relation between parts 

Starting at around 20 months, infants take into account the relative position of 
parts of an object to
categorize it <d-cite key="augustine2011parts"></d-cite>. Such ability becomes fully
mature for novel objects only later in development <d-cite 
key="mash2006multidimensional"></d-cite> and
is a fundamental component of one of the main theories of human object 
recognition<d-cite key="hummel1992dynamic"></d-cite>. To the best of our knowledge, two lines of study have 
investigated this ability in ML models <d-cite key="baker2022deep,farahat2023novel"></d-cite> with two opposite conclusions. 
We discuss their differences and assumptions, extending the experiments of Baker et al. (2022) (Figure 1, D) to include more models.

## TL;DR

Our analysis reveals two main visual milestones in machine learning models. The first 
one is the acquisition of the shape bias, which correlates with the ability to 
recognize normal objects and caricatures. This bias is close to that of 
toddlers in the strongest models. We also find that the shape bias is clearly 
connected to side views being better prototypes of object instances, *i.e.* they are 
relatively more similar to other views of an object. However, side-views are not prototypical 
enough to explain a side-view bias at the level of toddlers. The second one is the development of 
configural sensitivity, for which adults (and probably toddlers) level is out-of-reach 
for all investigated models. Neither of the two milestones are achieved by investigated 
models 
that were trained on a reasonable amount of visual egocentric data. In sum, 
comparing properties of toddlers object perception with those of artificial vision 
systems suggests that there is room for ML improvements in 1) making the development 
of the shape bias more data-efficient and 2) learning to generalize based on the configural arrangement of parts.
For 1), one should not overlook CNN architectures, as we found them to show a more 
robust shape bias on novel shapes, compared to ViTs.

# Experiments

## Caricature recognition

The main developmental protocol to evaluate caricature recognition is to ask a subject 
"where is the {object}" among a set of one object caricature matching the 
word and other
non-matching object caricatures <d-cite key="smith2003learning"></d-cite>. 
This has been recently adapted to vision-language models by giving the object name to the
language encoder and retrieving the closest image embeddings in a set of images <d-cite
key="sheybani2024modelvsbaby"></d-cite>. 

Here, we modify their evaluation protocol to include models beyond vision-language 
models.
We take the BabyVsModels dataset <d-cite key="sheybani2024modelvsbaby"></d-cite>, which 
contains 8 categories, each with 5 and 8 
different images for realistic and caricatures, respectively. We create the set of all 
possible triplets, each constructed with a realistic image of
an object, one caricature from the same category as the realistic image and one 
caricature from a different category. We compute their image representations and test whether the cosine
similarity between the representation of the realistic image and its 
category-matching caricature is higher than the cosine similarity between the 
realistic image representation and the non-matching caricature. We define the 
recognition accuracy as the success rate across all triplets. We replicate the
process with a harder 8-choices (versus two-choices) classification variant to check 
whether the difficulty of the task impacts our results. In this case, we use 100 tuples 
per realistic 
image. Furthermore, to estimate the recognition performance of models on normal images,
we also compute the success rate on realistic images.

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/caricaturesimg.png"
class="img-fluid" %}
<div class="caption">
Figure 2. Illustration of how we compute A) simple caricature recognition accuracy 
and B) the normal image recognition accuracy. d is the negative cosine similarity.
</div>


In Figure 3, A, we find that diverse models achieve a strong caricature recognition 
performance, at the estimated level of toddlers. Results confirm that supervised 
and vision-language models perform well on these tasks <d-cite key="sheybani2024modelvsbaby"></d-cite>. We notably find 
that ID models (DinoV2, BYOL, AA-SIMCLR) perform reasonably well on caricature 
recognition. Figure 3, B indicates that the hard caricature recognition accuracy 
correlates with hard image 
recognition accuracy 
(Pearson score: $0.78$). Overall, we conclude that strong caricature recognition may 
 be a side-effect of strong category recognition and that cutting-edge models 
already reach the level of toddlers, independent of their exact training procedure.


{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/caricatures.png"
class="img-fluid" %}
<div class="caption">
Figure 3. A) Simple caricature and B) hard caricature accuracy against normal image
accuracy recognition. We estimate 24-months toddler performance based on data from 
Pereira et al. (2009). This is consistent with results from Sheybani et al. (2024),  
which suggests that they perform similarly to pre-trained CLIP models. We refer to the 
"Models" section for more information about the groups of models.
</div>

## Shape bias

It seems obvious that the best investigated models leverage objects' shapes to
measure image similarities, as texture and color cues are absent from caricatures. 
Here, we examine the shape bias of current models, compared to toddlers, and analyze 
the relationship between shape bias and caricature recognition.

The cue-conflict protocol is the main experimental protocol to evaluate the shape bias 
in toddlers <d-cite key="diesendruck2003specific"></d-cite>. 
The idea is to present an object and two variants, one modifying only the texture of the 
original object and one modifying only the shape of the original object. The task is 
then to choose which of the two variants is the closest to the original one. If a 
toddler tends to select the same-shape variant, it means that they are shape biased. 

ML benchmarks used to evaluate the shape bias of models <d-cite key="
geirhos2021partial,hermann2020origins"></d-cite> do not follow the experimental protocols
of developmental studies, as they texture the whole image instead of the object itself<d-cite key="smith2003learning"></d-cite>. This is important as it confounds using the 
shape and being able to segment the object. Thus we follow the cue-conflict 
experimental protocol of Tartaglini et al. (2022), who only texture the objects. First,
we evaluate the shape bias on common (in-distribution) objects. We reuse the masks of common objects and 16 
textures used in prior works <d-cite key="hermann2020origins,tartaglini2022developmentally"></d-cite>; this set includes 16
categories of objects, each containing 10 masks of different object instances. With 
these images, we build a set of triplets that follow the cue-conflict paradigm. 
In each triplet, we construct an anchor image mixing a shape with a texture, a variant 
built with the same shape but a different texture
and another variant built with the same texture but a different shape. For each
triplet, we define a shape bias success as the representation of the anchor being closer 
to the same shape variant than to the same texture variant. The shape bias 
accuracy is simply the success rate. The set of triplets includes all possible 
combinations.
Second, to evaluate the shape bias on novel objects, we reproduce the same protocol 
using the 16 novel shapes from prior work <d-cite key="tartaglini2022developmentally,
parks2020statistical"></d-cite>.

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/shapebiasimg.png"
class="img-fluid" %}
<div class="caption">
Figure 4. Illustration of how we compute the degree of shape bias on A) common objects 
and B) novel shapes. $d$ is the negative cosine similarity.
</div>

In Figure 5, A, we verify that there is a correlation between hard caricature
recognition and the degree of shape bias over common objects (Pearson
correlation: $0.68$). This is higher than the correlation with hard object recognition 
($0.
56$). We also find that language is not a mandatory component of the shape bias as ID 
models seem to classify object based on shapes. In particular, DinoV2 shows a 
similar degree of shape bias as vision-language or supervised models. This is interesting because developmental studies clearly found 
that the productive vocabulary size is a strong predictor of the shape bias <d-cite 
key="gershkoff2004shape,jones2003late"></d-cite>. Our results suggest that one should not 
over-estimate the causal importance of
language for the acquisition of this bias <d-cite key="
gavrikov2024vision,smith2002object"></d-cite>. 


In Tartaglini et al. (2022), the gap of shape bias between toddlers and models increases 
for novel shapes. When reproducing their results, we noticed that the performance of a 
random ResNet50 (Random_RN50, averaged over three random initializations) decreases 
between common and novel shapes. Since all shapes are novel for a random model, this suggests that the 
used novel shapes are inherently less prone to induce a shape-based matching than the 
common objects. To naively correct for this effect, we multiply all scores by the ratio 
$$\frac{\texttt{ShapeBiasCommon}
(\texttt{Random_RN50})}{\texttt{ShapeBiasNovel}(\texttt{Random_RN50})}=1.334$$
and show the results in Figure 5, B. We find that the strongest models perform close 
to toddlers' performance; the best models, Noisy-Student and SWSL, match 
and outperforms toddlers performance, respectively. A thorough 
look at Figure 5, B seems to indicate that CNNs
better generalize the shape bias to novel objects than ViTs (e.g. CLIP_RN50 versus 
CLIP_ViT-L/14), despite two outliers (Sup_RN50 and CLIP-LAION_CNX_XXL/32). This nuances
previous claims of ViTs being more shape biased than CNNs<d-cite 
key="tuli2021convolutional,naseer2021intriguing"></d-cite>. Overall, the best ML models have a relatively similar level of shape bias 
to toddlers.

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/shapebias.png"
class="img-fluid" %}
<div class="caption">
Figure 5. Shape bias for A) common objects and B) novel objects. We extract
3-years old toddler performance from Diesendruck et al. (2003) <d-cite 
key="diesendruck2003specific"></d-cite> and
assume the performance would be similar with our set of masks and textures.
</div>

## Side views

The previous section indicates that ML models reach a high degree of shape bias with
common objects. Simultaneously to its emergence, toddlers also focus their attention on
side views of objects <d-cite key="pereira2010early,pereira2014main"></d-cite>, defined as
views showing the main axis of elongation perpendicular to the line of sight (when the object is in a canonical upright position).
To the best of our knowledge, the reasons for such a
bias are currently unclear. A hypothesis may be that side views are particularly
interesting for toddlers because they unambiguously display the whole shape, 
simplifying object recognition.

To investigate this question, we leverage the OmniObject3D dataset <d-cite key="
wu2023omniobject3d"></d-cite> which contains 6,000 high-quality textured meshes scanned
from real-world objects, distributed into 190 categories. We specifically use the provided
recordings of 24 in-depth rotations per object in a canonical upright position. We 
preprocess this dataset by removing objects that do not show a clear main axis of 
elongation (e.g. a rotationally symmetric bottle). To do
so, we remove objects for which the ratio between the narrowest and widest views on the
horizontal axis is below $0.8$. The provided views were randomly sampled around the yaw
axis, meaning that we do not access the exact side views of an object. Thus, we define 
the main side, main front, main 3/4 views of an object as the object's views that 
have the closest angular rotation, with respect to a 
canonical side view, to $\\{0, 180\\}$, $\\{-90, 90\\}$, $\\{-135, -45, 45, 135\\}$ 
degrees, respectively. 

To assess the specificity of each of these main views, we propose two metrics: 
- the intra-object distance as the average 
distance between the representation of a given main view of an object and the 
  representations of all 
  other views of the same object;
- the intra-category distance as the average distance between the representation of the 
  given main view of an object and the representations of all other views in the 
  same category.

For each object, we retrieve the main view that maximizes and minimizes each
metric. Finally, we compute the side-view accuracy as the average number of times the main
side view maximizes these metrics. We similarly compute the 3/4-view and front-view accuracy.
This allows us to assess which main orientation is more prototypical for an object or a 
category. Figure 6 illustrates the procedure.
  

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/viewsimg.png"
class="img-fluid" %}
<div class="caption">
Figure 6. Illustration of how we compute the side-view accuracy for intra-object 
distance. $d$ refers to the negative cosine similarity.
</div>

First, we compute the Pearson correlation between the intra-object side-view 
accuracy and our previously reported measures. We find that the metric that most highly 
correlates with 
the side-view accuracy is the shape bias on common objects (Pearson correlation: $0.86$). 
The correlation is similar for intra-category side-view accuracy (Pearson correlation: $0.
80$).
Visualizing the results in Figure 7 confirms that the higher the shape bias, the 
more prototypical the side views are. The fact that the shape bias correlates with 
side views being more prototypical aligns with the fact that the shape bias emerges 
during the same period as the side-view bias in toddlers. This correlation supports the 
hypothesis that toddlers may turn objects towards side views because they are more 
prototypical.

Interestingly, AA-SimCLR <d-cite 
key="aubret2024self"></d-cite> notably presents a boost in side-view 
accuracy for a given level of shape bias. This model leverages egocentric actions that 
rotate objects during training and has been shown to better generalize visual representations in a 
viewpoint-wise fashion. This suggests that side-view accuracy can be improved in the best-performing models through such an approach. 

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/views.png"
class="img-fluid" %}
<div class="caption">
Figure 7. View accuracies for intra-object and intra-category similarity maximization.
</div>

We did not find raw data on the time spent by toddlers on side views. 
However, current studies found (on a different set of objects) that toddlers focus 
more on these views than on 3/4 views <d-cite key="pereira2010early"></d-cite>. In our 
case, 
the 3/4 view accuracy remains beyond the reach for all models. Thus, if toddlers indeed 
target views because they are more prototypical, ML models show a far lower bias than toddlers. 

## Configural relation between parts

Paying attention to the shape of an object is different from looking at the relative
position of all parts. It could be that the model only extracts local <d-cite key="baker2020local"></d-cite> or 
intermediate <d-cite key="jarvers2023shape"></d-cite> shape features for
recognizing categories. For instance, looking at the shape of the small ears of a 
bunny is often enough to recognize it. This may not be enough to 
unveil the potential of side views as these views also clearly display the 
configural arrangement of parts of an object. For instance, side-views exhibit that
the head of an animal is usually located at the opposite side as its tail.

The ability to categorize objects based on the configural arrangement of their parts 
starts its development in toddlers <d-cite key="augustine2011parts"></d-cite> and 
becomes mature only for older children <d-cite key="mash2006multidimensional"></d-cite>. Two lines of prior works studied whether ML models rely on the configural relation between parts to compare
images <d-cite key="baker2022deep,farahat2023novel"></d-cite>. For both, the idea is the
same: they try to modify the positions of parts of an image (e.g. a head, a tail, a wheel)
without deteriorating the structure of these parts.

In <d-cite key="farahat2023novel"></d-cite>, they train a CNN (with 
supervision) with a specific
architecture that outputs representations corresponding to small parts of an image; then 
they spatially scramble these local representations and train (with supervision) another 
neural network on top of it.
Their assumption is that the first model learns representations of local and coherent 
features: in that case, the scrambling process will mimic a change in relative feature 
positions. They found that the scrambling process hurts the
recognition performance and conclude that current CNNs pay attention to the configural
relation between parts. To the best of our knowledge, the validity of their assumption 
 is unclear. In addition, their results could be
specific to their CNN and the two-steps training process.

In <d-cite key="baker2022deep,baker2018deep"></d-cite>, they propose to create 
*Frankenstein* silhouettes by taking an object silhouette and horizontally flipping 
the upper part of the silhouette, while keeping aligned the boundaries of the silhouette.
The underlying assumption is that the flip operation does not alter the parts themselves. 
They found that the recognition performance of ML models is largely unaffected by the flipping process,
unlike adults. We notice that they mostly use animals in their study, such that the 
lower and upper part of the silhouette contain distinct parts: the bottom part 
corresponds to the legs and the top part often displays the shape of the head (ears etc.
..).

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/frankenstein.png"
class="img-fluid" %}

Thus, we take the set of stimuli from Baker et al. (2022), which
comprises 9 categories of animals, each containing 40 original and Frankenstein 
silhouettes. To reproduce experiments from Baker et al. (2022) without a classifier, we sample a
normal silhouette from each category and one Frankenstein silhouette. We compute the
cosine similarity between the Frankenstein silhouette and all normal silhouettes and 
define the accuracy as how often the cosine similarity between the Frankenstein and the 
category-matching normal silhouette is the largest (Frankenstein test). To assess the 
relative
performance, we apply the procedure again after replacing the Frankenstein silhouette by
another normal silhouette (normal test). Finally, we compute the 
configural sensitivity as the difference in performance between the success 
rate of the Frankenstein test and the success 
rate of the normal test. To show sensitivity to the spatial arrangement of parts, the metric must be lower than $0$.

{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/configuralimg.png"
class="img-fluid" %}
<div class="caption">
Figure 9. Illustration of the Frankenstein test.
</div>

We first compute the Pearson correlation score between configural sensitivity and
hard image recognition ($-0.58$), hard caricature recognition ($-0.71$) shape bias on 
common objects ($-0.69$) and shape bias on novel objects ($-0.41$). Given that hard 
caricature recognition is the most highly correlated metric,
we plot configural sensitivity against hard caricature recognition in Figure 10. We 
observe that the performance of current models is very weak, barely lower than $0$. Only 
large-scale weakly supervised models are significantly more sensitive than a BagNet-9, a 
model that cannot show configural sensitivity by design. All models are less sensitive than 
the estimated adults. We conclude that even shape-biased models likely rely on local 
shape cues instead of the global arrangement of parts.


{% include figure.html path="assets/img/2025-04-28-toddlers-vs-vismodels/configural.png"
class="img-fluid" %}
<div class="caption">
Figure 10. Configural sensitivity of ML models against their shape bias on common objects.
</div>

# Conclusion

In this blog post, we have extended previous studies to investigate whether current ML 
models 
learn 4 fundamental visual properties that emerge in toddlers. While we did not observe 
major effects of the 
training setup (supervised, adversarial, self-supervised...), we 
found that cutting-edge 
models reach the estimated level of a toddler with respect to caricature recognition and the shape bias on common objects. For the shape bias on novel objects, 
the best models are close to toddlers. However, most of 
the considered models use bio-implausible data: a lot more diverse images, labels, millions of aligned language 
utterances, adversarial samples, etc... Models trained 
with egocentric visual images comparable to humans (VC-1, VIP, R3M) 
perform poorly on all benchmarks. In addition, other ML 
models still perform poorly on configural sensitivity and the proposed side-view bias 
compared to toddlers and adults, suggesting there is room for improvements. We cannot 
conclude on the performance of toddlers for configural sensitivity for the considered 
task. However, given the emerging configural
sensitivity of toddlers <d-cite key="augustine2011parts"></d-cite> and the amplitude of the 
difference between models and adults, we presume that ML models are 
likely inferior to toddlers.

We found that object recognition abilities positively correlate with caricature 
recognition, which 
also positively correlates with a strong shape bias. This suggests that these properties 
are connected in ML models. However, it does not mean that increasing the shape bias 
systematically leads to better recognition abilities, as evidenced by the Noisy-Student 
model (high shape bias, relatively low caricature recognition) and prior works <d-cite 
key="hermann2020origins"></d-cite>. Furthermore, showing a shape bias correlates with the 
side-view 
bias and the configural sensitivity, but it is far from enough 
to reach the level of humans. Thus, we speculate that the shape bias may be a first 
milestone of visual learning, on the way to generalizing objects based on the relative 
arrangement of parts.

From a developmental perspective, prior work found that productive vocabulary size is a 
strong predictor of the shape bias in toddlers <d-cite key="gershkoff2004shape,
jones2003late"></d-cite>. It could be that language aids in building visual 
representations or that strong visual representations are needed to produce language.
Although we cannot conclude about a potential causal direction of this relation, we 
discovered that a shape bias similar to toddlers' *can* emerge without language (DinoV2). Thus, 
one should not over-estimate the causal role of language in building visual representations. 

The origin and consequence of the side-view bias for toddlers is mostly unclear in the 
developmental literature. It may emerge because these views are very stable for small 
rotations <d-cite key="perrett1992use"></d-cite> or because they are standardized views for aligning
views of objects <d-cite key="pereira2010early"></d-cite>. 
In both cases, preliminary experiments indicate this bias may 
boost their object recognition abilities <d-cite key="james2001manipulating"></d-cite>.
We investigated for the first time the potential specificity of side-views for ML 
models and found that these views become more object-prototypical when the shape bias 
increases. This supports the hypothesis that toddlers' bias emerges because these 
views are more prototypical. However, in this case, it remains unclear why side-views are 
still less object-prototypical than 3/4 views on average for ML models. It could be 
that ML models lack some visual property like configural sensitivity or that our dataset is too different 
from the set of 16 simple stimulus objects used in Pereira et al. (2010).

This study presents several limitations. First, the experimental protocol does not 
perfectly follow either developmental experiments (often based on language) or machine 
learning protocols (often based on vision-language models or supervised labels).
We chose not to rely on language to test a broad set of models beyond vision-language 
models, and we decided to not use supervised heads as it is unclear how it relates to 
toddlers' visual representations. Thus, we assume that, during their task, toddlers 
perform a form of cognitive comparison that resembles our comparison of 
representations. Second, the set of stimuli generally varies between developmental 
studies and ours, allowing to only approximate how a toddler would perform. This is 
especially salient in our study of configural sensitivity: to the best of our knowledge, 
toddlers have not been tested on Frankenstein silhouettes. Third, the set of stimuli 
is small and often biased (very small for caricatures, only animals for configural 
sensitivity, white background in almost all datasets...). Despite that, our study 
provides a 
global picture of the presence
and interplay of toddlers' visual properties in ML models. We hope it will spur 
research aimed at addressing the gap between models and toddlers.


# Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft 
(DFG project Abstract REpresentations in Neural Architectures (ARENA)), 
as well as the projects The Adaptive Mind and The Third Wave of 
Artificial Intelligence funded by the Excellence Program of the 
Hessian Ministry of Higher Education, Science, Research and Art (HMWK). 
JT was supported by the Johanna Quandt foundation. We gratefully acknowledge 
support from the Goethe-University (NHR Center NHR@SW) and Marburg University (MaRC3a) 
for providing computing and data-processing resources needed for this work.


# Models

To reproduce and strengthen the claims reported papers and this blogpost, we test a very
diverse set of pre-trained models, including those addressing specific shortcomings of ML
models. We use a mix of convolutional architectures (CNN) and vision transformers (ViT).

As most of previous experiments, we assess standard models based on random 
initializations and supervised ImageNet training for both ResNet50 and ViT-L. We also
consider two different adversarially trained supervised models <d-cite key="salman2020adversarially"></d-cite>, as previous works argued that they share similar
adversarial robustness properties as humans <d-cite 
key="guo2022adversarially"></d-cite>. Finally,
we add three supervised BagNet models that extract representations of small image 
patches in the deepest
layer (corresponding to $9\times9$, $17\times17$ and $33\times33$ pixels)<d-cite 
key="brendel2018approximating"></d-cite>; this allows to give hints on whether the local
structure is enough to fulfill a given task.


On the weakly supervised side, we consider a strong ViT-L/16 trained on a large-scale
dataset of hashtags (SWAG) <d-cite key="singh2022revisiting"></d-cite>, four 
vision-language
models based on the openai CLIP with ResNet50 and ViT-L/14<d-cite key="
radford2021learning"></d-cite> and two open CLIP models trained on LAION-5B <d-cite key="
schuhmann2022laionb,cherti2023reproducible,ilharco_gabriel_2021_5143773"></d-cite> 
with a ConvNet-XXLarge and a ViT-G/14; these vision-language models 
show
classification errors similar to humans <d-cite key="geirhos2021partial"></d-cite> and 
are very
robust at recognizing caricatures <d-cite key="sheybani2024modelvsbaby"></d-cite>. 

We further add two strong methods for out-of-distribution image 
classification <d-cite key="geirhos2021partial"></d-cite>, namely Noisy-Student <d-cite key="xie2020self"></d-cite> and SWSL <d-cite 
key="yalniz2019billion"></d-cite>. 

We include state-of-the-art self-supervised instance discrimination (ID) methods trained
on ImageNet-1K, namely MoCoV2 <d-cite key="chen2020improved"></d-cite>, BYOL <d-cite key="
grill2020bootstrap"></d-cite> and DinoV2 as a mainstream foundation model<d-cite key="
oquab2024dinov2"></d-cite>. We further include models trained on MVImgNet <d-cite key="
yu2023mvimgnet"></d-cite> to be less sensitive to viewpoints (SimCLR-TT and AA-SimCLR, a
slightly more viewpoint-sensitive variant) <d-cite key="aubret2024self"></d-cite>. We also
add a Masked Auto-Encoder (MAE) trained on ImageNet with a ViT-L/16 <d-cite key="
he2022masked"></d-cite>, as MAEs keep more information than ID methods about the spatial
location of features.

Finally, we evaluate three
vision models trained on robotic and egocentric data, namely VIP <d-cite key="mavip"></d-cite>,
VC-1<d-cite key="majumdar2023we"></d-cite> and R3M <d-cite key="nairr3m"></d-cite>. 
Such training data is supposed to be relatively close to the visual experience of 
toddlers.




