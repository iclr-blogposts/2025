---
layout: distill
title: Does Editing Provide Evidence for Localization?
description: A basic aspiration for interpretability research in large language models is to localize semantically meaningful behaviors to particular components within the LLM. There are various heuristics for finding candidate locations within the LLM. Once a candidate localization is found, it can be assessed by editing the internal representations at the corresponding localization and checking whether this induces model behavior that is consistent with the semantic interpretion of the localization. The question we address here is, how strong is the evidence provided by such edits? To assess localization, we want to assess the effect of the optimal intervention at a particular location. The key new technical tool is a way of adapting LLM alignment techniques to find such optimal localized edits. With this tool in hand, we give an example where the edit-based evidence for localization appears strong, but where localization clearly fails. Indeed, we find that optimal edits at random localizations can be as effective as aligning the full model. In aggregate, our results suggest that merely observing that localized edits induce targeted changes in behavior provides little to no evidence that these locations actually encode the target behavior.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Zihao Wang
    affiliations:
      url: "https://zihao12.github.io"
      name: Department of Statistics, University of Chicago
  - name: Victor Veitch
    affiliations:
      url: "https://www.victorveitch.com"
      name: Department of Statistics, University of Chicago


bibliography: 2025-04-28-localization.bib  

toc:
  - name: Introduction
  - name: Backgrounds and results from ITI
  - name: Editing Localized Heads Modifies the Output as Expected
  - name: Finding "optimal" interventions
  - name: Optimal interventions at localized heads are nearly optimal, but so are random heads
  - name: Intervening a single head is just as effective
  - name: Are the Probing-Localized Heads Anything Special?
  - name: Discussion
  - name: Experiment Details

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


## Introduction

<p>
A basic goal of interpretability research for large language models is to map semantically meaningful behavior to particular subcomponents of the model.
Semantically meaningful encompasses a wide range of things, e.g., "when asked for directions to the Eiffel tower, the model gives directions to Paris", "the model responds truthfully", or "the model will refuse to response". The aim is to find, e.g., neurons, circuits, or regions of representation space that control these behaviors. If we could find such localizations, we could use them as building blocks to understand complex model behaviors.
Many interpretability approaches can be understood in terms of the following idealized template <d-cite key="zou2023representation,arditi2024refusal,wang2023backdoor,chen2024truth,wei2024assessing,li2024inference,meng2022locating,vig2020causal,geiger2021causal,soulos2019discovering,finlayson2021causal,wang2022interpretability,chan2022causal,hanna2024does,conmy2023towards,todd2023function,hendel2023context"></d-cite>:
</p>

<p>
1. We use some heuristic to find a candidate location in the model that is conjectured to be responsible for a particular behavior.
</p>

<p>
2. We then run the model with some set of inputs, and collect the model's internal representations for each input.
</p>

<p>
3. Then, we edit each of these representations at the candidate location, and generate new outputs according to the edited representations.
</p>

<p>
4. If the edit changes the model's behavior in the manner that would be expected from changing the target behavior, we take this as evidence in support of localization.
</p>

<p>
For example, if editing a particular location in the network shifts the model to give truthful answers, we may take this as evidence that the location meaningfully encodes truthfulness in some sense. Or, if editing a location causes the model to act as though the Eiffel tower is in Rome, we may take this as evidence that the location encodes the concept of the Eiffel tower. The basic question in this paper is: how strong is this evidence? That is, to what extent can we conclude that a particular location in the model is responsible for a particular behavior based on the success of editing at that location?
</p>

<p>
Our core contribution is an example where editing-based evidence appears very strong, but where localization clearly fails. The example replicates the setup of Inference-Time-Interference (ITI) <d-cite key="li2024inference"></d-cite>, where the target concept is truthfulness, and the localization is in a small subset of 16 attention heads. Following ITI, we use logit-linear probing to identify candidate heads. We then search for the optimal localized edit to apply at these heads. Remarkably, we find that the optimal edit induces truthfulness behavior that is essentially as good as finetuning the entire model to be truthful. That is, the localized edit is as effective as can possibly be expected. Intuitively, this appears to be strong evidence that the locations found by the heuristic (probing) are indeed closely linked to the target concept (truthfulness). However, we then show that this evidence is misleading. We find that applying optimal edits to <em>random heads</em> are just as effective as when applied to the localized heads. Accordingly, the edit-based evidence provides no support for the localization hypothesis.
</p>

<p>
A possible out here is that 16 attention heads is too many, leaving us with significant leeway to induce any behavior we want with editing. We further strengthen the example by showing that it is possible to find a <em>single</em> head in the model where editing at that head is as effective as finetuning the entire model. This appears to be the strongest edit-based evidence for localization possible. However, we show that there are in fact multiple such heads. That is, there is simply no single privileged location that can be identified as responsible for the target behavior.
</p>

<p>
Our results suggest that the evidence provided by editing is weak, and that the success of editing at a particular location is not a reliable indicator of the location's importance for the target behavior. This seems to significantly constrain what can be learned from interpretability methods. It also points to the need for a more rigorous development of such techniques, including both precise statements of what the goals are, and well-grounded standards for evidence that these goals have been met.
</p>

<p>
The technical development in this paper relies on finding the optimal intervention at a specified location. To that end, we develop a method for localizing LoRA type finetuning to specific locations. This then allows us to frame the search for optimal edits as a finetuning-type optimization problem. This method may also be of independent interest.
</p>


## Backgrounds and Results from ITI

<p>
We replicate the setup of ITI <d-cite key="li2024inference"></d-cite>.
</p>

<h3>Dataset and Model Architecture</h3>

<p>
We use TruthfulQA <d-cite key="lin2021truthfulqa"></d-cite> as our dataset. It contains 817 questions that humans might answer incorrectly due to misconceptions. Each question contains an average of 3.2 truthful answers and 4.1 false answers. We use 60% of the questions for training, and the rest for validation and testing.
</p>

<p>
We use an Alpaca-7B <d-cite key="taori2023alpaca"></d-cite> model that is finetuned from the Llama-7B base model. The model consists of $L = 32$ layers, each consisting of a Multi-head Attention (MHA) layer, and a Multilayer Perceptron (MLP) layer. We focus on the MHA layer, which has $H = 32$ attention heads, with each head having dimension $H = 128$ (the hidden dimension is $DH = 4096$).
</p>

<p>
Ignoring MLP and layer normalization, the computation at layer $l$ can be written as:
</p>

<a id="eq1"></a>
$$
\mathbf{o}_h^l := \text{Attn}_h^l(\mathbf{r}^l) \in \mathbb{R}^D
$$

<a id="eq2"></a>
$$
\mathbf{o}^l := [(\mathbf{o}_1^l)^T, \ldots, (\mathbf{o}_H^l)^T]^T \in \mathbb{R}^{DH}
$$

<a id="eq3"></a>
$$
W^l := [W_1^l, \ldots, W_H^l] \in \mathbb{R}^{DH \times DH}
$$

<a id="eq4"></a>
$$
\mathbf{r}^{l+1} := \mathbf{r}^l + W^l \mathbf{o} = \mathbf{r}^l + \sum_{h=1}^H W_h \mathbf{o}_h \in \mathbb{R}^{DH}
$$


<p>
where $\mathbf{r}^l \in \mathbb{R}^{DH}$ is the residual stream before layer $l$, $\text{Attn}_h^l$ is the $h$-th attention module at layer $l$, with $\mathbf{o}_h^$ being its output. $\mathbf{o}^l$ is the concatenated head outputs. $W^l$ is the project-out matrix, that applies $H$ independent linear transformations to the corresponding head outputs. Finally $\mathbf{r}^{l+1}$ is residual stream output after layer $l$.
</p>

<h3>Localization and Intervention Using Activation Statistics</h3>

<p>
To localize, we collect representations for positive and negative examples, and use probing to find where the truthfulness concept is represented. To intervene, we find the direction best separating activations for positive and negative examples, and apply this direction to the representation.
</p>

<p>
Each example is of the form, $(x, y, x_{\text{random}})$, concatenating a question $x$, a corresponding answer $y$, and another random question $x_{\text{random}}$. For positive examples, we use a truthful response $y = y_{+}$, and for negative examples, we use an untruthful response $y = y_{-}$. To collect the representations, we feed the positive and negative examples through the model, and collect the activations of the attention heads, $\{\mathbf{o}_h^l\}_{h \in [H], l \in [L]}$, at the last token.
</p>

<p>
For each of the $L \times H$ head locations, we train a logistic regression probe on the $D$-dimensional activations to predict whether it's a positive or negative example. Then we pick the attention heads with the highest probing accuracies as the localized heads.
</p>

<p>
For the selected head at $(l, h)$, we find the direction $\mathbf{u}_h^l$ that is "best" at separating the activations of positive and negative examples. There are several variants, but according to <d-cite key="li2024inference"></d-cite>, the best option is the mass mean shift, which is the difference between the average positive and negative activations. Then we estimate the standard deviation of activations along the direction to be $\sigma_h^l$, and use the weighted direction $\boldsymbol{\theta}_h^l := \sigma_h^l \mathbf{u}_h^l$ as the intervention vector, which we add to the corresponding head during inference autoregressively.
</p>

<p>
More specifically, the applied intervention is:
</p>

<a id="eq5"></a>
$$
\mathbf{r}^{l+1}_{\text{ITI}} := \mathbf{r}^l + W^l ( \mathbf{o} + \alpha \boldsymbol{\theta}^l)
$$

<a id="eq6"></a>
$$
= \mathbf{r}^{l+1}_{\text{orig}} + \alpha W^l \boldsymbol{\theta}^l = \mathbf{r}^{l+1}_{\text{orig}} + \alpha \sum_{h=1}^H W_h^l \boldsymbol{\theta}_h^l
$$

<p>
where $\mathbf{\theta}_l$ is the concatenated intervention vectors across all heads at layer $l$, and $\alpha$ is the intervention strength. This intervention is repeated for each next token prediction autoregressively until the whole answer is completed.
</p>

<h3>Evaluation Metrics</h3>

<p>
Since the goal is to assess model's generation quality, it's natural to use truthfulness score and informativeness score of generations as the evaluation metrics. They use GPT-judge models <d-cite key="lin2021truthfulqa"></d-cite> to evaluate the model's generations for truthfulness and informativeness, and use Info*Truth (the product of scalar truthful and informative scores) as the main metric.
</p>

<p>
We also report other metrics as in the ITI paper: KL divergence of the model's next-token prediction distribution post- versus pre-intervention, and multiple-choice accuracy (MC) which is determined via comparing the conditional probabilities of candidate answers given the question.
</p>

<h2>Editing Localized Heads Modifies the Output as Expected</h2>

<p>
In ITI, the authors find that editing on 16 localized heads (out of a total of 1024 heads) successfully steers model generations to be more truthful while still being informative. They also find intervening on all attention heads doesn't make model generations more truthful than intervening just at the localized heads. This seems to suggest that the truthfulness concept is indeed encoded in the localized heads. 
</p>

<p>
We now strengthen this evidence further. Similar to <d-cite key="hase2024does"></d-cite>, we check if interventions at random heads can also make model generations more truthful. More specifically:
</p>

<p>
1. Randomly select 16 heads, and compute intervention vectors $\theta$'s accordingly. 
</p>

<p>
2. Apply varying intervention strength α, collect model generations, and compute scores for truthfulness and informativeness using GPT-judge across all intervention strengths.
</p>

<p>
3. Repeat for 16 times. 
</p>

<p>
We find that interventions at the localized heads are more effective than interventions at random heads. In <a href="#fig-iti-hist">Figure below</a> we report the Info*Truth score (average truthfulness score times average informativeness score). We find that using localized heads have significantly higher Truth*Info scores than using random heads (p-value $1.6×10^{-8}$). In fact, using random heads often doesn't have noticeable effect on the truthfulness at all, as shown in the truth-info plot (<a href="#fig-truth-info">Figure</a>) and KL-MC trade-off plot (<a href="#fig-kl-mc">Figure</a>).
</p>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0" id="fig-iti-hist">
      {% include figure.html path="assets/img/2025-04-28-localization/hist_iti.png" title="Info*Truth Scores" class="img-fluid" %}
      <div class="subfig-caption" style="font-size: 0.5em; color: #ffffff; background-color: rgba(0, 0, 0, 0.7); padding: 1px; margin-top: 1px; text-align: center;">
          Info*Truth Scores
      </div>
  </div>
  <div class="col-sm mt-3 mt-md-0" id="fig-truth-info">
      {% include figure.html path="assets/img/2025-04-28-localization/iti_truth_info.png" title="Truth vs Info Scores" class="img-fluid" %}
      <div class="subfig-caption" style="font-size: 0.5em; color: #ffffff; background-color: rgba(0, 0, 0, 0.7); padding: 1px; margin-top: 1px; text-align: center;">
          Truth vs Info Scores
      </div>
  </div>
  <div class="col-sm mt-3 mt-md-0" id="fig-kl-mc">
      {% include figure.html path="assets/img/2025-04-28-localization/iti_kl_mc.png" title="KL vs MC Scores" class="img-fluid" %}
      <div class="subfig-caption" style="font-size: 0.5em; color: #ffffff; background-color: rgba(0, 0, 0, 0.7); padding: 1px; margin-top: 1px; text-align: center;">
          KL vs MC Scores
      </div>
  </div>
</div>
<div class="caption" id="fig-localization">
   Localized heads perform much better than random when using ITI interventions. We observe better Truth*Info scores, better truth-info score tradeoff, as well as better MC-KL tradeoff.
</div>

<p>
This appears to add further evidence that the localized heads are "special" for the truthfulness concept. However, this strong association could be because the intervention and localization are "correlated", since both use statistics of the same activations (determined by the design of the data, etc). E.g. for heads with very low probing accuracy, the estimated intervention vectors could be very noisy, and thus the interventions could be less effective.
</p>

<h2>Finding "Optimal" Interventions</h2>

<p>
To test whether a particular behavior is localized to specific location, we would like to assess the effect of the <em>optimal</em> intervention at that location. In the case of our running example, we want the localized edit to the representation space that does the best job of steering the model's generations to be more truthful while maintaining informativeness. Then, the questions are: what is the best we could hope to achieve? (I.e., what is "optimal"?) And, (how) can we find a localized edit that achieves it?
</p>

<h3>Fitting the Alignment Objective Gives Optimal Interventions</h3>

<div id="fig-ipo">
{% include figure.html path="assets/img/2025-04-28-localization/stronger_evidence_for_loc.png" class="img-fluid" %}
<div class="caption">
IPO interventions achieve much better performance than using ITI. Using IPO interventions at localized heads give nearly optimal info-truth tradeoff as well.
</div>
</div>

<p>
The key observation is that the dataset used to construct positive and negative examples can be restructured as paired "preference" data $\{(x_i, y_i^+, y_i^-)\}_i$, where $x_i$ is the question, $y_i^+$ is the truthful answer, and $y_i^-$ is the untruthful answer. Since the goal is to make model generations more truthful, we can directly adopt contrastive alignment methods for biasing the model towards the truthful answers. In this case, we use the IPO <d-cite key="azar2024general"></d-cite> learning objective, where the goal is to upweight probabilities for $y_i^+$ and downweight probabilities for $y_i^-$ (up to some threshold):
</p>

<p>
$
\text{argmax}_{\phi} \sum_i \left[\log \left( \frac{\pi_{\phi}(y_i^{+} | x_i)}{\pi_0(y_i^{+} | x_i)} / \frac{\pi_{\phi}(y_i^{-} | x_i)}{\pi_0(y_i^{-} | x_i)} \right) - \frac{\tau^{-1}}{2}\right]^2
$
</p>

<p>
where $\pi_\phi(\cdot \vert x)$ is the model's generation probability, $\pi_0(\cdot \vert x)$ is the original model's generation probability, and $\tau$ decides the threshold. Ideally, the optimized $\pi_{\phi^*}(\cdot \vert x)$ should generate responses that are more truthful than the original model, while minimally affecting the off-target aspects of the generation (in this case, the informativeness of the responses).
</p>

<p>
To test the effectiveness of IPO alignment, we finetune the weights for project-out matrices $W^l$'s defined in <a href="#eq3">Equation 3</a> using (rank 1) LoRA <d-cite key="hu2021lora"></d-cite>. The finetuned model gives nearly perfect trade-off between truthfulness and informativeness, that is far better than ITI interventions (see <a href="#fig-ipo">Figure</a>). This also suggests that ITI heuristics are very far from optimal and contrasts with ITI results that intervening on all heads doesn't make model generations more truthful.
</p>


<p>
Now we treat this result as the overall best performance that we can achieve with interventions. We want to see if optimal interventions at localized heads can achieve the same performance, and if random heads can achieve the same performance.
</p>

<h3>Connect Weight Updates to Representation Editing</h3>

<p>
The connection to IPO lets us search for the best possible update to the model's <em>weights</em>. However, we are interested in localized edits to model <em>representations</em>. To continue, we need to connect the weight editing to representation editing.
</p>

<h4>Rank-1 LoRA</h4>

<p>
Directly applying rank-1 LoRA to W^l, we can view the effect of adding in the modified LoRA weight matrix as an edit to the representation as follows:
</p>

<p>
$
\mathbf{r}_{\text{LoRA}}^{l+1} := \mathbf{r}^l + (W^l + \mathbf{b}^l (\mathbf{a}^l)^T)\mathbf{o}^l = \mathbf{r}_{\text{orig}}^{l+1} + \left< \mathbf{a}^l, \mathbf{o}^l\right> \mathbf{b}^l
$
</p>

<p>
where $a^l, b^l$ are the LoRA weights to optimize. Comparing with <a href="#eq5">the ITI intervention</a> , we see that $b^l$ plays the role of the added $W^l \theta^l$, and $\langle a^l, o^l \rangle$ is the intervention strength but is adapted to the representation $\mathbf{o}$.<d-footnote>One could replace $\langle a^l, o^l \rangle$ with a constant intervention strength, but allowing the extra flexibility is closer to the ideal of best-possible-localized-intervention.</d-footnote>
</p>

<p>
This formulation connects weight edits to representation edits. However, it doesn't yet allow us to localize edits to specific heads -- while $\theta^l$ can be read as concatenation of headwise intervention vectors, the projected $W^l \theta^l$ have no corresponding interpretations. Therefore, we can't restrict the edits to specific heads by imposing structure on $b^l$'s.
</p>

<h4>Rank-1 LoRA with Reparameterization</h4>

<p>
We can make more direct connections by reparameterizing $b^l$ with $W^l b^l$ (without changing expressiveness):
</p>

<p>
$
\mathbf{r}_{\text{LoRA-reparam}}^{l+1} := \mathbf{r}_{\text{orig}}^{l+1} + \left< \mathbf{a}^l, \mathbf{o}^l\right> W^l \mathbf{b}^l = \mathbf{r}_{\text{orig}}^{l+1} + \left< \mathbf{a}^l, \mathbf{o}^l\right> \sum_{h=1}^H W_h^l \mathbf{b}_h^l
$
</p>

<p>
Here $b_h^l$ plays the role of the intervention vector $\theta_h^l$, and $a^l$ decides the intervention strength adaptively.
</p>

<p>
Now we have the algorithm to find the optimal interventions for the chosen set of heads:
</p>

<p>
1. Finetune the model weights using reparameterized LoRA with the IPO objective.
</p>

<p>
2. And, restrict $b^l$ to be nonzero only for the chosen set of heads.
</p>

<h2>Optimal Interventions at Localized Heads are Nearly Optimal, but so are Random Heads</h2>

<div class="row mt-3">
   <div class="col-sm mt-3 mt-md-0" id="hist_ipo">
       {% include figure.html path="assets/img/2025-04-28-localization/hist_ipo.png" class="img-fluid" %}
   </div>
   <div class="col-sm mt-3 mt-md-0" id="random_vs_top">
       {% include figure.html path="assets/img/2025-04-28-localization/random_vs_top.png" class="img-fluid" %}
   </div>
</div>
<div class="caption">
Using IPO optimal localized interventions, randomly selected heads perform nearly optimally for steering model generations. In particular, random heads are as good as the conjectured localized heads. The random heads are the same as those in <a href="#fig-iti-hist">earlier truth-info plots</a>.
</div>

<h3>Optimal Edits at Conjectured Localization</h3>

<p>
We can now search for the best possible interventions at the localized heads. The earlier figure shows the result. We find that the optimal interventions strongly outperform the heuristic ITI interventions. Moreover, the localized interventions are about as effective as full IPO alignment! This appears to be the strongest edit-based evidence for localization that we could hope for.
</p>

<h3>Optimal Edits at Random Localization</h3>

<p>
Now, we apply the same optimal edit procedure to 16 <em>randomly selected</em> heads. The figure above shows the results. In short: the optimal interventions at random heads are often just as effective as the optimal interventions at the localized heads. Accordingly, the fact that editing at the localized heads was effective at steering generations provides no evidence that the truthfulness concept is localized to those heads.
</p>

<p>
Further, the random heads we use here are the same random heads used earlier. Using the ITI heuristic intervention, the selected heads looked highly different from these random heads. But we now see that this appears to be an artifact of the suboptimal interventions and choice of metric, rather than a meaningful difference in how the heads relate to truthfulness.
</p>

<h2>Intervening a Single Head is Just as Effective</h2>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0" id="hist_ipo_1">
      {% include figure.html path="assets/img/2025-04-28-localization/hist_ipo_1.png" class="img-fluid" %}
  </div>
  <div class="col-sm mt-3 mt-md-0" id="single_vs_top">
      {% include figure.html path="assets/img/2025-04-28-localization/single_vs_top.png" class="img-fluid" %}
  </div>
</div>
<div class="caption">
Using a single-head is as effective, and there are multiple of them!
</div>

<p>
It is now clear that edit-based evidence does not provide strong evidence for localization in the 16 head setup. However, a possible way of saving localization would be to argue that 16 heads is too many, giving too much leeway to induce any behavior we want with editing. For example, if we edited half the heads of the model, it would not be surprising if we could make the model do anything we wanted. Accordingly, we might hope that there is still a valid syllogism of the form "the localized edit is extremely constrainted" and "edits at this location optimally control the target behavior" implies "the target behavior is localized to this location".
</p>

<p>
To test this, we now focus on the single head case. The procedure is simple: we randomly sample 24 single heads, one at a time, and search for optimal interventions. The distribution of the best Truth*Info scores is shown in <a href="#hist_ipo_1">the figure above</a>. We find  <a href="#single_vs_top">5 single-heads that are as effective</a>, and none of them has high probing accuracy. Notice that, still, none of these heads can be understood as localizing the truthfulness concept. The reason is that there are multiple distinct locations that work equally well! That is, even in the extreme case of a very localized edit that replicates the target behavior essentially optimally, we still cannot conclude that there is evidence supporting localization.
</p>

<h2>Are the Probing-Localized Heads Anything Special?</h2>

<p>
So far what we mean by localization, is that we can change model generation on target concept by an edits at this location. And our experiments show no evidence for this type of localization, and probing-localized heads play no special role.
</p>

<p>
So, are the probing-localized heads anything special at all?
</p>

<h3>Probing-localized Heads Seems Special for MC Scores</h3>

<p>
We do observe that these heads achieve slightly better Multiple-Choice (MC) scores compared to randomly selected heads (see <a href="#random_vs_top_MC_KL"> figures below</a>), although this advantage is not as pronounced as with the ITI interventions (see <a href="#fig-kl-mc">earlier figures</a>). Thus, these heads may be special in terms of changing model probabilities on the given fixed dataset, which is what MC measures.
</p>

<h3>The Gap Between What the Model "Knows" and What it Generates</h3>

<p>
It's important to note that the model's probabilities for fixed responses, do not directly correspond to what the model actually generates. Even if the model assigns a higher probability to a truthful response than an untruthful one, it may still not generate the truthful response if the fixed dataset is off-policy (i.e. both probabilities are low). This highlights the well-known gap between what a model "knows" (which is the motivation behind probing) and what it ultimately generates <d-cite key="joshi2023personas,wang2020language,kadavath2022language,saunders2022self,burns2022discovering"></d-cite>.
</p>

<div id="random_vs_top_MC_KL">
{% include figure.html path="assets/img/2025-04-28-localization/random_vs_top_MC_KL.png" class="img-fluid" %}
<div class="caption">
Probing-localized heads seem somewhat special in MC scores.
</div>
</div>

<h3>Implications</h3>

<p>
It's possible that while probing-localized heads are not special at all for controlling model generations, they are special in changing what the model "knows". Though we caution that the results here are not rigorous evidence for localization even in this sense. Even if there is a knowledge localization in some sense, it is clear that this does not inform steering, and does not give a way of monitoring model behavior (because changes in completely unrelated locations can change the behavior). This points to the need for making the goal of localization precise.
</p>

<h2>Discussion</h2>

<p>
The main idea in this paper is that to assess the localization of a behavior we should study the effect of the <em>optimal</em> intervention at the conjectured localization. The main obstacle is that, in general, it is not clear how to define or find the optimal intervention. To overcome this, we map the problem of finding the optimal intervention to the problem of finding the optimal weight update, which can be solved using existing LLM alignment methods.
</p>

<p>
The main result is an example where, naively, the evidence for localization appears strong, but when we use optimal interventions, the evidence disappears. 
</p>

<p>
The particular example—truthfulness and ITI-based evidence—was selected simply because the data used to define the heuristic happens to also allow us to set up a contrastive alignment problem. The most limited read of the results here is that ITI interventions do not provide evidence for localization, and that truthfulness does not appear to be localizable. However, the broader point is that by giving an example where editing-based evidence doesn't support localization, we see that in general such edits—by themselves—cannot provide evidence for localization. This is true irrespective of the particular behavior or heuristic being evaluated.
</p>

<p>
Thus far, we've been a bit vague about what localization means. Editing does tautological evidence for localization in the sense of "it's possible to modify model behavior on such-and-such a behavior by an edit at this location". On the opposite end, the strongest possible standard would be to show that the location is unique, or at least necessary. This is the standard that would be required if our aim was, e.g., to establish that LLM truthfulness can be monitored by examining a small set of heads. Potentially, there are interesting and useful notions of localization in between these two extremes. However, we can see no useful sense of localization that is consistent with the location being only as good as a <em>randomly selected</em> alternative. As we have seen, heuristic edit-based evaluation cannot even rule out this case.
</p>

<p>
Our findings add to a growing body of work that assesses the validity of interpretability results. <d-cite key="niu2024does"></d-cite> argue that the Knowledge Neuron thesis, which suggests that facts are stored in MLP weights, is an oversimplification and does not adequately explain the process of factual expression in language models. <d-cite key="makelov2023subspace"></d-cite> demonstrate that subspace activation patching can lead to an illusory sense of interpretability, as the effects may be achieved through dormant parallel pathways rather than the hypothesized subspaces. Most relevant to our work, <d-cite key="hase2024does"></d-cite> find that localization conclusions from causal tracing do not provide insight into which model MLP layer would be best to edit to override an existing stored fact. 
</p>

<p>
Overall, the results here point to the need for precise statements of what the objectives are in interpretability. With clear objectives, it may be possible to develop theoretically grounded methods for evaluation. Precise, falsifiable, statements and clear standards of evidence would suffice to prevent the kind of failure we observe in this paper.
</p>

<h2>Experiment Details</h2>

<h3>Dataset and Model Architecture</h3>
<p>
We use the TruthfulQA dataset <d-cite key="lin2021truthfulqa"></d-cite> and the Alpaca-7B model <d-cite key="taori2023alpaca"></d-cite> for our experiments. The dataset contains 817 questions with truthful and untruthful answers. We turn them into pairs, and use 60% for training (6560 paired data) and the rest for validation and testing. The model consists of 32 layers, each with 32 attention heads and a hidden dimension of 4096.
</p>

<h3>Training Details</h3>
<p>
We use IPO objective <d-cite key="azar2024general"></d-cite> and use hyperparameter $\tau = 0.1, 0.2, 0.3, 0.4, 0.5$. We train for two epochs with a cosine scheduler, with a batch size of 4. We use "paged_adamw_32bit" optimizer. For training with different numbers of heads, we find a smaller number of heads benefit from a higher learning rate. For all-heads, we use a learning rate of $1 \times 10^{-4}$, and for 16 heads, we use $5 \times 10^{-4}$. For single-head, we use $2 \times 10^{-3}$.
</p>

<h3>Evaluation Metrics</h3>
<p>
We reuse code from ITI <d-cite key="li2024inference"></d-cite> for evaluation when possible. For GPT-judge models, we follow <d-cite key="lin2021truthfulqa"></d-cite> and finetune on truthfulness and informativeness dataset using OpenAI API <d-cite key="openai2020api"></d-cite>. Our finetuned model achieves similar validation error as in <d-cite key="lin2021truthfulqa"></d-cite>.
</p>