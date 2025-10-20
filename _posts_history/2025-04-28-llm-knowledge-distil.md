---
layout: distill
title: On LLM Knowledge Distillation - A Comparison between Forward KL and Reverse KL
description: In this blog post, we delve into knowledge distillation techniques for Large Language Models (LLMs), with a particular focus on using Kullback-Leibler (KL) Divergence as the optimization objective. Knowledge distillation is a powerful tool to reduce model size while maintaining comparable performance, making it especially useful in scenarios with constrained computational or serving resources. We specifically explore the nuances of Forward KL divergence and Reverse KL divergence, examining their roles in the distillation process. By comparing these two approaches, we aim to uncover their behaviours, strengths, and practical applications in LLM distillation.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-llm-knowledge-distil.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Background
  - name: Preliminaries
    subsections:
    - name: Knowledge Distillation in Conditional Language Modeling
    - name: Forward KL Divergence and Reverse KL Divergence
  - name: Modern LLM Distillation Methods
    subsections:
    - name: Modeling from Token Level
    - name: Modeling from Sequence Level
  - name: Empirical Comparison, how it works and which one is better?
    subsections:
    - name: Experiment Details
    - name: Implementation of Token-level Knowledge Distillation
    - name: Empirical Study on Task-Agnostic Knowledge Distillation
  - name: Closing Remarks

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

## Background
<!-- TODO: 11/18 -->
In recent years, knowledge distillation (KD)<d-cite key="hinton2015distilling"></d-cite> has gathered significant attention, particularly in the area of generative large language models (LLMs). Open-source LLMs, such as Qwen2<d-cite key="yang2024qwen2"></d-cite> and LLaMA-3<d-cite key="dubey2024llama"></d-cite>, come in a wide range of sizes, from 0.5 billion to 405 billion parameters. This diversity makes them ideal candidates for KD: training a high-performing model with a larger teacher model and subsequently distilling its knowledge into a smaller student model. This process is especially valuable when computational resources are constrained.

There are two primary approaches to knowledge distillation in LLMs:
1. **Sequence level distillation**: This involves prompting the teacher model to generate responses, which are then used for distillation training. This approach is particularly effective when the teacher is a black-box system, accessible only via APIs.

2. **Token level distillation**: This approach aligns the intermediate outputs, such as logits or embeddings, between the teacher and student models. By focusing on token-level or layer-level alignment, it enables the distillation of deeper knowledge beyond the model’s final outputs.

In this blog post, we specifically focus on white-box knowledge distillation, which provides greater access to the teacher model’s internal representations and mechanisms. 
<!-- While knowledge distillation has been extensively explored in computer vision, directly applying computer vision based KD methods to LLMs sometimes falls short.  -->
Unlike traditional knowledge distillation, which is primarily applied to classification-dominated tasks, knowledge distillation for modern generative language models presents unique challenges. For example, vanilla knowledge distillation using forward KL divergence as the loss function has been shown to introduce issues such as hallucination in language models. This could arise from forward KL's inherent tendency toward mean-seeking optimization, which can lead to distorted or unrealistic outputs in generative tasks.

In this blog post, we will:

- Provide an overview of knowledge distillation in generative language models, and the theoretical underpinnings of forward KL and reverse KL Divergence.
- Review and analyze recent advancements in KD for LLMs, conducting an empirical comparison of methods.
- Investigate forward KL divergence and reverse KL divergence fitting behaviour in detail.

By bridging these gaps, we aim to advance the understanding and application of KD in generative language modeling tasks. 
<!-- Knowledge distillation (KD) has been widely used in computer vision and many methods have been proposed to improve small model performance on image classification, vision understanding and many other tasks. However, directly applying popular knowledge distillation methods in CV doesn't always grant good performance on generative language models. For example, vanilla KD using forward KL divergence as the loss function, has been observed that language models distilled in this way could have hallucination problems due to the forward KL divergence's mean seeking problem. For computer visio In this blogpost, we first introduce language modeling, knowledge distillation methods and FKL, RKL in detail. Then, we review and delve deep into the prevailing knowledge distillation methods proposed recently, and do an empirical comparison. We also explore the reasons why knowledge distillation methods from computer vision does not always promise good results in language modeling tasks. -->

## Preliminaries
<!-- 11/18 -->

### Knowledge Distillation in Conditional Language Modeling
<!-- Given a source-target pair (or known as instruction-response) $(x,y)$, a language model $M$ aims to accept $x$, and output $\hat{y}=M(x)$, where the optimization goal is to minimize the distance between $\hat{y}$ and $y$. Here $x$, $y$ and $\hat{y}$ are sentences and the gradients are applied in the sentence level. 
For knowledge distillation in conditional language modeling, given the source, or instruction $x$, a teacher language model would output a distribution $p(y|x)$ and a student model would output a distribution $q_\theta{y|x}$ parameterized by $\theta$. The aim is to minimize the distance between the teacher distribution $p(y|x)$ and the student distribution $$q_\theta{y|x}, so that the student model can "mimic" the teacher distribution. Normally, to stablize the training, the distillation loss is always added to the supervised finetuning loss. -->

Given a source-target pair (commonly referred to as an instruction-response) $(x,y)$, a language model $M$ is trained to accept the input $x$ and produce an output $\hat{y}=M(x)$, with the optimization objective being to minimize the discrepancy between $\hat{y}$ and $y$. Here, $x$, $y$, and $\hat{y}$ are sentences, and gradients are computed at the sentence level.

In the context of **knowledge distillation** for conditional language modeling, given an input source or instruction $x$, a **teacher model** generates a probability distribution $p(y\mid x)$, while a **student model**, parameterized by $\theta$, generates a distribution $q_\theta(y\mid x)$. The goal of knowledge distillation is to minimize the divergence between the teacher's distribution $p(y\mid x)$ and the student's distribution $q_\theta(y\mid x)$, enabling the student model to "mimic" the teacher's behavior.

To ensure stable training, the distillation loss is typically combined with the supervised fine-tuning loss, allowing the student model to balance imitation of the teacher with alignment to ground truth data.

### KL Divergence
The Kullback-Leibler (KL) divergence is a commonly used measure of the "distance" of two distributions. It can identity how far one distribution is to another. This is very useful in knowledge distillation cause the optimization goal we mentioned above is to make the student distribution similar enough to the teacher distribution. Using the denotes we have to formulate the KL divergence in knowledge distillation problems, given a student distribution (approximate distribution) $q_\theta(y\mid x)$ and a teacher distribution (true distribution) $p(y\mid x)$, the KL divergence can be formulated as<d-footnote>https://dibyaghosh.com/blog/probability/kldivergence.html</d-footnote>:

$$
D_{KL}(p\|q_\theta)=\mathbb{E}_{y \sim p} \left[ \log \frac{p(y\mid x)}{q(y\mid x)} \right] \tag{1}
$$

To be noticed, KL divergence is not a "symmetric" measure, which means that $D_{KL}(p\|q_\theta)$ is not completely equal to $D_{KL}(q_\theta\|p)$, even though the "meaning" is the same -- how similar one distribution is to the other one.

### Forward KL Divergence and Reverse KL Divergence
The difference between $D_{KL}(p\|q_\theta)$ and $D_{KL}(q_\theta\|p)$ becomes very prominent when using this KL divergence in optimization, i.e. minimizing the difference between two distributions. When we let student distribution to fit the real distribution, or the teacher distribution here, different order of $p$ and $q_\theta$ will result in difference in fitting performance, especially in the first several steps.

Suppose $D_{FKL}=D_{KL}(p\|q_\theta)$ and $D_{RKL}=D_{KL}(q_\theta\|p)$, where $D_{FKL}$ refers to forward KL divergence and $D_{RKL}$ refers to reverse KL divergence, the optimization goal can be formulated as:

$$
\begin{aligned}
\arg\min_{\theta} D_{FKL} &= \arg\min_{\theta} D_{KL}(p\|q_\theta) \\
&= \arg\min_{\theta} \mathbb{E}_{y \sim p} \left[ \log \frac{p(y|x)}{q_\theta(y|x)} \right] \\
&= \arg\max_{\theta} \mathbb{E}_{y \sim p} \left[ \log q_\theta(y|x) \right]
\end{aligned} \tag{2}
$$

and for reverse KL:

$$
\begin{aligned}
\arg\min_{\theta} D_{RKL} &= \arg\min_{\theta} D_{KL}(q_\theta \| p) \\
&= \arg\min_{\theta} \mathbb{E}_{y \sim q_\theta} \left[ \log \frac{q_\theta(y|x)}{p(y|x)} \right] \\
&= \arg\max_{\theta} \mathbb{E}_{y \sim q_\theta} \left[ \log p(y|x) \right] + \mathcal{H}(q_\theta(y|x))
\end{aligned} \tag{3}
$$

Forward KL is a mean-seeking behavior, while Reverse KL is a Mode-Seeking behavior <d-footnote>https://dibyaghosh.com/blog/probability/kldivergence.html</d-footnote> . This observation is often demonstrated by using one gaussian distribution to fit sum of two gaussian distributions. the fit progress could be found in the following figures.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/fwd_kl.gif" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/rev_kl.gif" class="img-fluid " %}
    </div>
</div>
<div class="caption">
    Figure 1. One Gaussian distribution fitting sum of two Gaussian distributions.
</div>

To understand this phenomenon, $D_{FKL}$ represents the expectation calculated under the $p$ distribution, so it will match $q_\theta$ to $p$ where $p$ is high and the $q_\theta$ is low during the first steps, so $D_{FKL}$ will firstly increase $q_\theta$ where $q_\theta$ is low and $p$ is high. Under the condition of fitting two Gaussian distributions with one Gaussian distribution, the forward KL divergence will make the fitting distribution to be mean-seeking behavior

On the other hand, $D_{RKL}$ represents the expectation calculated under the $q_\theta$ distribution, so it will match $q_\theta$ to $p$ where $q_\theta$ is high and $p$ is low during the first steps, so $D_{RKL}$ will firstly decrease $q_\theta$ where $q_\theta$ is high and $p$ is low. Under the condition of fitting two Gaussian distributions with one Gaussian distribution, after reverse KL find a peak, it will stay at that Local optimum, which makes the fitting distribution to be mode-seeking behavior.

The behavior changes when a stronger student model is employed. In the following figures, we illustrate this by fitting the sum of two Gaussian distributions using sum of two Gaussian distributions. Both forward KL and reverse KL are capable of approximating the sum. Under these optimization settings, forward KL converges to a solution around step 100, while reverse KL achieves convergence around step 350. This suggests that with a sufficiently powerful student model and enough training steps, forward KL and reverse KL are likely to exhibit similar performance.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/fwd_kl_dual.gif" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/rev_kl_dual.gif" class="img-fluid " %}
    </div>
</div>
<div class="caption">
    Figure 2. Two Gaussian distributions fitting sum of two Gaussian distributions
</div>




The detailed code to generate these images could be found in `toy_example_fkl_rkl.py` and `toy_example_fkl_rkl_v2.py` 



## Modern LLM Distillation Methods
To better suit knowledge distillation methods with modern large language model finetuning, multiple methods have been proposed. In this section, we summarize these methods from two levels, distillation from token level and from sequence (sentence) level.

As previously discussed, source/input $x$, output $\hat{y}$ and target $y$ are all sentences. During finetuning and knowledge distillation, the gradients can be applied to the general sentence, or the separated tokens. Here we denote $y=\\{y_t\\}_{t=1}^T$ where $y_t$ refers to the token at position $t$, and $T$ refers to the length of the sentence, i.e. number of tokens in $y$. 

{% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/token-sequence-level.png" class="img-fluid" width="" %}

<div class="caption">
    Figure 3. Illustration of token level knowledge distillation and sequence level knowledge distillation.
</div>

### Modeling from Token Level
<!-- FKL, RKL, AKL -->
Most methods now are modeling the sentence distributions from token levels. By tokenizing the sentence $y$ into a sequence $\\{y_t\\}_{t=1}^T$, we can formulate the distillation optimization goal as:

$$
\begin{aligned}
\arg\min \mathcal{L}_{KL} &= \arg\min D_{KL}(p\|q_\theta) \\
&= \arg\min \sum_{t=1}^T D_{KL}( p( y_t \mid x,y_{1:t-1} ) \  \| \  q_{\theta}( y_t \mid x,y_{1:t-1} ) ) 
\end{aligned} \tag{4}
$$

For token level knowledge distillation, the optimization goal per token is the same as the ones frequently used in embedding distillation, and computer vision. Forward KL divergence and reverse KL divergence are both commonly used loss functions in token level distillation. There's no very clear observation or proof of which one would be a better choice for which case. Since the optimization goal is the same, the performance doesn't seem to differ a lot when the model is fully trained. Sometimes both work, and inspired by this, some people begin to add the forward KL and reverse KL together. <d-cite key="wu2024rethinking"></d-cite> proposes Adaptive KL (AKL) divergence, which is an adaptive weighted sum of reverse KL and forward KL.

### Modeling from Sequence Level
Different from token level distillation, sequence level distillation aims to let the student model match the teacher's output probability over the whole sequence. 
For a generative model, it acquires knowledge by learning from the real-world distributions, which is natural language here. By performing Monte Carlo sampling from this distribution, the model generates sentences. From a token-level perspective, learning at each token position can be seen as a token classification task. However, from a sequence-level perspective, the entire sentence represents a sample drawn from the generative model’s learned distribution. This fundamental characteristic emphasizes a key distinction between sequence-level and token-level knowledge distillation in large language models. In sequence-level knowledge distillation, Monte Carlo sampling is typically used to draw samples that approximate the target distribution, capturing the sequence-level dynamics of the model's behavior. This approach inherently differs from the token-level distillation process, where focus lies on individual token probabilities rather than the whole sequence.

From the perspective of implementation, the monte carlo sampling refers to `model.generate`. For a given input `source`, we can get two kinds of outputs from the model:
```
tokenized_inputs = tokenizer(**tokenized_inputs)
output_logits = model(**tokenized_inputs) # This is the logits/token level distribution.
output_sentence = model.generate(**tokenized_inputs) # This is the decoded/sampled sentence from the model.
```
where `output_logits` is the token level distribution, which is used to do token level distillation, and `output_sentence` is the sampled sequence, which is used in sequence level distillation.

Forward KL and Reverse KL are often employed in sequence-level knowledge distillation. The forward KL optimization goal can be formulated as:

$$
\arg\min D_{KL}(p\|q_\theta)=\arg\min \mathbb{E}_{y\sim p}\log \frac{p(y\mid x)}{q_\theta(y\mid x)}
$$

we can directly sample sentences $y from the teacher distribution. In simple terms, this optimization function lets the teacher model generate responses and uses the forward KL divergence as the loss function in knowledge distillation.

However, when the objective switches to reverse KL, i.e.,

$$
\arg\min D_{KL}(q_\theta\|p)=\arg\min -\mathbb{E}_{y\sim q_\theta}\log \frac{(y\mid x)}{q_\theta(y\mid x)}
$$

we need to sample from the student distribution. Since the student distribution is parameterized, it becomes infeasible to directly calculate the KL divergence for optimization, as in the case of forward KL. MiniLLM<d-cite key="gu2024minillm"></d-cite>addresses this challenge by approximating $D_{KL}(q_\theta\|p)$ using policy gradient methods, making it possible to optimize effectively even under the reverse KL setup. GKD<d-cite key="agarwal2023gkd"></d-cite> introduce Generalized Knowledge Distillation (GKD) to provide flexibility in choosing both the generation source (teacher model or student model) and the loss functions between the student and teacher.

<!-- TODO [ ] add GKD paper in somewhere -->

<!-- ### The relationship between LLM distillation and preference optimization -->

## Empirical Comparison, how it works and which one is better?
In this section, we present our empirical study on using forward KL and reverse KL for large language model distillation. We walk through the implementation of token-level knowledge distillation and common problems in knowledge distillation.

### Experiment Details
We use a subset of 20,000 examples randomly sampled from `HuggingFaceH4/ultrachat_200k`<d-footnote>https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k</d-footnote> dataset as sampled from real natural language distribution. For base models, we use `Qwen2.5` series models as the training starting point. All experiments are completed using one node with 8 Nvidia A100 80G GPUs. Code and datasets will be released in [ADD GITHUB LINK HERE].

### Implementation of Token-level Knowledge Distillation
In this section, we use forward KL as an simple example. For easy implementation and experimentation, we recommend using `trl` trainers<d-footnote>https://github.com/huggingface/trl</d-footnote> and Huggingface `alignment-handbook`<d-footnote>https://github.com/huggingface/alignment-handbook</d-footnote>.

We inherits `DistilTrainer` from `trl`'s `SFTTrainer`, so that we don't need to add some commonly used hyperparameters and functions. Similar implementation can be found in [this code repo](https://github.com/arcee-ai/DistillKit). Below is a comparison between the token level forward KL distillation implementation and the one at sequence level.



<div style="display: flex; width: 120%; height: 500px;">

<div style="flex: 1; resize: both; overflow: auto; border: 0px solid #ccc; padding: 10px;">
<pre>
{% highlight python %}
# Forward KL distillation at token level:
class TokenDistilTrainer(SFTTrainer):
    def distillation_loss(self, ...):
        
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature

        # Compute probabilities and log probabilities for both student and teacher
        student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)
        student_probs = F.softmax(student_logits_scaled, dim=-1) # noqa

        loss_kd = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction='batchmean',
            log_teacher=True,
        ) * (temperature ** 2) / max_seq_length

        return alpha * loss_kd + (1 - alpha) * original_loss

{% endhighlight %}

</pre>
</div>
<div style="width: 2px; background-color: #ccc; cursor: col-resize;"></div>
<div style="flex: 1; resize: both; overflow: auto; border: 0px solid #ccc; padding: 10px;"> 
<pre>
{% highlight python %}
# Forward KL distillation at sequence level
class SeqDistilTrainer(SFTTrainer):
    def distillation_loss(self, ...)
        prompt_lengths = (inputs['labels'] == -100).sum(dim=1)

        max_prompt_length = prompt_lengths.max()
        prompt_input_ids = inputs['input_ids'][:, :max_prompt_length].clone()

        positions = torch.arange(max_prompt_length).unsqueeze(0)  
        prompt_mask = positions < prompt_lengths.unsqueeze(1)

        prompt_input_ids[~prompt_mask] = self.tokenizer.pad_token_id
        prompt_attention_masks = prompt_mask.long()

        generated_sequences = teacher_model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_masks,
            max_new_tokens=getattr(self.args, 'max_new_tokens', 100),
            do_sample=True
        )

        gen_input_ids = generated_sequences
        gen_attention_mask = (gen_input_ids != self.tokenizer.pad_token_id).long()

        gen_inputs = {
            'input_ids': gen_input_ids, 
            'attention_mask': gen_attention_mask
        }

        student_outputs_gen = student_model(**gen_inputs)
        with torch.no_grad():
            teacher_outputs_gen = teacher_model(**gen_inputs)

        student_logits = student_outputs_gen.logits
        teacher_logits = teacher_outputs_gen.logits

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        loss_kd = self.forward_kl_loss(
            student_log_probs,
            teacher_log_probs,
            temperature
        )

        return alpha * loss_kd + (1 - alpha) * original_loss

{% endhighlight %}

</pre>
</div>

</div>



Here we mainly emphasize the distillation loss. For token level distillation, it's relatively easier since we don't need to sample the sequence from the teacher model. The `original_loss` refers to the original pretraining/supervised finetuning loss to stable the training. In token level distillation, we only need to call `model(**inputs)` to get the causal LM logits output, which means, probability distribution at each token position, and then call `F.kl_div` to calculate the forward KL divergence between the student logits and the teacher logits. However, for sequence level, we'll need to call `model.generate(prompt)` first the sample the sequence output from the teacher model, and then use `model(**gen_inputs)` to get the logits.

Two common problems arise during distillation training: (1) out-of-memory (OOM) issues and (2) low efficiency in sequence-level training. To address frequent OOM issues during distillation, we strongly recommend leveraging the DeepSpeed strategies. Specifically, refer to [the PPOTrainer example here](https://github.com/huggingface/trl/blob/10c2f63b2ac8564cca28aa1598a1f3ac6a5fc63c/trl/trainer/ppo_trainer.py#L1461) and ensure `_prepare_deepspeed` is implemented accordingly to optimize resource utilization. And for the second issue, low efficiency in sequence-level training occurs because it requires calling `model.generate`, which involves sampling from the learned distribution. To accelerate this process, we can (1) optimize `model.generate` with `vllm` and optimzie the GPU utilization (2) save the precomputed sequence outputs first and then load them during training. This also works when the teacher model is exceptionally large.

### Empirical Study on Task-Agnostic Knowledge Distillation
#### Token-level Forward KL and Reverse KL
In this section, we present our experiment results on token-level forward KL and reverse KL in LLM knowledge distillation in the Table. All models are evaluated using the same eval set on ROUGE1, ROUGE2, ROUGEL and BARTScore. We continue using the above denotations, where $q_\theta$ refers to the parameterized student model, and $p$ refers to the teacher model. Except for distilled models, we also present the supervised finetuning results as baselines. All experiments settings are kept the same, which are available in code repo.

| $q_\theta$ | $p$      | $q_\theta$ Size | $p$ Size | Loss                      | ROUGE1 | ROUGE2 | ROUGEL | BARTScore |
| ---------- | -------- | --------------- | -------- | ------------------------- | ------ | ------ | ------ | --------- |
| Instruct   | -        | 7B              | -        | -                         | 0.4613 | 0.2059 | 0.2705 | -2.5047   |
| Ultra      | -        | 1.5B            | -        | 1.0*SFT                   | 0.5295 | 0.2562 | 0.3414 | -2.5242   |
| Ultra      | -        | 7B              | -        | 1.0*SFT                   | 0.5576 | 0.283  | 0.364  | -2.4594   |
| Instruct   | Instruct | 7B              | 1.5B     | 0.5*SFT+0.5*FKL           | 0.5369 | 0.2595 | 0.3435 | -2.5134   |
| Ultra      | Instruct | 7B              | 1.5B     | 0.5*SFT+0.5*FKL           | 0.5404 | 0.2615 | 0.3463 | -2.5104   |
| Ultra      | Instruct | 7B              | 1.5B     | 0.8*SFT+0.2*FKL           | 0.5292 | 0.2567 | 0.3406 | -2.5235   |
| Ultra      | Instruct | 7B              | 1.5B     | 0.5*SFT+0.5*RKL           | 0.5291 | 0.2558 | 0.3408 | -2.5211   |


<div class="caption">
Table 1. Token level forward KL and reverse KL knowledge distillation results. Since the base model used here is Qwen2.5 family, we use Instruct to refer to Qwen2.5-instruct series models, for instance, Instruct 1.5B refers to Qwen2.5-1.5B-Instruct. Ultra refers to Qwen2.5-instruct models finetuned on the Ultrachat subset training set. For loss functions, SFT refers to supervised finetuning cross entropy loss, FKL refers to forward KL divergence loss, and RKL refers to reverse KL divergence loss.
</div>

From the above results, we can see that for token level, there's not much difference between forward KL and reverse KL. By adding the supervised finetuning cross entropy loss, the general learning can be more stablized. 

#### Convergence speed

Let's begin the comparison with a simple task. Consider a single-layer fully connected network without bias, where both the input and output dimensions are 64. The network's output is directly passed through a softmax layer.

We generated a fixed weight matrix with varying expected ranks and used this matrix as a fixed teacher model. Two student models with identical structures were trained, one guided by forward KL loss and the other by reverse KL loss.

Since forward KL loss and reverse KL loss differ in their formulations, their loss values are not directly comparable. Instead, we assess their convergence speeds using two proxy metrics: the L2 distance between the student and teacher probability distributions, and the L2 distance between the teacher's weights and the student's weights.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank2.jpg" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank4.jpg" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank8.jpg" class="img-fluid " %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank16.jpg" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank32.jpg" class="img-fluid " %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-llm-knowledge-distil/kl_cmp_rank64.jpg" class="img-fluid " %}
    </div>
</div>
<div class="caption">
    Figure 4. Convergence speed for Reverse KL and Forword KL.
</div>

Across all the images above, regardless of changes in the target matrix's rank, forward KL consistently outperforms reverse KL in both weight-L2 and L2 loss.

<!-- #### Token-level or Sentence-level? Comparison between AKL and GKD -->


## Closing Remarks

In this blog post, we examined knowledge distillation techniques for Large Language Models (LLMs), specifically comparing Forward KL and Reverse KL Divergence. Our empirical results demonstrate that both divergence measures perform similarly at the token level, with Forward KL showing faster convergence in simple scenarios. However, the choice between Forward KL and Reverse KL may depend on specific model architectures and training conditions.

We also highlighted the challenges of applying knowledge distillation methods from computer vision to generative language models, emphasizing the need for specialized approaches in the context of LLMs. Future research could explore hybrid divergence strategies or adaptive weighting to further optimize distillation performance.

Effective knowledge distillation remains crucial for developing efficient LLMs that maintain high performance while reducing computational requirements. By continuing to refine these techniques, we can enable broader deployment of powerful language models in diverse applications.


