---
layout: distill
title: Sigmoid and Softmax
description: Explanations of sigmoid and softmax rely on probabilistic interpretation of the output(s) but do not provide the source of probabilistic interpretation. In contrast, both sigmoid and softmax have a simple definition with their roots in basic information theory and calculus. This blog takes an isolated look at sigmoid and softmax and explains how and why they materialize. The definition and derivation show that they are related.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
   - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-sigmoid-and-softmax.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Sigmoid
  - name: Softmax
  - name: Discussion and Conclusion
  - name : References

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
    font-size: 16px;\
  }
---

## Introduction
Let $$x \in \mathbf{R}$$. Sigmoid function $$\sigma : \mathbf{R} \rightarrow  \left[0, 1\right]$$ is given by
$$
\begin{align}
    \sigma\left(x\right) &= \frac{1}{1 + \exp\left(-x\right)}.
\end{align}
$$
Sigmoid function is used in logistic regression and binary classification.


Let $$\mathbf{x} \in \mathbf{R^d}$$ and $$\mathbf{S} = \left\{\left(\mathbf{w}_0, \mathbf{w}_1, \ldots, \mathbf{w}_d\right) \mid \sum_{i=1}^d \mathbf{w}_i = 1\right\} \subset \left[0, 1\right]^d$$. Softmax function $$\mathbf{f} : \mathbf{R^d} \rightarrow  \mathbf{S} $$ is given by
$$
\begin{align}
    \mathbf{f}\left(\mathbf{x}\right)_i &= \frac{\exp{\left(-\mathbf{x}_i\right)}}{\sum_{j=1}^d \exp{\left(-\mathbf{x}_j\right)}}.
\end{align}
$$

Softmax function is used in multiclass classification, smooth maximum and softargmax. Softmax does not approximate the $$\arg\max$$ function but approximates the one-hot encoding of the $$\arg\max$$ function [1]. Let $$\mathbf{p} \in \left\{0, 1\right\}^d$$, $$\mathbf{\hat{p}} \in \left[0, 1\right]^d$$ and $$\mathbf{p}, \mathbf{\hat{p}} \in \mathbf{S}$$. Then
$$
\begin{align}
\left(\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_i, \ldots, \mathbf{p}_d\right) &\approx \left(\mathbf{\hat{p}}_0, \mathbf{\hat{p}}_1, \ldots, \mathbf{\hat{p}}_i, \ldots, \mathbf{\hat{p}}_d\right) \\
 &= \mathbf{f\left(\mathbf{x}\right)}.
\end{align}
$$

The approximation error is quantified by cross entropy
$$
\begin{align}
    \mathcal{L} &= \sum_{i=1}^d -\mathbf{p}_i\log\left(\mathbf{\hat{p}}_i\right).
\end{align}
$$

This should not confuse the reader whether softmax does or does not have its roots in $\arg\max$. The next section shows that indeed it does not. But it is still not known where softmax arises from. An analysis of an online time-series algorithm sheds some light on the softmax function [2] and the derivation of softmax can be established. Derivation of sigmoid is discussed before proceeding to softmax.


## Sigmoid
Consider the transformation
$$
\begin{align}
    p &= \arg\min_{ w \in [0,1]} wx
\end{align}
$$

$$p$$ with entropy regularizer
$$
\begin{align}
        \hat{p} &= \arg\min_{ w \in \left[0,1\right] } wx - w\log\left(w\right) - \left(1-w\right)\log\left(1-w\right)
    \end{align}
$$

Differentiate the objective function with respect to $w$ and equate to 0
$$
    \begin{align}
    x - 1 - \log(w) + 1 + \log(1-w) &= 0 \\
    \log(1-w) - \log(w) &= -x \\
    \log\left(\frac{1-w}{w}\right) &= -x \\
    \log\left(\frac{1}{w} - 1\right) &= -x  \\
    \frac{1}{w} &= 1 + \exp(-x) \\
     w^\star &= \frac{1}{1+\exp(-x)} \\
     \hat{p} &= w^\star
    \end{align}
$$


## Softmax
Consider the transformation
$$
\begin{align}
    \mathbf{p} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle
\end{align}
$$
where $$\mathbf{x} \in \mathbf{R^d}, \mathbf{w} \in \mathbf{S} \subset \left[0, 1\right]^d$$ such that $$\mathbf{S} = \left\{\mathbf{w} \mid \sum_{i}^d \mathbf{w}_i = 1\right\}$$, $$\mathbf{p} \in \left[0, 1\right]^d$$ and $$\sum_{i=1}^d \mathbf{p}_i = 1 $$.


$$\mathbf{p}$$ with negative entropy regularizer
$$
\begin{align}
    \mathbf{\hat{p}} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) 
\end{align}
$$
    Since  $$\mathbf{w} \in \mathbf{S}$$, add a Lagrange multiplier $$\lambda\left(\langle \mathbf{w}, \mathbf{1} \rangle - 1\right)$$ to the objective function. 
    $$
    \begin{align}
        \mathbf{\hat{p}} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) + \lambda\left(\langle \mathbf{w}, \mathbf{1} \rangle - 1\right) \\
        &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \sum_{i=1}^d \mathbf{w}_i \mathbf{x}_i  + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) + \lambda\left(\sum_{i=1} ^d \mathbf{w}_i - 1\right) 
    \end{align}
    $$
    Differentiate the objective function with respect to $$\mathbf{w}_i$$ and equate to 0
    $$
    \begin{align}
    \mathbf{x}_i + 1 + \log\left(\mathbf{w}_i\right) + \lambda &= 0 \\
    \mathbf{w}_i^\star &= \exp\left(-\mathbf{x}_i\right)\exp\left(-1 - \lambda\right) \\
    &= \frac{\exp\left(-\mathbf{x}_i\right)}{\exp\left(1 + \lambda\right)}
    \end{align}
    $$
Set $$ \lambda$$ such that $$\sum_{i}^d \mathbf{w}_i^\star = 1$$
$$
\begin{align}
     \mathbf{w}_i^\star &= \frac{\exp\left(-\mathbf{x}_i\right)}{\sum_{j=1}^d\exp\left(-\mathbf{x}_j\right)} \\
     \mathbf{\hat{p}} &= \mathbf{w}^\star
\end{align}
$$

Note that $\langle \mathbf{w}^\star, \mathbf{x} \rangle$ is smooth maximum whereas $$\langle \mathbf{w}, \begin{bmatrix}
1& 2& \ldots & d
\end{bmatrix} \rangle$$ for some $\mathbf{w} \in \mathbf{S}$ is softargmax [3].


## Discussion and Conclusion

Figure 1 shows the values of the objective function for different weights to the entropy term.

<a target="_blank" href="https://github.com/sigmoidandsoftmax/2025/blob/main/assets/img/2025-04-28-sigmoid-and-softmax/sigmoid_neg_entropy.png?raw=true"><img src="https://github.com/sigmoidandsoftmax/2025/blob/main/assets/img/2025-04-28-sigmoid-and-softmax/sigmoid_neg_entropy.png?raw=true" alt="Sigmoid with entropy" width="500"></a>


<a target="_blank" href="https://github.com/sigmoidandsoftmax/2025/blob/main/assets/img/2025-04-28-sigmoid-and-softmax/softmax_neg_entropy.png?raw=true"><img src="https://github.com/sigmoidandsoftmax/2025/blob/main/assets/img/2025-04-28-sigmoid-and-softmax/softmax_neg_entropy.png?raw=true" alt="Softmax with entropy" width="500"></a>


Figure 1. Effect of the entropy term in values of the objective function corresponding to sigmoid and softmax.

Figure 1 (top) shows the effect of entropy term in value of the objective function corresponding to sigmoid. The curve obtained by fixing the output probability and varying only the weight to the entropy term has a sigmoidal shape. On the other hand the curve obtained by fixing the weight to the entropy term and varying the output probability has a parabolic shape. Sigmoid is the function when applied to a scalar gives the probability that has minimum entropy and multiplication value with the input scalar.

Figure 1 (bottom) shows the effect of entropy term in value of the objective function corresponding to softmax. The curve obtained by fixing the output probabilities and varying only the weight to the entropy term has a sigmoidal shape. On the other hand the region obtained by fixing the weight to the entropy term and upper bounding value of the objective function and varying the output probabilities is an ellipse. Softmax is the function that when applied on a vector gives a probability vector that has minimum negative entropy and dot product value with the input vector.


## References 
1. Ian Goodfellow, Yoshua Bengio and Aaron Courville. Deep Learning. 2016.
2. [Luca Trevison. The ``Follow-the-Regularized-Leader'' algorithm. Topics in computer science and optimization (Fall 2019).](https://lucatrevisan.github.io/40391/lecture12.pdf)
3. Ross Goroshin, Michael Mathieu and Yann LeCun. Learning to Linearize Under Uncertainty. In Advances in Neural Information Processing Systems, 2015.
