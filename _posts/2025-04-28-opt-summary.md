---
layout: distill
title: Think Twice Before Claiming Your Optimization Algorithm Outperformance - Review and Beyond
description: In this blog, we revisit the convergence analysis of first-order algorithms in minimization and minimax optimization problems. Within the classical oracle model framework, we review the state-of-the-art upper and lower bound results in various settings, aiming to identify gaps in existing research. With the rapid development of applications like machine learning and operation research, we further identify some recent works that revised the classical settings of optimization algorithms study.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-opt-summary.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Framework
  - name: Notations
  - name: Summary of Results
  - name: What's next?
  - name: Conclusion

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

## Introduction

In this blog, we review the convergence rate of (stochastic) first-order methods in optimization. 

Regarding the **problem structure**, we specifically consider *minimization* problems above and *minimax* optimization problems of the following forms.

$$
\min_{x\in\mathcal{X}}\ f(x).
$$

$$
\min_{x\in\mathcal{X}}\ \left[f(x)\triangleq \max_{y\in\mathcal{Y}}\ g(x,y)\right].
$$

Based on the **stochasticity**, we divide our discussions into three cases:

- Deterministic (General) Optimization,

$$
\min_{x\in\mathcal{X}}\ f(x).
$$

- Finite-Sum Optimization

$$
\min_{x\in\mathcal{X}}\ f(x)\triangleq\frac{1}{n}\sum_{i=1}^n f_i(x).
$$

- (Purely) Stochastic Optimization

$$
\min_{x\in\mathcal{X}}\ f(x)\triangleq\mathbb{E}_{\xi\sim\mathcal{D}}[f(x;\xi)].
$$

A subtle while important difference between finite-sum and stochastic optimization problems lies in the ability to access to the overall objective function $f(x)$, i.e., stochastic optimization usually does not have access to $f(x)$ while the finite-sum problem has. As a result, algorithms that require access to $f(x)$, such as the classical stochastic variance reduced gradient (SVRG) algorithm<d-cite key="johnson2013accelerating"></d-cite>, are not directly applicable in the purely stochastic setting.

Based on the **convexity** of the objective, we will divide our discussions into various cases, including strongly-convex (SC), convex (C) and nonconvex cases (NC).
For minimax problems, based on convexity of $g(\cdot,y)$ for a given $y$ and the concavity of $g(x,\cdot)$ for a given $x$, we review results for strongly convex strongly concave (SC-SC), convex-concave (C-C),nonconvex strongly concave (NC-SC) and other combinations. 

### Literature

This notes aims to review state-of-the-art (SOTA) first-order optimization algorithms convergence results. In fact, there are several great works for comprehensive review of optimization algorithm from different perspectives. Besides many well-known textbook and course materials like the one from Stephen Boyd<d-cite key="boyd2024text"></d-cite><d-cite key="boyd2024video"></d-cite>, maybe one of the most impressive works is the blog post by Ruder<d-cite key="ruder2016overview"></d-cite>, which received more than 10k citations according to Google Scholar. This post reviewed algorithm design of gradient descent (GD), stochastic gradient descent (SGD) and their variants, especially those commonly used in machine learning community like AdaGrad<d-cite key="duchi2011adaptive"></d-cite> and Adam<d-cite key="kingma2014adam"></d-cite>. Several monographs reviewed optimization algorithms in various settings, e.g., <d-cite key="bubeck2015convex"></d-cite>, <d-cite key="bottou2018optimization"></d-cite>, <d-cite key="sun2019survey"></d-cite>, <d-cite key="dvurechensky2021first"></d-cite> and <d-cite key="garrigos2023handbook"></d-cite>; the page by Ju Sun<d-cite key="sun2021list"></d-cite> was a popular repository tracking research effort on achieving global optimality for nonconvex optimization<d-footnote>Latest update: Dec 11 2021.</d-footnote>. The review by Ruoyu Sun<d-cite key="sun2019optimization"></d-cite> further specified the survey of optimization algorithm study in the context of deep learning. A recent survey by Danilova et al.,<d-cite key="danilova2022recent"></d-cite> revisited algorithm design and complexity analysis specifically in nonconvex optimization, which to some extent is the closest one to our blog post. Our blog aims to serve as an easy accessible tool for optimizers to check the SOTA theoretical convergence rate from both upper and lower bounds perspectives in an easier manner.

---

## Framework: Oracle Complexity Model

We formally recall the definition of complexity. In particular, we stick to the classical **oracle complexity model**<d-cite key='nemirovskij1983problem'></d-cite>.
As the figure below suggest, generally this framework consists of the following components:
- *Fucntion class* $\mathcal{F}$, e.g., convex Lipschitz continuous function, and (nonconvex) Lipschitz smooth function.
- *Oracle class* $\mathbb{O}$, e.g., zeroth-order oracle (function value), first-order oracle (function gradient or subdifferential).
- *Algorithm class* $\mathcal{A}$, e.g., a common algorithm class studied in optimization literature is the *linear-span algorithm* interacting with an oracle $\mathbb{O}$, which encounters many well-known algorithms. This algorithm class is characterized as that 
if we let $(x^t)_t$ be the sequence history queries generated by the algorithm, for the next iterate, we have 

$$
x^{t+1}\in\mathrm{Span}\left\{x^0,\cdots,x^t;\mathbb{O}(f,x^0),\cdots,\mathbb{O}(f,x^t)\right\}.
$$

- *Complexity measure* $\mathcal{M}$, e.g., 
  - Optimality gap $f(x)-f(x^\star)$ where $x^\star$ is the global minimum.
  - Point distance $\|\|x-x^\star\|\|$.
  - Function stationarity $\|\|\nabla f(x)\|\|$, which is common in nonconvex optimization. 

{% include figure.html path="assets/img/2025-04-28-opt-summary/complexity_analysis.jpg" class="img-fluid" %}

<div class="caption">
    Oracle Complexity Framework (adapted from Prof. Yangyang Xu's Slides<d-cite key="xu2019slides"></d-cite>)
</div>


The efficiency of algorithms is quantified by the *oracle complexity*: for an algorithm $\mathtt{A}\in\mathcal{A}(\mathbb{O})$ interacting with an oracle $\mathbb{O}$, an instance $f\in\mathcal{F}$, and the corresponding measurement $\mathcal{M}$, we define

$$
T_{\epsilon}(f,\mathtt{A})\triangleq\inf\left\{T\in\mathbb{N}~|~\mathcal{M}(x^T)\leq\epsilon\right\}
$$

as the minimum number of oracle calls $\mathcal{A}$ makes to reach convergence. So given an algorithm $\mathtt{A}$, its upper complexity bound for solving one specific function class $\mathcal{F}$ is defined as

$$
\mathrm{UB}_\epsilon(\mathcal{F};\mathtt{A})
  \triangleq  
	\underset{f\in\mathcal{F}}{\sup}\ 
	T_{\epsilon}(f,\mathtt{A}),
$$

One of the mainstreams of optimization study is trying to design algorithms with better upper complexity bounds, corresponding to decreasing $\mathrm{UB}_\epsilon(\mathcal{F};\cdot)$ with their own algorihms.
On the other hand, another stream of study focuses on the performance limit in terms of the worst-case complexity, i.e., the lower complexity bound (LB) of a class of algorithm under certain settings, which can be written as:

$$
\mathrm{LB}_\epsilon(\mathcal{F},\mathcal{A},\mathbb{O})
  \triangleq
	\underset{\mathtt{A}\in{\mathcal{A}(\mathbb{O})}}{\inf}\   
	\underset{f\in\mathcal{F}}{\sup}\ 
	T_{\epsilon}(f,\mathtt{A}),
$$

{% include figure.html path="assets/img/2025-04-28-opt-summary/upper_lower.png" class="img-fluid" %}

<div class="caption">
    Illustration of Upper and Lower Complexity Bounds
</div>

As the figure above suggests, the optimization community keep facilitating the understanding of algorithm complexity from both upper and lower complexity directions.
An ultimate goal in optimization algorithm complexity study is to find the *optimal algorithm* $\mathtt{A}^\star$ in a given setting, which means its upper bound matches with the lower bound of the algorithm class under the function setting, 
i.e.,

$$
\mathrm{UB}_\epsilon(\mathcal{F};\mathtt{A}^\star)\asymp\mathrm{LB}_\epsilon(\mathcal{F},\mathcal{A},\mathbb{O}).
$$

In this notes, we will focus on **first-order algorithms** in various optimization problem settings, trying to summarize the state-of-the-art (SOTA) UB and LB results, aiming to identify the gaps in existing reseach, and develop new trends. 

---

## Notations

Besides the structure above, optimization algorithm convergence literature often require some other regularity conditions like 
Lipschitz smoothness, Lipschitz continuity, unbiased gradient estimator, bounded variance. Here we assume readers are already familiar with such context,
interested readers may refer to these nice handbooks <d-cite key="garrigos2023handbook"></d-cite> and <d-cite key="danilova2022recent"></d-cite> for detailed definitions.

For convenience, we summarize some of the notations commonly used in tables below.
- SC / C / NC / WC: strongly convex, convex, nonconvex, weakly-convex.
- FS: finite-sum optimization.
- Stoc: stochastic optimization
- $L$-S: The objective function is $L$-Lipschitz smooth (or jointly Lipschitz smooth in minimax optimization).
- $L$-Lip Cont.: $L$-Lipschitz continuous.
- $L$-IS / AS / SS<d-footnote>For clarification, $L$-IS means in finite-sum problems, each component function $f_i$ itself is $L$-smooth, for the definition of $L$-AS, please refer to the definition of "mean-squared smoothness" in <d-cite key="arjevani2023lower"></d-cite>, and $L$-SS means the summation $f$ is $L$-smooth while each component $f_i$ may not be Lipschitz smooth. Clearly IS is stronger than AS, AS is stronger than SS.</d-footnote>: $L$-Lipschitz individual / averaged / summation smoothness.
- NS: Nonsmooth
- PL: Polyak-Łojasiewicz Condition
- $\mathcal{O},\tilde{\mathcal{O}},\Omega$: For nonnegative functions $f(x)$ and $g(x)$, we say $f=\mathcal{O}(g)$ if $f(x)\leq cg(x)$ for some $c>0$, and further write $f=\tilde{\mathcal{O}}(g)$ to omit poly-logarithmic terms on some constants, and $f=\Omega(g)$ if $f(x)\geq cg(x)$.
- $\Delta$, $D$: The initial function value gap $\Delta\triangleq(x_0)-f(x^\star)$, and the initial point distance $D\triangleq\|\|x_0-x^\star\|\|$.
- Optimality gap: the function value gap $f(x) - f^\star$.
- Stationarity: the function gradient norm $\|\| \nabla f(x) \|\|$.
- Near-stationarity: the gradient norm $\|\| \nabla f_\lambda(x) \|\|$, where $f_\lambda$ is the Moreau envelope of the original function $f$.
- Duality Gap (for minimax optimization): the primal-dual gap of a given point $(\hat{x},\hat{y})$, defined as $G_f(\hat{x},\hat{y})\triangleq\max_y f(\hat{x},y)-\min_x f(x,\hat{y})$

- Primal Stationarity (for minimax optimization)<d-footnote>Here we always define the convergence in terms that the **norm** is driven be smaller than $\epsilon$, note that some works may deduce the final result measured by the square, e.g., $\|\| \nabla f(x) \|\|^2$ or $\|\| \nabla f_\lambda(x) \|\|^2$. </d-footnote>: the primal function gradient norm $\|\| \nabla \Phi(x) \|\|$, where $\Phi(x)\triangleq\max_{y\in\mathcal{Y}}f(x,y)$ is the primal function. It is different from the function stationarity in terms of the original objective function $f$.

---

## Summary of Results

As mentioned above, we categorize the discussion based on the problem, stochasticity and function structure, for convenience of presentation, we divide the presentation into the following cases:

**Minimization Problems**:
1. Deterministic optimization
2. Finite-sum and stochastic optimization

**Minimax Problems**, based on the convexity combination of each component, we consider the following cases:
1. SC-SC/SC-C/C-C deterministic minimax
2. SC-SC/SC-C/C-C finite-sum and stochastic minimax optimization
3. NC-SC/NC-C deterministic minimax optimization
4. NC-SC/NC-C finite-sum and stochastic minimax optimization

We present the lower and upper bound results in tables below<d-footnote>Given the extensive body of literature on this topic, it is possible that some relevant references may have been inadvertently overlooked, or some results we cited are not fully correct. We welcome any comments or questions and are happy to discuss.</d-footnote>. Note that:

- Here we use $\checkmark$ to denote that the upper and lower bounds already match in this setting, otherwise we use $\times$ or directly present existing best result.
- Here "(within logarithmic)" means the upper bound matches the lower bound within logarithmic factors which depends on $\epsilon$, generally it can be regarded as a neglectable term, so we still denote it as $\checkmark$.
- For some cases, we denote the LB as "Unknown" if there is not a specific (nontrivial) lower bound built for this case.

<div class="l-page" markdown="1">

### Case 1-1: Deterministic Minimization

| Problem Type               | Measure                   | Lower Bound            | Upper Bound      | Reference (LB)      | Reference (UB)<d-footnote>Note that here possibly we may not choose the most original work which proposed the results, rather we may select the one which may come with a clearer presentation. Readers are encouraged to check the reference therein for the original works.</d-footnote> |
|----------------------------|---------------------------|------------------------|-------------|--------------------------------|-----------|
| $L$-Smooth Convex          | Optimality gap           | $\Omega \left( \sqrt{L \epsilon^{-1}} \right)$                | $\checkmark$ | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 2.1.7       | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 2.2.2 |
| $L$-Smooth $\mu$-SC        | Optimality gap           | $\Omega \left( \sqrt{\kappa} \log \frac{1}{\epsilon} \right)$ | $\checkmark$ | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 2.1.13      | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 2.2.2 |
| NS $L$-Lip Cont. Convex       | Optimality gap           | $\Omega (L^2 \epsilon^{-2})$                               | $\checkmark$ | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 3.2.1       | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 3.2.2 |
| NS $L$-Lip Cont. $\mu$-SC     | Optimality gap           | $\Omega (L^2 (\mu \epsilon)^{-1})$                         | $\checkmark$ | <d-cite key="nesterov2018lectures"></d-cite>, Theorem 3.2.5       | <d-cite key="bubeck2015convex"></d-cite>, Theorem 3.9<d-footnote>The algorithm design therein requires projection.</d-footnote> |
| $L$-Smooth Convex (function case)          | Stationarity    | $\Omega \left( \sqrt{\Delta L }\epsilon^{-1} \right)$  | $\checkmark$ (within logarithmic) | <d-cite key="carmon2021lower"></d-cite>, Theorem 1                | <d-cite key="carmon2021lower"></d-cite>, Appendix A.1 |
| $L$-Smooth Convex (point case)          | Stationarity    | $\Omega \left( \sqrt{D L \epsilon^{-1}} \right)$          | $\checkmark$ (within logarithmic) | <d-cite key="carmon2021lower"></d-cite>, Theorem 1                | <d-cite key="nesterov2012make"></d-cite> |
| $L$-Smooth NC              | Stationarity    | $\Omega (\Delta L \epsilon^{-2})$                                      | $\checkmark$ | <d-cite key="carmon2020lower"></d-cite>, Theorem 1                | <d-cite key="carmon2021lower"></d-cite>, Theorem 10.15 |
| NS $L$-Lip Cont. $\rho$-WC    | Near-stationarity        | Unknown                                                    | $\mathcal{O}(\epsilon^{-4})$  | /                                                | <d-cite key="davis2018stochastic"></d-cite>, Corollary 2.2 |
| $L$-Smooth $\mu$-PL     | Optimality gap | $\Omega \left( \kappa \log \frac{1}{\epsilon} \right)$                     | $\checkmark$      | <d-cite key="yue2023lower"></d-cite>, Theorem 3              | <d-cite key="karimi2016linear"></d-cite>, Theorem 1 |

</div>

### Case 1-2: Finite-sum and Stochastic Optimization

| Problem Type            | Measure | Lower Bound                                                | Upper Bound                               | Reference (LB-UB)                                       |
|-------------------------|---------|---------------------------------------------------|----------------------------------|-----------------------------------------------------------|
| FS $L$-IS $\mu$-SC      | Optimality gap        | $\Omega \left( n + \sqrt{\kappa n} \log \frac{1}{\epsilon} \right)$ | $\times$          | [<d-cite key="woodworth2016tight"></d-cite>, Thm 8] [<d-cite key="defazio2016simple"></d-cite>, Cor 6], [<d-cite key="allen2018katyusha"></d-cite>, Cor 2.1]                          |
| FS $L$-IS C             | Optimality gap        | $\Omega \left( n + D \sqrt{n L \epsilon^{-1}} \right)$         | $\times$          | [<d-cite key="woodworth2016tight"></d-cite>, Thm 7] [<d-cite key="allen2018katyusha"></d-cite>, Cor 3.7] |
| FS $L$-AS $\mu$-SC      | Optimality gap        | $\Omega \left( n + n^{\frac{3}{4}} \sqrt{\kappa} \log \frac{\Delta}{\epsilon} \right)$ | $\checkmark$                     | [<d-cite key="xie2019general"></d-cite>, Thm 3.5], [<d-cite key="allen2018katyusha"></d-cite>, Sec 5]                                 |
| FS $L$-AS C             | Optimality gap        | $\Omega \left( n + n^{\frac{3}{4}} D \sqrt{L \epsilon^{-1}} \right)$ | $\checkmark$                     | [<d-cite key="zhou2019lower"></d-cite>, Thm 4.2], [<d-cite key="allen2018katyusha"></d-cite>, Sec 5]                                 |
| FS $L$-IS NC            | Stationarity        | $\Omega \left( \Delta L \epsilon^{-2} \right)$               | $\times$                          | [<d-cite key="zhou2019lower"></d-cite>, Thm 4.7], [<d-cite key="wang2019spiderboost"></d-cite>, Thm 1] |
| FS $L$-AS NC            | Stationarity       | $\Omega \left( \sqrt{n \Delta L \epsilon^{-2}} \right)$       | $\checkmark$                     | [<d-cite key="zhou2019lower"></d-cite>, Thm 4.5], [<d-cite key="fang2018spider"></d-cite>, Thm 2, 3]|
|                         |         |                                                     |                                  |                                                           |
| Stoc $L$-$S$ $\mu$-SC   | Stationarity        | $\Omega (\epsilon^{-1})$                            | $\checkmark$                     | [<d-cite key="rakhlin2012making"></d-cite>, Thm 2], [<d-cite key="ghadimi2012optimal"></d-cite>, Prop 9]                        |
| Stoc $L$-$S$ C          | Stationarity        | $\Omega (\epsilon^{-2})$                            | $\checkmark$                     | [<d-cite key="woodworth2018graph"></d-cite>, Thm 1], [<d-cite key="lan2012optimal"></d-cite>, Thm 1]          |
| Stoc NS $\mu$-SC        | Stationarity        | $\Omega (\epsilon^{-2})$                            | $\checkmark$                     | [<d-cite key="agarwal2009information"></d-cite>, Thm 2], [<d-cite key="nemirovski2009robust"></d-cite>, Sec 2.1]                          |
| Stoc NS C               | Stationarity        | $\Omega (\epsilon^{-2})$                            | $\checkmark$                     | [<d-cite key="agarwal2009information"></d-cite>, Thm 1], [<d-cite key="nemirovski2009robust"></d-cite>, Sec 2.2]                        |
| Stoc $L$-$S$ $\mu$-SC   | Stationarity        | Unknown                  | $\mathcal{O} \left( \sqrt{\frac{LD}{\epsilon}} \log k + \frac{\sigma^2}{\epsilon^2} \log^3 k \right)$                     | [<d-cite key="foster2019complexity"></d-cite>, Thm 1] |
| Stoc $L$-$S$ C   | Stationarity        | $\Omega \left( \frac{LD}{\epsilon} + \frac{\sigma^2}{\epsilon^2} \log \frac{LD}{\epsilon} \right)$ | $\mathcal{O} \left( \sqrt{\frac{LD}{\epsilon}} \log k + \frac{\sigma^2}{\epsilon^2} \log^3 k \right)$       | [<d-cite key="foster2019complexity"></d-cite>, Thm 1, Cor 1] |
| Stoc L-SS NC   | Stationarity        | $\Omega \left( \Delta \sigma \epsilon^{-4} \right)$         | $\checkmark$                     | [<d-cite key="arjevani2023lower"></d-cite>, Thm 1], [<d-cite key="ghadimi2013stochastic"></d-cite>, Cor 2.2]                                           |
| Stoc L-AS NC            | Stationarity        | $\Omega \left( \Delta \sigma^2 + 3 \sigma \epsilon^{-2} \right)$ | $\checkmark$                     | [<d-cite key="arjevani2023lower"></d-cite>, Thm 2], [<d-cite key="fang2018spider"></d-cite>, Thm 1]                          |
| NS $L$-Lip $\rho$-WC    | Near-stationarity        | Unknown     | $\mathcal{O} (\epsilon^{-4})$ | [<d-cite key="davis2018stochastic"></d-cite>, Thm 2.1]              |

**Remark:**

1. Here $n$ corresponds to the number of component functions $f_i$, and $\kappa\triangleq L/\mu$ is the condition number, $\sigma^2$ corresponds to the variance of gradient estimator.
2. "Thm" and "Cor" stand for "Theorem" and "Corollary" respectively.
 
### Case 2-1: SC-SC/SC-C/C-C Deterministic Minimax Optimization

| Problem Type          | Measure | Lower Bound                                             | Upper Bound                           | Reference (LB-UB)                                      |
|-----------------------|---------|-------------------------------------------------|------------------------------|----------------------------------------------------------|
| C-C, bilinear, NS     | Duality Gap        | $\Omega(L / \epsilon)$                          | $\checkmark$                 | [<d-cite key="ouyang2021lower"></d-cite>, Thm 9], [<d-cite key="chambolle2011first"></d-cite>, Thm 1]                                            |
| C-C, general          | Duality Gap        | $\Omega(L D^2 / \epsilon)$                  | $\checkmark$                 | [<d-cite key="xie2020lower"></d-cite>, Thm 3], [<d-cite key="nemirovski2004prox"></d-cite>, Thm 4.1]  |
| SC-C, bilinear, NS    | Duality Gap        | $\Omega(\sqrt{\kappa_x} / \epsilon)$            | $\mathcal{O}(\kappa_x^2 / \sqrt{\epsilon})$      | [<d-cite key="ouyang2021lower"></d-cite>, Thm 10], [<d-cite key="chambolle2011first"></d-cite>, Thm 2]  |
| SC-C, general         | Duality Gap        | $\Omega(D \sqrt{L \kappa_x} / \epsilon)$      | $\tilde{\mathcal{O}}(D \sqrt{L \kappa_x} / \epsilon)$      | [<d-cite key="xie2020lower"></d-cite>, Thm 2], [<d-cite key="yang2020catalyst"></d-cite>, Sec 3.2]  |
| C-SC, bilinear        | Duality Gap        | Unknown                                               | $\mathcal{O}(\log \frac{1}{\epsilon})$ | [<d-cite key="du2019linear"></d-cite>, Thm 3.1] |
| SC-SC, bilinear       | Duality Gap        | $\Omega(\sqrt{\kappa_x \kappa_y} \log \frac{1}{\epsilon})$ | $\checkmark$               | [<d-cite key="zhang2022lower"></d-cite>, Thm 3.5], [<d-cite key="chambolle2016ergodic"></d-cite>, Thm 5]                                      |
| SC-SC, general        | Duality Gap        | $\Omega(\sqrt{\kappa_x \kappa_y} \log \frac{1}{\epsilon})$ | $\times$    | [<d-cite key="zhang2022lower"></d-cite>, Thm 4.5], [<d-cite key="wang2020improved"></d-cite>, Thm 3]                                            |
| PL-PL                 | Duality Gap        | Unknown                                            | $\mathcal{O}(\kappa^3\log \frac{1}{\epsilon})$ | [<d-cite key="yang2020global"></d-cite>] |


**Remark:**

1. Here $\kappa_x$ and $\kappa_y$ corresponds to condition numbers on $x$ and $y$ components, respectively.

### Case 2-2: SC-SC/SC-C/C-C Finite-sum and Stochastic Minimax Optimization

| Problem Type         | Measure | LB                                         | UB                            | Reference-LB      | Reference-UB                                   |
|----------------------|---------|---------------------------------------------|-------------------------------|-------------------|------------------------------------------|
| C-C, FS         | Duality Gap | $\Omega(n + L / \epsilon)$                 | $\mathcal{O}(\sqrt{n}/\epsilon)$                      | [<d-cite key="xie2020lower"></d-cite>, Thm 3]     | [<d-cite key="yazdandoost2023stochastic"></d-cite>, Cor 2.1]        |
| C-C, Stoc, SS        | Duality Gap | $\Omega(\epsilon^{-2})$                    | $\checkmark$                  | /                                             | [<d-cite key="juditsky2011solving"></d-cite>, Cor 1]      |
| C-C, Stoc, NS        | Duality Gap | $\Omega(\epsilon^{-2})$                    | $\checkmark$                  | /                                             | [<d-cite key="nemirovski2009robust"></d-cite>, Lemma 3.1]      |
| SC-C, FS        | Duality Gap | $\Omega\left(n + \sqrt{n L / \epsilon}\right)$ | $\tilde{\mathcal{O}}(\sqrt{n L / \epsilon})$   | /                                                  | [<d-cite key="yang2020catalyst"></d-cite>, Sec 3.2]      |
| SC-SC, FS       | Duality Gap | $\Omega\left((n + \kappa) \log \frac{1}{\epsilon}\right)$ | $\checkmark$ | [<d-cite key="xie2020lower"></d-cite>, Thm 1]      | [<d-cite key="palaniappan2016stochastic"></d-cite>, Thm 1]      |
| SC-SC, Stoc, SS      | Duality Gap | $\Omega(\epsilon^{-1})$                    | $\checkmark$                  | /                                            | [<d-cite key="hsieh2019convergence"></d-cite>, Thm 5]       |
| SC-SC, Stoc, NS      | Duality Gap | $\Omega(\epsilon^{-1})$                    | $\checkmark$                  | /                                            | [<d-cite key="yan2020optimal"></d-cite>]       |
| PL-PL, Stoc      | Duality Gap | Unknown                    | $\mathcal{O}(\kappa^5\epsilon^{-1})$                  | /                                            | [<d-cite key="yang2020global"></d-cite>]       |

### Case 2-3: NC-SC/NC-C Deterministic Minimax Optimization

| Type               | Measure | LB                                          | UB                               | Reference-LB      | Reference-UB                                   |
|--------------------|---------|---------------------------------------------|----------------------------------|-------------------|------------------------------------------------|
| NC-SC, Deter       | Primal Stationarity | $\Omega(\sqrt{\kappa}\Delta \mathcal{L} \epsilon^{-2})$  | $\checkmark$ | [<d-cite key="zhang2021complexity"></d-cite>, Thm 3.1] | [<d-cite key="zhang2021complexity"></d-cite>, Thm 4.1] |
| NC-C, Deter        | Near-Stationarity | Unknown  | $\mathcal{O}(\Delta L^2 \epsilon^{-3} \log^2 \frac{1}{\epsilon})$               |   /      | [<d-cite key="lin2020near"></d-cite>, Cor A.8] |
| WC-C, Deter        | Near-Stationarity | Unknown | $\mathcal{O}(\epsilon^{-6})$                       |  /       | [<d-cite key="boct2023alternating"></d-cite>, Thm 3.7]         |
| NC-PL, Deter       | Primal Stationarity | Unknown | $\Omega(\kappa^2\epsilon^{-4})$                   | /        | [<d-cite key="yang2022faster"></d-cite>]         |

**Remark:**

1. Some other works also studied the above problems in terms of the fucnction stationarity (i.e., the gradient norm of $f$, rather than it primal), e.g., <d-cite key="lin2020near"></d-cite><d-cite key="xu2023unified"></d-cite>. As discussed in <d-cite key="yang2022faster"></d-cite>, it has been shown that function stationarity and primal stationarity are transferable with a mild efforts, here we do not present the results specifically.

### Case 2-4: NC-SC/NC-C Finite-sum and Stochastic Minimax Optimization

| Type             | Measure | LB                                      | UB                                      | Reference (UB & LB)                                                                 |
|------------------|---------|-----------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------|
| NC-SC, FS, AS    | Primal Stationarity        | $\Omega\left(\sqrt{n\kappa}\Delta L\epsilon^{-2}\right)$ | $\mathcal{O}\left(\sqrt{n}\kappa^2 L\Delta\epsilon^{-2}\right)$ | [<d-cite key="zhang2021complexity"></d-cite>]  |          
| NC-C, FS, IS     | Near-stationarity        | Unknown | $\mathcal{O}\left(n^{3/4}L^2D\Delta\epsilon^{-3}\right)$ | [<d-cite key="yang2020catalyst"></d-cite>]                               |
| NC-SC, Stoc, SS  | Primal Stationarity        | $\Omega\left(\kappa^{1/3}\Delta L\epsilon^{-4}\right)$ | $\mathcal{O}\left(\kappa\Delta L\epsilon^{-4}\right)$ | [<d-cite key="zhang2022sapd+"></d-cite>, <d-cite key="li2021complexity"></d-cite>]                                                 |
| NC-SC, Stoc, IS  | Primal Stationarity        | Unknown | $\mathcal{O}\left(\kappa^2\Delta L\epsilon^{-3}\right)$ | [<d-cite key="zhang2022sapd+"></d-cite>] |
| NC-C, Stoc, SS   | Near-stationarity        | Unknown | $\mathcal{O}\left(L^3\epsilon^{-6}\right)$              | [<d-cite key="zhang2022sapd+"></d-cite>]            |

---

## What is next?
The section above summarize the upper and lower bounds of the oracle complexity for finding an $\epsilon$-optimal solution or $\epsilon$-stationary points for minimization and minimax problems. Clearly this is not the end of the story, there are more and more optimization problems arising from various applications like machine learning and operation research<d-cite key="bottou2018optimization"></d-cite>, which come with more involved problem structure and complicated landscape characteristics; also sometimes we find it harder to explain algorithm behavior in practice using existing theory, e.g., <d-cite key="defazio2019ineffectiveness"></d-cite> shows that variance reduction may be ineffective on accelerating the training of deep learning models, which contrast the classical convergence theory. Here we discuss what could be potential interesting next steps. 

### Richer Problem Structure

In this notes, we only discussed minimization and minimax problems, while there are also many other important optimization problems with different structure, for example:

* Bilevel Optimization 

  $$
  \min_{x \in \mathcal{X}} \Phi(x) = F(x, y^\star(x))  \quad \text{where} \quad y^\star(x) = \underset{y \in \mathcal{Y}}{\arg\min} \, G(x, y),
  $$

  Bilevel optimization covers minimax optimization as a special case. Over the past seven years, bilevel optimization has become increasingly popular due to its applications in machine learning. In particular, researchers have revisited first-order methods for solving (stochastic) bilevel optimization problems. Starting from <d-cite key="ghadimi2018approximation"></d-cite>, which investigates double-loop methods for solving bilevel optimization, <d-cite key="hong2020two"></d-cite> initiated the development of single-loop, single-timescale methods for stochastic bilevel optimization. This line of research leads to a simple single-timescale algorithm <d-cite key="chen2021tighter"></d-cite> and multiple variance reduction techniques to achieve single-loop <d-cite key="guo2021randomized"></d-cite><d-cite key="khanduri2021near"></d-cite>. 
  
  Subsequent developments have focused on developing fully first-order methods for solving bilevel optimization <d-cite key="kwon2023fully"></d-cite>, achieving global optimality <d-cite key="xiao2024unlocking"></d-cite>, addressing contextual/multiple lower-level problems <d-cite key="hu2024contextual"></d-cite><d-cite key="guo2021randomized"></d-cite>, handling constrained lower-level problems <d-cite key="jiang2024barrier"></d-cite>, and bilevel reinforcement learning <d-cite key="chen2022adaptive"></d-cite><d-cite key="chakraborty2024parl"></d-cite><d-cite key="thoma2024contextual"></d-cite> for model design and reinforcement learning with human feedback.
  Several questions remain open and are interesting to investigate:
  1. How to handle general lower-level problems with coupling constraints. 
  2. How to accelerate fully first-order methods to match the optimal complexity bounds.
  3. How to establish non-asymptotic convergence guarantees for bilevel problems with convex lower levels.

Also there appeared several other optimization problems with different formulations arising from practice, e.g.,

* Compositional Stochastic Optimization<d-cite key="wang2017stochastic"></d-cite>

$$
\min_{x \in \mathcal{X}} F(x) =\mathbb{E}_{\xi}\left[f\left(\mathbb{E}_{\eta}\left[g(x;\eta)\right];\xi\right)\right].
$$

* Conditional Stochastic Optimization <d-cite key="hu2020sample"></d-cite>

  $$\min_{x \in \mathcal{X}} F(x) =\mathbb{E}_{\xi}\left[f\left(\mathbb{E}_{\eta\mid\xi}\left[g(x;\eta,\xi)\right];\xi\right)\right].$$

  Conditional stochastic optimization differs from composition stochastic optimization mainly in the conditional expectation inside $f$. This requires one to sample from the conditional distribution of $\eta$ for any given $\xi$, while compositional optimization can sample from the marginal distribution of $\eta$. 

* Performative Prediction (or Decision-Dependent Stochastic Optimization)<d-cite key="perdomo2020performative"></d-cite><d-cite key="drusvyatskiy2023stochastic"></d-cite>

$$
\min_{x\in\mathcal{X}}\ f(x)\triangleq\mathbb{E}_{\xi\sim\mathcal{D}(x)}[f(x;\xi)].
$$

* Contextual Stochastic Optimization<d-cite key="bertsimas2020predictive"></d-cite><d-cite key="sadana2024survey"></d-cite>

$$
\min_{x\in\mathcal{X}}\ f(x;z)\triangleq\mathbb{E}_{\xi\sim\mathcal{D}}[f(x;\xi)~|~Z=z]
$$

* Distributionally robust optimization<d-cite key="kuhn2024distributionally"></d-cite>

  $$ \min_{x\in\mathcal{X}}\sup_{\mathcal{D}\in U_r(Q)} \triangleq\mathbb{E}_{\xi\sim \mathcal{D}}[f(x;\xi)],$$

  where $U_r(Q)$ refers to an uncertainty set containing a family of distributions around a nominal distribution $Q$ measured by some distance between distributions with radius $r$.

### Landscape Analysis
  
  Since most deep learning problems are nonconvex, a vast amount of literature focus on finding a (generalized) stationary point of the original optimization problem, but the practice often showed that one could find global optimality for various structured nonconvex problems efficiently. In fact there is a line of research tailored for the global landscape of structured nonconvex optimization, for example in neural network training, the interpolation condition holds for some overparameterized neural networks<d-cite key="sun2020global"></d-cite>. 
  
  Regarding such a mismatch between theory and practice, one reason may be the coarse assumptions the community applied in the theoretical analysis, which cannot effectively characterize the landscape of objectives. Here we briefly summarize a few structures arising in recent works, which try to mix the gap between practice and theory:

  - *Hidden convexity* says that the original nonconvex optimization problem might admit a convex reformulation via a variable change. It appears in operations research <d-cite key="chen2024efficient"></d-cite><d-cite key="chen2023network"></d-cite>, reinforcement learning <d-cite key="zhang2020variational"></d-cite>, control <d-cite key="anderson2019system"></d-cite>. Despite that the concrete transformation function is unknown, one could still solve the problem to global optimality efficiently <d-cite key="fatkhullin2023stochastic"></d-cite> with $\mathcal{O}(\epsilon^{-3})$ complexity for hidden convex case and $\mathcal{O}(\epsilon^{-1})$ complexity for hidden strongly convex case. In the hidden convex case, one could further achieve $\mathcal{O}(\epsilon^{-2})$ complexity in the hidden convex case via mirror stochastic gradient <d-cite key="chen2024efficient"></d-cite> or variance reduction <d-cite key="zhang2021convergence"></d-cite>.
  
  - Another stream considers *Polyak-Łojasiewicz* (PL) or *Kurdyka-Łojasiewicz* (KL) type of conditions, or other gradient dominance conditions <d-cite key="karimi2016linear"></d-cite>. Such conditions imply that the (generalized) gradient norm could upper bound the optimality gap, implying that any (generalized) stationary point are also global optimal. However, establishing hidden convexity, PL, or KL condition is usually done in a case-by-case manner and could be challenging. See <d-cite key="chen2024landscape"></d-cite> for some examples about KL conditions in finite horizon MDP with general state and action and its applications in operations and control. See <d-cite key="fatkhullin2022sharp"></d-cite> and reference therein for convergence rate analysis under KL conditions.
  
  - With numerical experiments disclosing structures in objective functions, some works proposed new assumptions which drive the algorithm design and corresponding theoretical analysis, which in turn reveals acceleration in empirical findings. For example, <d-cite key="zhang2019gradient"></d-cite> introduced the *relaxed smoothness assumption* (or $(L_0, L_1)$-smoothness) inspired by empirical observations on deep neural networks, and proposed a clipping-based first-order algorithm which enjoys both theoretical and practical outperformance.
  
    Another noteworthy work is <d-cite key="zhang2020adaptive"></d-cite>, which verified the ubiquity of *heavy-tailed noise* in stochastic gradients in neural network training practices, such evidence drove them to revise SGD and incorporate strategies like clipping in the algorithm design, which also outperformed in numerical experiments. The above two works, along with their more practical assumptions, inspired many follow-up works, evidented by their high citations according to Google Scholar<d-cite key="zhang2019citation"></d-cite><d-cite key="zhang2020citation"></d-cite>.

### Unified Lower Bounds
For lower bounds, we adapt the so-called optimization-based lower bounds proved via *zero-chain arguments* <d-cite key="nesterov2013introductory"></d-cite>. The narrative of such lower bounds admits the following form: For any given accuracy $\epsilon>0$, there exists a hard instance $f:\mathbb{R}^{d_\epsilon}\rightarrow\mathbb{R}$ in the function class $\mathcal{F}$, where the dimension $d_\epsilon$ depends on the accuracy $\epsilon>0$, such that it takes at least $\mathrm{poly}(\epsilon^{-1})$ number of oracles to find an $\epsilon$-optimal solution or $\epsilon$-stationary point. Note that the dimension $d_\epsilon$ depends on the accuracy, particularly $d_\epsilon$ increases as $\epsilon$ decreases. For a function class with a given dimension $d$, which is independent of the accuracy, the dependence on $\epsilon$ becomes loose especially when $d$ is small. In other words, for a $d$-dimension function classes and a given accuracy $\epsilon>0$, the upper bounds on the complexity of first-order methods given in the tables still hold, yet the lower bounds becomes loose, which could lead to a mismatch between upper and lower bounds. This leads to a fundamental question: how to prove lower bounds of first-order methods for any given $d$-dimensional function classes. 

Such a question has been partially addressed for stochastic optimization using *information theoretic-based lower bounds*<d-cite key="agarwal2009information"></d-cite> and for deterministic one-dimensional optimization<d-cite key="chewi2023complexity"></d-cite>. For stochastic convex optimization, <d-cite key="agarwal2009information"></d-cite> shows that  for any given $d$-dimensional Lipschitz continuous convex function class and any $\epsilon>0$, it takes at least $\mathcal{O}(\sqrt{d} \epsilon^{-2})$ number of gradient oracles to find an $\epsilon$-optimal solution. Such information-theoretic lower bounds admits explicit dependence on the problem dimension. In addition, even ignoring dimension dependence, the dependence on the accuracy matches the upper bounds of stochastic gradient descent.  Thus such a lower bound addresses the aforementioned question for stochastic convex optimization. Yet, it raises another interesting observation, i.e., the obtained lower bounds  $\mathcal{O}(d\epsilon^{-2})$<d-cite key="agarwal2009information"></d-cite> is larger than the upper bounds $\mathcal{O}(\epsilon^{-2})$<d-cite key="nemirovski2009robust"></d-cite>. This is of course not a conflict as the two papers make different assumptions. Yet, it would be interesting to ask, if first-order methods such as mirror descent are really dimension independent or is it the case that existing optimization literature is treating some parameters that could be dimension dependent as dimension independent ones. 
  

### Beyond Classical Oracle Model
  Regarding the oracle complexity model, because it mainly focuses on *worst-case instances* in the function class which may be far from *practical instances*, possibly the derived complexities can be such conservative and vacuous that they may not match the practice well, as the figure below illustrated.

  {% include figure.html path="assets/img/2025-04-28-opt-summary/practice_gap.png" class="img-fluid" %}

  <div class="caption">
      Gap Between General Worst-Case Complexity and Instance-Level Complexity Analysis (adapted from <d-cite key="zhang2022beyond"></d-cite>)
  </div>

  Recently there appeared some works which go beyond the classical oracle model on evaluating the optimization algorithm efficiency, for example:

  - In <d-cite key="pedregosa2020acceleration"></d-cite> and <d-cite key="paquette2021sgd"></d-cite>, they considered *average-case complexity* in optimization algorithm analysis, this topic is well established in theoretical computer science. Such pattern considers the convergence behavior in the average sense rather than the classical worst-case flavor, and it further specified the objective formulation and data distribution, which results in more refined complexity bounds than the common worst-case complexities, while at the expense of more complicated analysis. The study of average-case complexity in the context of optimization algorithms is still less mature and there are still many open questions.
  - On the other hand, with the development of higher-order algorithms, some recent works<d-cite key="doikov2023second"></d-cite> further considered the *arithmetic complexity* in optimization by incorporating the computational cost of each oracle into accout<d-footnote>In Nesterov's book<d-cite key="nesterov2018lectures"></d-cite>, these two are also called "analytical complexity" and "arithmetical complexity".</d-footnote>
  - Also in distributed optimization (or federated learning) literature, the communication cost is one of the main bottlenecks compared to computation<d-cite key="konevcny2016federated"></d-cite><d-cite key="mcmahan2017communication"></d-cite><d-cite key="karimireddy2020scaffold"></d-cite><d-cite key="kairouz2021advances"></d-cite>, so many works modified the oracle framework a bit turn to study the complexity bound in terms of *communication cost* (or communication oracle) rather than the computation efforts, such change also motivated the fruitful study of *local algorithms*<d-cite key="stich2019local"></d-cite><d-cite key="mishchenko2022proxskip"></d-cite>, which try to skip unnecessary communications while still attain the convergence guarantees. 
  - Recently another series of recent works<d-cite key="grimmer2024provably"></d-cite><d-cite key="altschuler2023acceleration1"></d-cite><d-cite key="altschuler2023acceleration2"></d-cite> consider *long stepsize* by incorporating a craft stepsize schedule into first-order methods and achieve a faster convergence rate, which is quite counterintuitive and may be of interests to the community. At the same time, as indicated in <d-cite key="kornowski2024open"></d-cite><d-cite key="altschuler2023acceleration1"></d-cite>, such theoretical outperformance generally only works for some specific iteration numbers, while in lack of guarantees of *anytime convergence*, also the extension of the study beyond the deterministic and convex case is still an open problem. 

---

## Conclusion

In this post, we review SOTA complexity upper and lower bounds of first-order algorithms in optimization tailored for minimization and minimax regimes with various settings, the summary identified gaps in existing research, which shed light on the open questions regarding accelerated algorithm design and performance limit investigation. Under the oracle framework, people should be careful when claiming one algorithm is better than the others, and double check whether the comparison is fair, in terms of the settings like function class, oracle information and algorithm class definition.

Regarding the rapid development and interdisciplinary applications in areas like machine learning and operation research, we revisited several recent works which go beyond the classical research flow in optimization community, these works advocate a paradigm shift in research: besides an elegant and unified theory trying to cover all cases, sometimes we should also try to avoid the "Maslow's hammer", focus on the detailed applications first, identify their unique structure, and correspondently design algorithms tailored for these problems, which in turn will benefit the practice. Such instance-driven pattern may help the optimization community to devise theory that fits the practice better. Given the extensive body of literature in this field, there is a possibility that some relevant references may have been inadvertently overlooked. We welcome any comments or questions and are happy to discuss.