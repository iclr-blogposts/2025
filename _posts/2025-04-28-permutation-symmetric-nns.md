---
layout: distill
title: Permutation Symmetric Neural Nets from the Ground Up
description: Dealing with lists, matrices or symbols? When your problem has obvious permutation symmetries, carefully designed permutation symmetric neural networks can enable highly efficient learning, sometimes solving the task from just a few examples. In this post, we'll walk through a simple first principle derivation of permutation symmetric network parameterization using Taylor series, a practical network design using Einsum pooling for various types of permutation symmetry, and from scratch implementations on several problems, including matrix operation, knowledge graph reasoning and a few ARC-AGI challenge problems with permutation symmetry.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

#Anonymize when submitting
authors:
   - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-permutation-symmetric-nns.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: I. In Theory - Paramterizing Symmetric Functions in Taylor Series
    subsections:
    - name: I.1 Example: 1D permutation invariance
    - name: I.2 Exercises
    - name: I.3 What we have learned so far
  - name: II. Engineering a Network with Permutation Symmetry
    subsections:
    - name: II.1 Common types of permutation symmetry
    - name: II.2 Creating a permutation equivariant layer with einsum pooling
    - name: II.3 Putting everything together: The equivariant Einsum network
  - name: III. Use Cases
    subsections:
    - name: III.1 Matrix pseudo inverse
    - name: III.2 Knowledge graph reasoning
    - name: III.3 Permutation symmetry in ARC-AGI problems

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
    .my_dropdown{
    display: block;
    background: #eee;
    }
    .my_dropdown:focus{
    display: block;
    outline: none;
    background: #eee;
    }
    .my_dropdown:focus div{
    display: block;
    background: #eee;
    }
    .my_dropdown a{
    background: #cdf;
    cursor: pointer;
    display: block;
    }
    .my_dropdown > div{
    display:none;
    background: #eee;
    }
    .my_dropdown > div:target{
    display:block; 
    background: #eee;
    }
---


## I. In Theory - Paramterizing Symmetric Functions in Taylor Series

Speaking of permutation invariance, you may already have your faviorite ways to design invariant neural networks for certain types of problems. 
But here we'll introduce a simple yet general Taylor series-based view necessary for studying complex symmetry patterns. Specifically, we need to enable efficient universal learners -- that can represent any such invariant or equivariant functions. 
For advanced readers, the general idea is similar to [Equivariant Multilayer Perceptrons (EMLP)](https://github.com/mfinzi/equivariant-MLP)<d-cite key="finzi2021practical"></d-cite>, universal equivariant MLPs<d-cite key="ravanbakhsh2020universal"></d-cite> as well as many others, but in a more intuitive form. 

Given the desired input-output shapes and symmetry constraints, we would proceed with the following steps: 
1. Express a general function that matches the input-output shapes in Taylor series form.
2. Map the symmetry constraints into equations about the Taylor series coefficients.
3. Solve the equations for free parameters and the parameter sharing patterns, and parameterize the function using the free parameters.
4. Simplify the parameterization for efficient computation.

Let's start from 1D permutation invariance as an example to demonstrate how the process works.


### I.1 Example: 1D permutation invariance

Let's say we want to make a function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
$$
invariant to permutation of $x_0$, $x_1$, $x_2$.

Consider the Taylor series

$$
\begin{aligned}
f & \left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right) \\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Since we want $f(\cdot)$ to be invariant to any permutation matrix $P$, the invariant constraint says 

$$
f\left(\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)=f\left(
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
    \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix}\right)
$$

For our Taylor series form, because of the uniqueness of Taylor series, all order-k coefficients on the left hand side need to match the corresponding order-k coefficients on the right hand side. That is for any permutation matrix $P$ we have

$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix}  = &
\begin{bmatrix} b_0 & b_1 & b_2\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    &   &  \\
    & P^T &  \\
    &   &  
\end{bmatrix} 
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02}\\
    c_{10} & c_{11} & c_{12}\\
    c_{20} & c_{21} & c_{22}
\end{bmatrix} 
    \begin{bmatrix} 
      &   &  \\
      & P &  \\
      &   &  
    \end{bmatrix} \\
& \ldots
\end{aligned}
$$

These equations are all linear equations about coefficients $a$, $b_i$ and $c_{ij}$. So we can just enumerate all $P$ to get all the equations, and then solve them. For $b_i$ for example, enumerating different permutations $P$ would give

$$
\begin{bmatrix} b_0 \\ b_1 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_0 \\ b_2 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_0 \\ b_2\end{bmatrix}  = 
\begin{bmatrix} b_1 \\ b_2 \\ b_0\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_0 \\ b_1\end{bmatrix}  = 
\begin{bmatrix} b_2 \\ b_1 \\ b_0\end{bmatrix} 
$$

That is more than enough to say $b_0=b_1=b_2$. So the order-1 term has only 1 degree of freedom.

For $c_i$ there are more equations, but it turns out that solving the equations across all permutations would yield 
$$c_{00}=c_{11}=c_{22}$$ and $$c_{01}=c_{10}=c_{10}=c_{12}=c_{20}=c_{21}$$. So the order 2 term has 2 degrees of freedom, one for the diagonal and one for everywhere else.

Applying what we have learned, we can now write

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{1}\\
    c_{1} & c_{0} & c_{1}\\
    c_{1} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \end{bmatrix}
+ \ldots
\end{aligned}
$$

For a total of 4 free parameters up to order 2, instead of 13 free parameters without the invariance constraint. More generally, for $N$ inputs, we still only need 4 parameters to express any permutation invariant function, whereas a non-invariant function needs $N^2+N+1$ parameters. In practice, parameterizing with symmetry often **reduces parameter count** exponentially. 

We can further simplify by focusing on the free parameters

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2}\end{bmatrix} \right)\\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 \sum_{i} \sum_{j} x_i x_j + \ldots \\
= & a + b \sum_i x_i + (c_0-c_1) \sum_i x_i^2 + c_1 (\sum_{i}x_i )^2 + \ldots
\end{aligned}
$$

An important effect of this simplification is **reduced compute**. It now requires $O(N)$ compute for $N$ inputs instead of $O(N^2)$ for order-2.  

In math terms, the number of free parameters is the dimensionality of the null space of the symmetry equations. The free parameters can be numerically solved from the basis of this null space. But note that as the basis is often not unique, numerical solutions can vary by a linear combination and therefore may not be compute-optimal, so further simplification is still needed.

Although we didn't unroll order-3 and higher terms because they are difficult to visualize, they can still be analyzed with the same approach. Just imagine a cube or a hypercube of parameters, apply the symmetry transformations simultaneously along all dimensions and solve for the parameter sharing pattern.

### I.2 Exercises
If you are interested in going a little deeper, test yourself on the following list of exercises and gain new insights. Click to expand the reference solution. 

**A. 1D translation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix} x_{3} & x_{0} & x_{1} & x_{2}\end{bmatrix} \right)
$$

<div class="my_dropdown" tabindex="1">
<a>Solution</a> 
<div>

According to equivariant constraints, the coefficients of the Taylor series satisfy

$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_1 & b_2 & b_3 & b_0\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{11} & c_{12} & c_{13} & c_{10} \\
    c_{21} & c_{22} & c_{23} & c_{20} \\
    c_{31} & c_{32} & c_{33} & c_{30} \\
    c_{01} & c_{02} & c_{03} & c_{00} \\
\end{bmatrix}  \\
& \ldots
\end{aligned}
$$

Which means there are 6 free parameters up to order-2.

$$
\begin{aligned}
b_0=b_1=&b_2=b_3 \\
c_{00}=c_{11}=&c_{22}=c_{33} \\
c_{01}=c_{12}=&c_{23}=c_{30} \\
c_{02}=c_{13}=&c_{20}=c_{31} \\
c_{03}=c_{10}=&c_{21}=c_{32} \\
\end{aligned}
$$

Considering Hessian transpose symmetry, we would additionally have $c_{01}=c_{03}$ and that reduces number of free parameters to 5. The parameterization with 6 parameters has an unrolled circular convolution on the order-2 term.

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} & x_{2} & x_{3}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} +
\begin{bmatrix} x_0 & x_{1} & x_{2} & x_3\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{3} & c_{0} & c_{1} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{1} & c_{2} & c_{3} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Computing the 2nd order term naively would require $O(N^2)$ multiplies for length-$N$ input. But because the coefficients implement a circular convolution operation,  fast fourier transforms would reduce compute complexity down to $O(N\log{N})$.

</div>
</div>


**B. Scale.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =f\left(\alpha \begin{bmatrix}x_{1} & x_{2} & x_{0}\end{bmatrix} \right)
$$
For any $\alpha\ne0$.

<div class="my_dropdown" tabindex="1">
<a>Solution</a>
<div>


With Taylor series, you'll run into a conclusion that no terms could exist and $y=a$. That is because scale invariant functions are often not smooth at $x_i=0$ so Taylor series could not capture them. Let us instead look into Laurent Series

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
$$

Applying the invariant constraint

$$
\begin{aligned}
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} c_{ijk} x_0^i x_1^j x_2^k 
=
\sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} \sum_{k=-\infty}^{\infty} \alpha^{i+j+k} c_{ijk} x_0^i x_1^j x_2^k 
\end{aligned}
$$

This only holds when the coefficients match, that is for any $(i,j,k)$, $$c_{ijk}=\alpha^{i+j+k} c_{ijk}$$. That means only terms with $i+j+k=0$ would have non-zero coefficients. For example, $\frac{xy}{z^2}$. Within terms up to order-2, that is $i,j,k\in \left\{ -2,-1,0,1,2 \right\}$, the degrees of freedom is $19$ out of $5^3=125$ as the following

<table>
<tr><th>(i,j,k)</th><th>DoF</th></tr>
<tr><td>0,0,0</td><td>1</td></tr>
<tr><td>-1,0,1</td><td>6</td></tr>
<tr><td>-2,0,2</td><td>6</td></tr>
<tr><td>-2,1,1</td><td>3</td></tr>
<tr><td>2,-1,-1</td><td>3</td></tr>
</table>

The full parameterization is

$$
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) \\
=&a + \sum_{i}\sum_{j\ne i} b_{ij} \frac{x_i}{x_j} + \sum_{i}\sum_{j\ne i} c_{ij} \frac{x_i^2}{x_j^2} + d_0 \frac{x_1 x_2}{x_0^2} + d_1 \frac{x_0 x_2}{x_1^2} + d_2 \frac{x_0 x_1}{x_2^2} \\
&+ e_0 \frac{x_0^2}{x_1 x_2} + e_1 \frac{x_1^2}{x_0 x_2} + e_2 \frac{x_2^2}{x_0 x_1} 
\end{aligned}
$$

Nevertheless, for scale invariance it is easier to reparameterize the input with

$$
z_0=\frac{x_0}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_1=\frac{x_1}{\sqrt{x_0^2+x_1^2+x_2^2}} \quad
z_2=\frac{x_2}{\sqrt{x_0^2+x_1^2+x_2^2}}
$$

and express

$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} &x_{2}\end{bmatrix} \right) =g\left(z_1,z_2\right)
$$

</div>
</div>

**C. 1D permutation with latent.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{0} & x_{1} \\ x_{2} & x_{3}\end{bmatrix} \right)
$$

<div class="my_dropdown" tabindex="1">
<a>Solution</a>
<div>

According to the equivariant constraint, the coefficients of the Taylor series satisfy

$$
\begin{aligned}
a = & a \\
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix}  = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
\end{aligned}
$$

Solving the equations gives

$$
\begin{aligned}
\begin{bmatrix}b_0 & b_1\end{bmatrix}=&\begin{bmatrix}b_2 & b_3\end{bmatrix} \\

\begin{bmatrix} 
    c_{00} & c_{01} \\
    c_{10} & c_{11} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{22} & c_{23} \\
    c_{32} & c_{33} \\
\end{bmatrix}  \\
\begin{bmatrix} 
    c_{02} & c_{03} \\
    c_{12} & c_{13} \\
\end{bmatrix} 
=&
\begin{bmatrix} 
    c_{20} & c_{21} \\
    c_{30} & c_{31} \\
\end{bmatrix} 
\end{aligned}
$$

If we view the rows of inputs as vectors, the coefficients can be partitioned into blocks that process those vectors, and the row-permutation invariant constraint leads to parameter sharing at the block level. We can parameterize

$$
\begin{aligned}
f&\left( \begin{bmatrix}x_{0} & x_{1}\end{bmatrix}, \begin{bmatrix}x_{2} & x_{3}\end{bmatrix} \right) \\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_{0}+x_2 \\ x_{1}+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0 & c_1\\c_2 & c_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
\\
&+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}\\
=&a + 
\begin{bmatrix}b_{0} & b_{1}\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
+ 
\begin{bmatrix}x_{0} & x_{1}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1}\end{bmatrix}
+ 
\begin{bmatrix}x_{2} & x_{3}\end{bmatrix}
\begin{bmatrix}c_0' & c_1'\\c_2' & c_3'\end{bmatrix}
\begin{bmatrix}x_{2} \\ x_{3}\end{bmatrix}\\
&+ 
\begin{bmatrix}x_0+x_2 & x_1+x_3\end{bmatrix}
\begin{bmatrix}d_0 & d_1\\d_2 & d_3\end{bmatrix}
\begin{bmatrix}x_0+x_2 \\ x_1+x_3\end{bmatrix}
\end{aligned}
$$

The size of order-$k$ coefficient blocks for processing length-$H$ latent vectors is $H^k$. This is already much better than the full coefficients $(NH)^k$ for a set of $N$ vectors but is still large. Now, the bread and butter of deep learning comes in, namely 1) stacking more layers, 2) low-rank factorization and 3) adding non-linearities which we'll discuss more in Section II.

</div>
</div>

**D. 2D permutation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) = f\left(\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix}\begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix} \right)
$$

<div class="my_dropdown" tabindex="1">
<a>Solution</a>
<div>


According to the equivariant constraint, the coefficients of the Taylor series satisfy

$$
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_2 & b_3 & b_0 & b_1\end{bmatrix} =
\begin{bmatrix} b_1 & b_0 & b_3 & b_2\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{22} & c_{23} & c_{20} & c_{21} \\
    c_{32} & c_{33} & c_{30} & c_{31} \\
    c_{02} & c_{03} & c_{00} & c_{01} \\
    c_{12} & c_{13} & c_{10} & c_{11} \\
\end{bmatrix} 
=
\begin{bmatrix} 
    c_{11} & c_{10} & c_{13} & c_{12} \\
    c_{01} & c_{00} & c_{03} & c_{02} \\
    c_{31} & c_{30} & c_{33} & c_{32} \\
    c_{21} & c_{20} & c_{23} & c_{22} \\
\end{bmatrix} 
\end{aligned}
$$

Solving the equations gives the following parameterization with 6 degrees of freedom

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b & b & b &b\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{1} & c_{0} & c_{3} & c_{2}\\
    c_{2} & c_{3} & c_{0} & c_{1}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
$$

Let us perform a bit of merging and simplification

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + b \sum_i \sum_j x_{ij} + (c_0-c_1-c_2-c_3) \sum_i \sum_j x_{ij}^2\\
&+ (c_1-c_3) \sum_i (\sum_j x_{ij})^2+(c_2-c_3) \sum_j (\sum_i x_{ij})^2  + c_3 (\sum_i \sum_j x_{ij})^2
+ \ldots
\end{aligned}
$$

An interesting pattern emerges, that all terms are some forms of tensor contractions. In fact, this seems to be true for all flavors of permutation symmetry and the motivation behind Section II. Don't believe it? Try another case below!

</div>
</div>


**E. 2D joint permutation.** Parameterize function 
$$
y=f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right) =f\left( \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}\right)
$$

<div class="my_dropdown" tabindex="1">
<a>Solution</a>
<div>

According to the equivariant constraint, the coefficients of the Taylor series satisfy

$$
\begin{aligned}
\begin{bmatrix} b_0 & b_1 & b_2 & b_3\end{bmatrix} = &
\begin{bmatrix} b_3 & b_2 & b_1 & b_0\end{bmatrix} 
\\
\begin{bmatrix} 
    c_{00} & c_{01} & c_{02} & c_{03}\\
    c_{10} & c_{11} & c_{12} & c_{13}\\
    c_{20} & c_{21} & c_{22} & c_{23}\\
    c_{30} & c_{31} & c_{32} & c_{33}
\end{bmatrix} 
=& 
\begin{bmatrix} 
    c_{33} & c_{32} & c_{31} & c_{30} \\
    c_{23} & c_{22} & c_{21} & c_{20} \\
    c_{13} & c_{12} & c_{11} & c_{10} \\
    c_{03} & c_{02} & c_{01} & c_{00} \\
\end{bmatrix} 
\end{aligned}
$$

Solving the equations gives the following parameterization with 11 degrees of freedom

$$
\begin{aligned}
y=&f\left( \begin{bmatrix}x_{00} & x_{01} \\ x_{10} & x_{11}\end{bmatrix} \right)\\
= & a + 
\begin{bmatrix} b_0 & b_1 & b_1 &b_0\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix} +
\begin{bmatrix} x_{00} & x_{01} & x_{10} & x_{11}\end{bmatrix}
\begin{bmatrix} 
    c_{0} & c_{1} & c_{2} & c_{3}\\
    c_{4} & c_{5} & c_{6} & c_{7}\\
    c_{7} & c_{6} & c_{5} & c_{4}\\
    c_{3} & c_{2} & c_{1} & c_{0}
\end{bmatrix} 
\begin{bmatrix} x_{00} \\ x_{01} \\ x_{10} \\ x_{11} \end{bmatrix}
+ \ldots
\end{aligned}
$$

With Hessian transpose symmetry, we may further have $c_1=c_4$ and $c_2=c_7$ which reduces free parameters count down to 9, still 3 more than regular 2D permutation invariance. If you squint really hard (and maybe try Exercise D), there exists a tensor contraction form:

$$
\begin{aligned}
f&\left( \{x_{ij}\} \right)\\
= & a + b_0' \sum_i x_{ii} + b_1' \sum_i \sum_j x_{ij} + 
c_0' \sum_i x_{ii}^2 
+ c_1' \sum_i x_{ii}\sum_j x_{ij} \\
&+ c_2' \sum_i x_{ii}\sum_j x_{ji} 
+ c_3' \sum_i x_{ii}\sum_j x_{jj}
+ c_5' \sum_i \sum_j x_{ij}^2
+ c_6' \sum_i \sum_j x_{ij} x_{ji}  
+ \ldots
\end{aligned}
$$

What's different from regular 2D permutation invariance are terms involving diagonal and transpose. Also all tensor contractions here are at or below $O(N)$ compute for input size $\sqrt{N}\times \sqrt{N}$, which is exponentially less compute than $O(N^2)$ for the default Taylor series.

</div>
</div>


**F. 1D permutation equivariance.** Parameterize function 
$$
\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) =F\left( \begin{bmatrix}&&\\&P&\\&&\end{bmatrix}\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right)
$$
For any permutation $P$.


<div class="my_dropdown" tabindex="1">
<a>Solution</a>
<div>


The Taylor series up to order 1 can be expressed as

$$
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a_0 \\ a_1 \\a_2\end{bmatrix}
+ 
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
$$

The equivariant constraints are for any $P$

$$
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
=
\begin{bmatrix}
 & & \\ 
 & P^T & \\ 
 & & \\ 
\end{bmatrix}
\begin{bmatrix}
b_{00} & b_{01} & b_{02} \\ 
b_{10} & b_{11} & b_{12} \\
b_{20} & b_{21} & b_{22}
\end{bmatrix}
\begin{bmatrix}
 & & \\ 
 & P & \\ 
 & & \\ 
\end{bmatrix}
$$

Which is identical to the invariant constraints on order-2 terms. In general, the parameterization of an equivariant function up to order-k is very much the same as an invariant function up to order-(k+1). In the case of 1D permutation equivariance, the order-1 parameterization would be

$$
\begin{aligned}
&\begin{bmatrix}y_{0} \\ y_{1} \\y_{2}\end{bmatrix}=F\left( \begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} \right) \\
&=\begin{bmatrix}a \\ a \\a\end{bmatrix}
+ 
\begin{bmatrix}
b_{0} & b_{1} & b_{1} \\ 
b_{1} & b_{0} & b_{1} \\
b_{1} & b_{1} & b_{0}
\end{bmatrix}
\begin{bmatrix}x_{0} \\ x_{1} \\ x_{2}\end{bmatrix} 
+
\ldots
\end{aligned}
$$

Rewriting in tensor contraction form using Numpy's einsum notations

$$
\begin{aligned}
Y=&F\left( X \right) \\
=& a \cdot \text{einsum('i->i',X)} 
+ (b_0-b_1) \cdot \text{einsum('i,i->i',X,X)} \\
&+ b_1 \cdot \text{einsum('i,j->i',X,X)} 
\end{aligned} 
$$


</div>
</div>


### I.3 What we have learned so far

In this section, we have learned that
1. Symmetry constraints reduce the number of free parameters.
2. A Taylor-series technique can be used to parameterize symmetric functions.
3. Different symmetries can have different impacts on degrees of freedom.
4. Certain parameterizations can reduce compute exponentially.
5. Parameterization of equivariant functions are tied to parameterization of invariant functions
6. Permutation invariant and equivariant functions can be parameterized solely using tensor contraction terms.


A Taylor series parameterization is sound in theory. In practice however, functions compound and high order interactions are common. Taylor series often provides too little relevant capacity and too much irrelevant capacity to be useful. Engineering is key in the journey to create universal learners of symmetric functions. In the next section, we'll focus on permutation symmetry and design a family of practical invariant and equivariant networks for various flavors of permutation symmetry.


## II Engineering a Network with Permutation Symmetry

From matrices to sets to symbolic processing, permutation symmetry is found in many problems and requires extra attention during modeling. When handled properly however, permutation symmetry is also a blessing. As we have learned in the previous section, if parameterized properly, permutation symmetry has the potential to exponentially reduce parameter count and compute for highly efficient learning. At the other end of the spectrum, reciting the success recipe of deep learning, we can scale the latent dimension and stack equivariant layers to create exponentially more expressive networks at the same parameter count and compute as a regular network.


Devil's in the details, in this section we'll walk through the design of permutation symmetric neural networks for various types of permutation symmetry.


There are many places where you'll see permutation symmetry and they often come in different forms. So in Section II.1 we'll first start from a summary of common types of permutation symmetry. 
And then Section II.2 will discuss the design of permutation equivariant layers. Permutation symmetry turns out to be closely tied to tensor contractions. That would allow us to synthesize efficient high-order permutation equivariant layers automatically in a procedural manner.
And finally in Section II.3 we discuss further optimizations that helps practical implementation. 

### II.1 Common types of permutation symmetry

In the following table we analyze a few common problems by their type of permutation symmetry.

| Problem       | Illustration  | Symmetry type  | Dependency |
| ------------- |:-------------:|:--------------:|:---------- |
| Predicting important weights to keep in a linear layer while pruning. Equivariant to neuron permutations along input `a` and output `b`. | ![Weight matrices](../../assets/img/2025-04-28-permutation-symmetric-nns/ab.png) | `ab` |            |
| Group classes by common misclassifications given confusion matrices. Equivariant to class permutations `a` which simultaneously permutes both axes.  | ![Confusion matrices](../../assets/img/2025-04-28-permutation-symmetric-nns/aa.png) | `aa` |            |
| Predicting who's winning given player's hands. Equivariant to player ID `a`. Invariant to permutation of suits `c`. Invariant to card reordering `b` for each player independently. No symmetry against card number `H`.  | ![Poker hands](../../assets/img/2025-04-28-permutation-symmetric-nns/abcH.png) | `abcH` |  `a->b`  |






To aid discussions, we use a custom notation to describe the specific type of permutation symmetry, to capture both the input shape and the unique permutation axes. We have multiple dimensions and joint permutations whose symmetry notations are straight forward. 
There's also a dependency aspect, which captures a concept of "list of different lists". 
A fully independent batch dimension `Z` and a non-symmetric latent dimension `H` may be added optionally.


### II.2 Creating a permutation equivariant layer with einsum pooling

Across all types of permutation symmetry, as we learned in Section I through Taylor series, it turns out that tensor contractions are are all you need for parameterizing permutation invariant and equivariant layers, which can then be stacked into a deep network.

Intuitively, tensor contractions like

$$
Y_{ij}=\sum_k \sum_l X_{ik} X_{lk} X_{lj} 
$$

create a new tensor that has the same shape as the input while summing over unused dimensions. They achieve a permutation equivariant effect. And tensor contractions like

$$
y=\sum_i \sum_j \sum_k \sum_l X_{ik} X_{lk} X_{lj} 
$$

that sum over all dimensions achieve a permutation invariant effect. 

As the math equations can get quite long, we will use the [einsum notation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) which represents a tensor contraction using the indices involved. It is widely used across deep learning frameworks to denote tensor contractions. For example,

```python
Y=einsum('Zik,Zlk,Zlj->Zij',X,X,X)
y=einsum('Zik,Zlk,Zlj->Z',X,X,X)
```

Here a batch dimension Z is added to make sure the right hand side is not empty.  

**How to create an equivariant layer given permutation symmetry type from tensor contractions?**
Our answer is two fully connected layers with a pooling layer in between. 

Let us use a 1D + latent `aH`-type equivariant constraint as an example to illustrate the design.

The Taylor Series parameterization up to order 2 is

```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ab,ca->cb',b0_HH,X_aH)
     +einsum('ab,ca->db',b1_HH,X_aH)
     +einsum('abc,da,db->dc',c0_HHH,X_aH,X_aH)
     +einsum('abc,da,db->ec',c1_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->dc',c2_HHH,X_aH,X_aH)
     +einsum('abc,da,eb->fc',c3_HHH,X_aH,X_aH)
     +...
```

We can immediately see that the order-1 terms have $H^2$ parameters and order-2 terms have $H^3$ parameters, which would naturally need a low-rank($=K$) treatment, such as

```python
Y_abH=einsum('a->ba',a_H)
     +einsum('ka,kb,ca->cb',b0U_KH,b0V_KH,X_aH)
     +einsum('ka,kb,ca->db',b1U_KH,b1V_KH,X_aH)
     +einsum('ka,kb,kc,da,db->dc',c0U,c0V,c0W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,db->ec',c1U,c1V,c1W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->dc',c2U,c2V,c2W,X_aH,X_aH)
     +einsum('ka,kb,kc,da,eb->fc',c3U,c3V,c3W,X_aH,X_aH)
     +...
```


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-permutation-symmetric-nns/layer.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Low-rank decomposition of high order coefficients. Right: An equivariant einsum layer with two linear layers an einsum pooling in between, which is the result of moving low-rank coefficients into linear fan-in and fan-out layers.
</div>


We can move the order-0 parameters, as well as $U$, $V$, $W$ matrices into fully connected layers along $H$ that perform input preprocessing and output post_processing. So the end result is two linear layers with pooling in between, and for pooling we need 

```python
Y_abH_0V=einsum('ck->ck',X_aH_0U)
Y_abH_1V=einsum('ck->dk',X_aH_1U)
Y_abH_0W=einsum('dk,dk->dk',X_aH_0U,X_aH_0V)
Y_abH_1W=einsum('dk,dk->ek',X_aH_1U,X_aH_1V)
Y_abH_2W=einsum('dk,ek->dk',X_aH_2U,X_aH_2V)
Y_abH_3W=einsum('dk,ek->fk',X_aH_3U,X_aH_3V)
...
```

Notice that `'dk,ek->fk'` can be composed with `'ck->dk'` for each operand individually, and then combine using `'dk,dk->dk'`. As we can stack more layers, not all pooling operations are needed and less pooling operations would reduce network complexity. In fact, this might be a good point to step back and ask: **Given equivariance type, e.g. `aH`, how can we identify the minimum yet sufficient set of pooling operations?**

The following recipe might be helpful for designing pooling operations given equivariance type in practice:

1) Enumerate all valid and unique einsum operations up to order-k that are compatible with the given equivariance type. For example `einsum('ab,bc->ac',X_ab,X_ab)` is compatible with `aa`-type equivariance, but not compatible with `ab`-type equivariance. Also notice that `ba,ac->bc` is the just a renaming of `ab,bc->ac`. There is a graph homomorphism problem under the hood for listing unique einsum operations and interested readers can dig deeper.

2) Filter einsum operations based on dependency requirement of the given equivariance type. For example `einsum('ab,cb->cb',X_ab,X_ab)` satisfy `b->a` dependency but does not satisfy `a->b` dependency for `ab`-type equivariance.

3) Filter out order-2+ "breakable" operations that can be divided into two lower order terms with a simple pointwise multiplication. For example `ab,cb,cd,ad->ad` can be divided into `ab,cb,cd->ad` and `ad->ad` which can the be put together with `ab,ab->ab`, so it is not necessary as long as the lower order terms exist. 

4) Normalize the rotation of input/output terms. For example for `aa`-type equivariance, `ab,cb->ca` is not necessary because it can be achieved with `ab,bc->ac`, through applying rotations `ab->ba` on the input and output.

5) Remove order-2+ operations that expand new dimensions in the output term. For example `ab,bc->ad` is redundant because it can be achieved through `ab,bc->ac` followed by a dimension expansion operation `ab->ac`.

An algorithm that properly de-duplicates through compositions remains to be developed. But after all the filtering listed here, there is usually a quite compact initial set of pooling operations for further optimizations.

The following is a quick lookup table of pooling operations for a few common equivariance types.


| Symmetry type | Order | Pooling operation(s) |
| ------------- |-------|:------------------:|
| `aH`          | 1     | `aH->aH`, `aH->bH` |
|               | 2     | `aH,aH->aH`  |
|               | 3+    | No need  |
| `abH`         | 1     | `abH->abH`, `abH->cbH`, `abH->acH` |
|               | 2     | `abH,abH->abH` |
|               | 3     | `abH,cbH,cdH->adH` |
|               | 4     | No need |
|               | 5     | `abH,acH,dbH,dcH,deH->aeH`, `abH,acH,dbH,dcH,ecH->ebH` |
| `aaH`         | 1     | `abH->abH`,`aaH->aaH`,`abH->baH`,`abH->cbH` |
|               | 2     | `abH,bcH->acH`,`abH,abH->abH` |
|               | 3+    | No need  |


<blockquote>
Did you know: linear self-attention ab,cb,cd->ad is an order-3 term for ab-type equivariance. Self-attention operation by itself is equivariant to not only token permutation but also latent rotation, although other linear layers in the transformer architecture do not retain the latent symmetry.
</blockquote>


### II.3 Putting everything together: The equivariant Einsum network

With an equivariant layer, we can stack them to create a practical high capacity neural net that learns well. Let's apply the following recipe

1. Stacking multiple equivariant layer to create a deep network.
2. GELU nonlinearity between equivariant layers to add to the depth and create bottlenecks.
3. Residual connections for better optimization dynamics.
4. Average pooling to create invariant dimensions if the symmetry involves invariance. 

The result is an equivariant backbone as follows.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-permutation-symmetric-nns/arch.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Stacking multiple equivariant layers to create a practical high capacity network.
</div>

High order einsum operations can result in large values. One consideration in practice is how to normalize them to be in line with lower order terms. Our experience is that mean subtraction and standard deviation normalization seems to help learning, but obviously more exploration is still needed. Found any better strategies? Let us know!

Another consideration in practice is [einsum path optimization](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html). For example, the einsum string `ab,dc,ae,ac,db->de` by default is programmed to be computed pairwise from left to right. By the third term, a large factor `abcde` would be created and stress the memory. Instead, if we compute pairwise via path `ab,db->ad`, `ac,dc->ad`, `ad,ad->ad` and `ad,ae->de`, the largest intermediate factor would only be 2-dimensional and the computation can also be done much faster. For modeling complex higher-order interations under certain types of symmetries, large einsums may be unavoidable, and computing them might be an interesting compute challenge.

Putting it all together, here is a reference Pytorch implementation of an Equivariant Einsum Network that would served as the backbone, to be followed by averaging for dimensions that need invariance. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implements minimal aH-type pooling
class einpool_a(nn.Module):
    fan_in=4
    fan_out=3
    ndims=1
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-2:]) # Apply pooling only to the last 2 dims, supposedly `aH`
        N,KH=x.shape[-3:]
        H=KH//self.fan_in
        x=x.split(H,dim=-1)
        y0=x[0]
        y1=x[1].mean(-2,keepdim=True).repeat(1,N,1)
        y2=x[2]*x[3]
        y=torch.cat((y0,y1,y2),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y

#Implements minimal aaH-type pooling
class einpool_aa(nn.Module):
    fan_in=8
    fan_out=6
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `aaH`
        N,M,KH=x.shape[-3:]
        H=KH//self.fan_in
        xn=x.view(-1,N*M,KH)
        xn=F.normalize(xn-xn.mean(-2,keepdim=True),dim=-2,p=2,eps=1e-1).view(*x.shape)
        x=x.split(H,dim=-1)
        xn=xn.split(H,dim=-1)
        y0=x[0]
        y1=x[1].diagonal(dim1=-2,dim2=-3).diag_embed(dim1=-2,dim2=-3)
        y2=x[2].transpose(-2,-3)
        y3=x[3].mean(-2,keepdim=True).repeat(1,1,M,1)
        y4=x[4]*x[5]
        y5=torch.einsum('ZabH,ZbcH->ZacH',xn[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4,y5),dim=-1)
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y


#Implements order-3 abH-type pooling
class einpool_ab(nn.Module):
    fan_in=8
    fan_out=5
    ndims=2
    def forward(self,x_):
        x=x_.view(-1,*x_.shape[-3:]) # Apply pooling only to the last 3 dims, supposedly `abH`
        N,M,KH=x.shape[-3:] 
        H=KH//self.fan_in
        xn=x.view(-1,N*M,KH)
        xn=F.normalize(xn-xn.mean(-2,keepdim=True),dim=-2,p=2,eps=1e-12).view(*x.shape)
        #xn=F.softmax(x.view(-1,N*M,KH),dim=-2).view(*x.shape)
        x=x.split(H,dim=-1)
        xn=xn.split(H,dim=-1)
        y0=x[0]
        y1=x[1].mean(-2,keepdim=True).repeat(1,1,M,1)
        y2=x[2].mean(-3,keepdim=True).repeat(1,N,1,1)
        y3=xn[3]*x[4]
        y4=torch.einsum('ZacH,ZbcH,ZadH->ZbdH',xn[5],xn[6],x[7])
        y=torch.cat((y0,y1,y2,y3,y4),dim=-1) #
        y=y.view(*x_.shape[:-1],-1) #Recover original tensor shape
        return y


#Equivariant EinNet layer
class einnet_layer(nn.Module):
    def __init__(self,ninput,nh0,noutput,pool):
        super().__init__()
        self.fan_in=nn.Linear(ninput,nh0*pool.fan_in)
        self.fan_out=nn.Linear(nh0*pool.fan_out,noutput)
        self.pool=pool
    
    def forward(self,x):
        h=self.fan_in(x)
        h=self.pool(h)
        return self.fan_out(h)

#Equivariant EinNet backbone
class einnet(nn.Module):
    def __init__(self,ninput,nh0,nh,noutput,nstacks,pool):
        super().__init__()
        self.t=nn.ModuleList()
        self.t.append(einnet_layer(ninput,nh0,nh,pool))
        for i in range(nstacks-1):
            self.t.append(einnet_layer(nh,nh0,nh,pool))
        
        self.t.append(einnet_layer(nh,nh0,noutput,pool))
        self.pool=pool
    
    # Forward call
    #    x: tensor shape matches equivariance type, e.g. *abH
    def forward(self,x):
        h=self.t[0](x)
        for i in range(1,len(self.t)-1):
            h=h+self.t[i](F.gelu(h))
        
        return self.t[-1](F.gelu(h))

#Example usage
#    net=einnet(1,16,64,1,2,einpool_ab())
```




## III Use cases

Now that we have designed a network, let's use a variety of toy problems to test its capability. 

### III.1 Matrix pseudo inverse

A good sanity check of a permutation equivariant network is whether it can learn matrix inverse. The Moore-Penrose inverse is defined as

$$
A^{+}=(A^TA)^{-1}A^T
$$

For any permutation matrix $P$ the Moore-Penrose inverse satisfies

$$
(PA)^+=A^+ P^T \\
(AP)^+=P^T A^+
$$

So $F(A)=(A^+)^T$ is row-column permutation equivariant. In fact, it is further equivariant to arbitrary rotation.



In this exercise, we will train an `abH`-type equivariant EinNet to perform Moore-Penrose inverse on $8\times 16$ matrices from 1000 training examples to test its capability.


The python implementation is as the follows. We use a 12-stack EinNet backbone with 128 hidden dims and 32 head dims, and minimize L2 loss against ground truth using AdamW with lr=1e-3.

```python
import torch
import torch.optim as optim
ntrain,ntest,H,W=1000,100,8,16

torch.manual_seed(0)
x_train=torch.Tensor(ntrain,H,W).normal_().cuda()
y_train=torch.linalg.pinv(x_train).transpose(-1,-2)
x_test=torch.Tensor(ntest,H,W).normal_().cuda()
y_test=torch.linalg.pinv(x_test).transpose(-1,-2)

net=einnet(ninput=1,nh0=32,nh=128,noutput=1,nstacks=6,pool=einpool_ab()).cuda()

opt=optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)

for i in range(10000):
    opt.zero_grad()
    pred=net(x_train.unsqueeze(-1)).squeeze(-1)
    loss=((y_train-pred)**2).mean()
    loss.backward()
    opt.step()
    
    if i%100==0:
        with torch.no_grad():
            pred=net(x_train.unsqueeze(-1)).squeeze(-1)
            err_train=(torch.bmm(pred,x_train.transpose(-1,-2))-torch.eye(H).cuda())
            err_train=err_train.abs().mean()
            
            pred=net(x_test.unsqueeze(-1)).squeeze(-1)
            err_test=(torch.bmm(pred,x_test.transpose(-1,-2))-torch.eye(H).cuda())
            err_test=err_test.abs().mean()
        
        print('Iter %d, mae_train %.4f, mae_test %.4f'%(i,err_train,err_test))
```

Let us plot the element-wise mean absolute error between $A^+A$ and $I$ over the course of learning. 


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-permutation-symmetric-nns/MM.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Learning from 1000 examples for 10k iterations gives near 2 orders of magnitude drop in Mean Absolute Error starting from random, with no signs of overfitting.
</div>



With only 640k parameters (roughly the size of a $800\times800$ linear layer), 1000 training points and 10k iterations, we get to 0.0015 MAE lowest which is 2 orders of magnitudes below random 0.1250 MAE. Running the optimization for longer may further reduce MAE.


### III.2 Knowledge graph reasoning

Knowledge graphs are structured representations of real-world entities and their relations, organized in the form of graphs. They consist of nodes representing entities (such as people, places, and objects) and edges that denote the relationships between these entities. 

Permutation symmetry can play an important role in knowledge graph reasoning. An edge (entity1, relation, entity2) indicates a relationship between entity1 and entity2. For example, (A, father_of, B), or (C, sister_of, A). Knowledge extracted from text is often incomplete. In the case of missing edges, knowledge graph completion or link prediction is the task of inferring missing edges in the knowledge graph, such as (C, aunt_of, B). That is because the underlying logic rule "if (A, father_of, B) and (C, sister_of, A), then (C, aunt_of, B)" generally stands for any A, B and C. That is why knowledge graph reasoning is expected to be equivariant to arbitrary permutation of the entities.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-permutation-symmetric-nns/KG.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Knowledge graph completion requires logic reasoning. Permutation symmetry plays an important role, because the logic rule used for reasoning is independent of entity permutations.
</div>



Let us represent a knowledge graph as a sparse tensor $X_{ijr}$, where $X(i,j,r)=1$ if there is an edge between entity i and entity j with relation r, and $X(i,j,r)=0$ otherwise. We want to learn an `aaH`-type permutation equivariant network equivariant to entity permutation

$$Y^{\text{head}}_{ijr},Z^{\text{tail}}_{ijr}=F(X_{ijr})$$

Here we output two sets of logits for "head" and "tail" prediction -- predicting entity1 given (*, relation, entity2) and predicting entity2 given (entity1, relation, *) respectively. Standard loss function is

```python
loss_ijr=cross_entropy(Y[:,j,r],i)+cross_entropy(Z[i,:,r],j)
```
to be enumerated across all heldout edges (i,r,j).

Here we have a python implementation for knowledge graph completion on the Alyawarra Kinship dataset. The Alyawarra Kinship dataset records the kinship terms among 102 Alyawarra-speaking Aboriginal people of central Austraila. It can be thought of as a real world instance of the "sister of father is aunt" example. It is one of the classical datasets for evaluating knowledge graph reasoning.

The dataset (`train.txt`, `valid.txt` and `test.txt`) are available from the [ConvE Github repository](https://github.com/TimDettmers/ConvE/blob/master/kinship.tar.gz).

```python
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
#Read *.txt file for tuples
def read_data(fname):
    data=[]
    with open(fname,'r') as f:
        for line in f:
            data.append(line.rstrip('\n').split('\t'))
    return data

#Compute metrics. pred, ref and exclude are [R,N,N] tensors
#pred=predictions, ref=gt, exclude=all non-gt tuples 
def perf(pred,ref,exclude):
    R,N,_=ref.shape
    ref=ref.nonzero().tolist()
    exclude=exclude.nonzero().tolist()
    ref=[tuple(x) for x in ref]
    exclude=[tuple(x) for x in exclude]
    ref_lookup={k:1 for k in ref+exclude}
    #head
    rank_head=[]
    pred_head=pred[:R]
    for t in ref:
        options=torch.LongTensor([i for i in range(N) if i==t[1] or not (t[0],i,t[2]) in ref_lookup])
        sgt=pred_head[t[0],t[1],t[2]]
        sopt=pred_head[t[0],options,t[2]]
        rank=sopt.ge(sgt).long().sum()
        rank_head.append(rank)
    
    #tail
    rank_tail=[]
    pred_tail=pred[R:]
    for t in ref:
        options=torch.LongTensor([i for i in range(N) if i==t[2] or not (t[0],t[1],i) in ref_lookup])
        sgt=pred_tail[t[0],t[1],t[2]]
        sopt=pred_tail[t[0],t[1],options]
        rank=sopt.ge(sgt).long().sum()
        rank_tail.append(rank)
    
    rank_head=torch.LongTensor(rank_head)
    rank_tail=torch.LongTensor(rank_tail)
    mrr_head=(1/rank_head.float()).mean()
    mrr_tail=(1/rank_tail.float()).mean()
    
    top1=(rank_head.eq(1).float().mean()+rank_tail.eq(1).float().mean())/2
    top3=(rank_head.le(3).float().mean()+rank_tail.le(3).float().mean())/2
    top10=(rank_head.le(10).float().mean()+rank_tail.le(10).float().mean())/2
    mrr=(mrr_head+mrr_tail)/2
    return float(top1),float(top3),float(top10),float(mrr)

def to_matrix(links,N,R):
    X=torch.sparse_coo_tensor(links.t(),[1.0 for i in range(len(links))],[N,N,R])
    return X.coalesce().to_dense().cuda()

train=read_data('train.txt')
val=read_data('valid.txt')
test=read_data('test.txt')

entities=sorted(list(set([x[0] for x in train+val+test]+[x[2] for x in train+val+test])))
relations=sorted(list(set([x[1] for x in train+val+test])))
N=len(entities)
R=len(relations)

x_train=torch.Tensor([[entities.index(x[0]),entities.index(x[2]),relations.index(x[1])] for x in train])
x_val=torch.Tensor([[entities.index(x[0]),entities.index(x[2]),relations.index(x[1])] for x in val])
x_test=torch.Tensor([[entities.index(x[0]),entities.index(x[2]),relations.index(x[1])] for x in test])
X_train=to_matrix(x_train,N,R)
X_val=to_matrix(x_val,N,R)
X_test=to_matrix(x_test,N,R)

def split(X,p=0.9):
    mask=torch.rand_like(X).lt(p).float()
    return X*mask,X*(1-mask)

def loss_ce(pred,Y):
    R=Y.shape[-1]
    pred_head=pred[:,:,:R]
    pred_tail=pred[:,:,R:]
    ind=Y.nonzero()
    s=pred_tail[ind[:,0],:,ind[:,2]]
    loss_tail=F.cross_entropy(s,ind[:,1])
    s=pred_head[:,ind[:,1],ind[:,2]].contiguous().t()
    loss_head=F.cross_entropy(s,ind[:,0])
    return loss_head,loss_tail

net=einnet(R,64,128,R*2,6,einpool_aa()).cuda()
opt=optim.Adam(net.parameters(),lr=1e-3)
t0=time.time()
for epoch in range(100):
    opt.zero_grad()
    net.train()
    for i in range(100): 
        X,Y=split(X_train)
        pred=net(X)
        loss_head,loss_tail=loss_ce(pred,Y)
        loss=loss_head+loss_tail
        loss.backward()
        print('%d, loss %.4f, time %.2f    '%(epoch,loss,time.time()-t0),end='\r')
        if (i+1)%16==0:
            opt.step()
            opt.zero_grad()
    
    net.eval()
    #Calculate performance on val split
    with torch.no_grad():
        pred=net(X_train)
    
    top1,top3,top10,mrr=perf(pred.permute(2,0,1),X_val.permute(2,0,1),(X_train+X_test).permute(2,0,1))
    print('epoch %d val top1 %.4f top3 %.4f top10 %.4f mrr %.4f'%(epoch,top1,top3,top10,mrr))
    #Calculate performance on test split
    #Use both train and val edges to make prediction on test
    with torch.no_grad():
        pred=net(X_train+X_val)
    
    top1,top3,top10,mrr=perf(pred.permute(2,0,1),X_test.permute(2,0,1),(X_train+X_val).permute(2,0,1))
    print('epoch %d test top1 %.4f top3 %.4f top10 %.4f mrr %.4f'%(epoch,top1,top3,top10,mrr))
```


Standard evaluation metric for knowledge graph completion is top-1, top-5, top-10 and mean reciprocal rank (MRR) for head- and tail-prediction tasks, after filtering of known edges. 

| Model | MRR  | Hits@10 | Hits@3 | Hits@1|
| ----- |:----:|:-------:|:------:|:-----:|
| ConvE | 0.83 |  0.98   | 0.91   | 0.73  |
| EinNet `aaH` | 0.94 |  1.00   | 0.99   | 0.90  |

At 94% MRR, the equivariant network has a strong showing here and does much better than the classic embedding-based ConvE model and even more recent models. Maybe its closer to actually doing some logic reasoning? The limitation here is that memory cost is quadratic to number of entities and 1000 entities seems to be the current limit, where as embedding-based methods can scale to millions of entities. Further research on design is needed for practical applications.


### III.3 Permutation symmetry in ARC-AGI problems


The Abstraction Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a collection of complex puzzles for which even advanced Large Language Models (LLMs) struggle to solve. The task is to construct an output "grid" or colors given an input "grid", following several demonstrations of input-output pairs. 

While the ARC-AGI benchmark tests a wide range of intelligence capabilities, there are a few interesting ones with permutation symmetry that can seemingly be solved by a permutation symmetric NN trained only on the ~3 given demonstrations. Here are a few of them.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-permutation-symmetric-nns/ARC-AGI.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Four ARC-AGI tasks that can be solved by permutation symmetric neural network, training solely on the ~3 provided demonstrations.
</div>

Using problem `a406ac07` as an example, we first need to implement an `aab`-type pooling operation. Here we show an implementation for einsum pooling up to order-2.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class einpool_aab(nn.Module):
    fan_in=10
    fan_out=8
    def forward(self,x):
        N,M,T,KH=x.shape
        H=KH//self.fan_in
        xn=x.view(-1,N*M*T,KH)
        xn=F.normalize(xn-xn.mean(-2,keepdim=True),dim=-2,p=2,eps=1e-6).view(*x.shape)
        x=x.split(H,dim=-1)
        xn=xn.split(H,dim=-1)
        y0=x[0]
        y1=x[1].mean(-2,keepdim=True).repeat(1,1,T,1)
        y2=x[2].mean(-3,keepdim=True).repeat(1,M,1,1)
        y3=x[3].mean(-4,keepdim=True).repeat(N,1,1,1)
        y4=x[4].diagonal(dim1=-3,dim2=-4).diag_embed(dim1=-3,dim2=-4)
        y5=x[5].transpose(-3,-4)
        y6=xn[6]*x[7]
        y7=torch.einsum('abdH,bcdH->acdH',xn[8],x[9])
        y=[y0,y1,y2,y3,y4,y5,y6,y7]
        y=torch.cat(y,dim=-1)
        return y

```

The following implementation use a 3-stack `aab`-type EinNet to solve the `a406ac07` task, using only provided training instances.

```python

import torch
import torch.nn.functional as F

#evaluation/a406ac07.json
data_train=[{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 5], [9, 9, 8, 8, 7, 7, 6, 6, 5, 5]], "output": [[9, 9, 0, 0, 0, 0, 0, 0, 0, 9], [9, 9, 0, 0, 0, 0, 0, 0, 0, 9], [0, 0, 8, 8, 0, 0, 0, 0, 0, 8], [0, 0, 8, 8, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 7, 7, 0, 0, 0, 7], [0, 0, 0, 0, 7, 7, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 6, 6, 0, 6], [0, 0, 0, 0, 0, 0, 6, 6, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 5, 5], [9, 9, 8, 8, 7, 7, 6, 6, 5, 5]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 9], [5, 6, 6, 7, 7, 7, 8, 9, 9, 9]], "output": [[5, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 6, 6, 0, 0, 0, 0, 0, 0, 6], [0, 6, 6, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 7, 7, 7, 0, 0, 0, 7], [0, 0, 0, 7, 7, 7, 0, 0, 0, 7], [0, 0, 0, 7, 7, 7, 0, 0, 0, 7], [0, 0, 0, 0, 0, 0, 8, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 9, 9, 9], [0, 0, 0, 0, 0, 0, 0, 9, 9, 9], [5, 6, 6, 7, 7, 7, 8, 9, 9, 9]]}, {"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [8, 8, 4, 4, 4, 5, 5, 3, 3, 3]], "output": [[8, 8, 0, 0, 0, 0, 0, 0, 0, 8], [8, 8, 0, 0, 0, 0, 0, 0, 0, 8], [8, 8, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 4, 4, 4, 0, 0, 0, 0, 4], [0, 0, 4, 4, 4, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 5, 5, 0, 0, 5], [0, 0, 0, 0, 0, 5, 5, 0, 0, 5], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3], [8, 8, 4, 4, 4, 5, 5, 3, 3, 3]]}]
data_test=[{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 7], [3, 3, 4, 6, 6, 6, 9, 9, 7, 7]], "output": [[3, 3, 0, 0, 0, 0, 0, 0, 0, 3], [3, 3, 0, 0, 0, 0, 0, 0, 0, 3], [3, 3, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 4, 0, 0, 0, 0, 0, 0, 4], [0, 0, 4, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 6, 6, 6, 0, 0, 0, 6], [0, 0, 0, 6, 6, 6, 0, 0, 0, 6], [0, 0, 0, 0, 0, 0, 9, 9, 0, 9], [0, 0, 0, 0, 0, 0, 0, 0, 7, 7], [3, 3, 4, 6, 6, 6, 9, 9, 7, 7]]}]

#Describe KG into grid
def make_example(input,output):
    x=torch.LongTensor(input)
    y=torch.LongTensor(output)
    x=F.one_hot(x,num_classes=10).float()
    H,W,_=x.shape
    i=torch.arange(10).eq(0).float().view(1,1,-1).repeat(H,W,1) #special indicator for 0
    x=torch.stack((x,i),dim=-1).float()
    return x,y

X_train=[]
Y_train=[]
for i,ex in enumerate(data_train):
    x,y=make_example(ex['input'],ex['output'])
    X_train.append(x.cuda())
    Y_train.append(y.cuda())

X_test=[]
Y_test=[]
for i,ex in enumerate(data_test):
    x,y=make_example(ex['input'],ex['output'])
    X_test.append(x.cuda())
    Y_test.append(y.cuda())

net=einnet(ninput=2,nh0=16,nh=16,noutput=1,nstacks=2,pool=einpool_aab()).cuda()

def forward(X,Y):
    s=net(X).squeeze(-1)
    loss=F.cross_entropy(s.view(-1,s.shape[-1]),Y.view(-1))
    _,pred=s.max(dim=-1)
    acc=pred.eq(Y).float().mean()
    return loss,pred,acc

opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)

loss=[]
for i in range(2000):
    net.zero_grad()
    for j in range(len(X_train)):
        loss_j,_,_=forward(X_train[j],Y_train[j])
        loss_j.backward()
        loss.append(float(loss_j.data))
    
    opt.step()
    loss_i=sum(loss)/len(loss)
    print('iter %d, loss %f   '%(i,loss_i),end='\r')
    
    if i%100==0:
        with torch.no_grad():
            for j in range(len(X_test)):
                loss_eval,pred,acc=forward(X_test[j],Y_test[j])
                print('iter %d, loss %f, loss_eval %f, acc %f,   '%(i,loss_i,loss_eval,acc))
                print(pred.view(-1).tolist())
            
            loss=[]

```



Arguably, hardcoding symmetry constraints specific to a ARC-AGI task can be considered cheating as much as writing a python program to solve the task. But maybe one day just like program synthesis, the LLMs may be able to generate neural network architectures that solve those tasks, so who knows?


There have you, we've designed and verified various kinds of permutation symmetric network from the basic principles. Building permutation symmetry into neural network architecture helps make learning highly efficient, sometimes solving tasks from as few as 3 examples. 

The same bottom-up approach is applicable to many other types of symmetries, such as translation, scale, rotation. In fact, designing equivariant and invariant networks has been a hot topic. In addition to the EMLP architecture, many network designs for specific types of permutation symmetry have also been proposed, such as <d-cite key="navon2023equivariant"></d-cite> and <d-cite key="zhou2024permutation"></d-cite>. 
In fact, the matrix pseudo inverse and knowledge graph completion use cases in our post are in part inspired by the <d-cite key="wangsmpe"></d-cite> work and comments from OpenReview. We'd like to encourage curious readers to read those papers as well.


