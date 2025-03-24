---
layout: about
title: about
permalink: /about/
nav: true
nav_order: 1
# subtitle: 

profile:
  align: right
  image: 
  image_circular: false # crops the image to make it circular
  address: 

news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page
---

# Announcing Accepted Blogposts

>The International Conference on Learning Representations (ICLR) is proud to announce the fourth edition of its Blogposts Track, our 2022 established publication venue to address the lack of up-to-date educational, critical, and insightful communications in the fast-paced fields of Machine Learning and Artificial Intelligence. The Blogposts Track offers a unique venue to discuss recent developments, research directions, and reflect on past work while offering accessible education.
>Over the past years, the Blogposts Track has seen increasing engagement, reinforcing its role in supporting an open exchange of ideas within the machine learning community. By providing a platform for blog post publications, ICLR continues to support knowledge exchange in our research community.
>Thanks to the growing interest in this format, the track received **96 submissions**, with **48** exceptional blog posts accepted after a double-blind review. These posts explore a variety of topics in machine learning, offering unique educational, critical, and reflective positions.
>
>Among the large list of this year's notable contributions, we would like to highlight
>- ***Flow With What You Know*** - provides an accessible introduction to flow-matching and rectified flow models by exploring the physical representation of the involved processes.
>
> as best blog post, with
>- ***Intricacies of Feature Geometry in Large Language Models*** - reflects on the geometry of a language model's embedding space. While recent works found structured geometry corresponding to concepts, this blog posts uncovers that these trivially arise in high-dimensional spaces.
>- ***Understanding Model Calibration - A gentle introduction and visual exploration of calibration and the expected calibration error (ECE)*** - provides a gentle introduction into model calibration and discusses the use of the commonly used expected calibration error as evaluation metric.
>
>as close runner-ups. These contributions underscore the Blogposts Track mission to promote meaningful, high-quality scientific dialogue and educational communication across machine learning.
>
>To encourage engagement, the accepted blog posts will be featured in a poster session at ICLR in Singapore from April 24th to April 28th. These "Affinity Posters" sessions will provide a forum for direct discussions between authors and attendees.
>All accepted blog posts are available on the Blog page of the track's website. 
>
>Blogposts Track Program Chairs
>
>- David Dobre, Nicholas Gao, Jonas Köhler, Leo Schwinn, Sophie Xhonneux

## Key Dates

- **Abstract & Submission deadline**: ~~Nov 15th~~ **Extended to** Nov 22nd 23:59 AOE, 2024 
  - *OpenReview and any modifications to your blog post, via a pull request on GitHub*.
- **Meta Review Deadline**: January 20th, 2025
- **Decision Notification**: January 22nd, 2025
- **Camera-ready merge**: March 31st, 2024

## Contents

- [ICLR 2025 Blogposts Track](#iclr-2025-blogposts-track)
- [Submissions](#submissions)
- [Organizers](#organizers)




# ICLR 2025 Blogposts Track

The Machine Learning community is currently experiencing a [reproducibility crisis](https://neuripsconf.medium.com/designing-the-reproducibility-program-for-neurips-2020-7fcccaa5c6ad) and a reviewing crisis [[Littman, 2021]](#Litt). Because of the highly competitive and noisy reviewing process of ML conferences [[Tran et al., 2020]](#Tran), researchers have an incentive to oversell their results, slowing down the progress and diminishing the integrity of the scientific community. Moreover with the growing number of papers published and submitted at the main ML conferences [[Lin et al., 2020]](#Lin), it has become more challenging to keep track of the latest advances in the field.

Blog posts are becoming an increasingly popular and useful way to talk about science [[Brown and Woolston, 2018]](#Brow). They offer substantial value to the scientific community by providing a flexible platform to foster open, human, and transparent discussions about new insights or limitations of a scientific publication. However, because they are not as recognized as standard scientific publications, only a minority of researchers manage to maintain an active blog and get visibility for their efforts. Many are well-established researchers ([Francis Bach](https://francisbach.com/), [Ben Recht](https://www.argmin.net/), [Ferenc Huszár](https://www.inference.vc/), [Lilian Weng](https://lilianweng.github.io/lil-log/)) or big corporations that leverage entire teams of graphic designers designer and writers to polish their blogs ([Facebook AI](https://ai.facebook.com/blog/?page=1), [Google AI](https://ai.googleblog.com/), [DeepMind](https://deepmind.com/blog), [OpenAI](https://openai.com/blog/)). As a result, the incentives for writing scientific blog posts are largely personal; it is unreasonable to expect a significant portion of the machine learning community to contribute to such an initiative when everyone is trying to establish themselves through publications.

**Submit** your blogpost on [Openreview](https://openreview.net/group?id=ICLR.cc/2025/BlogPosts).

## A Call for Blog Posts

Last year, we ran the **third** iteration of the [Blogpost track](https://iclr-blogposts.github.io/2024/about) at ICLR 2024! 
It was very successful, with accepted posts presented in person at the main conference.
<!-- Our goal is to create a formal call for blog posts at ICLR to incentivize and reward researchers to: -->
We invite all researchers and practitioners to submit a blog post which:

1. Reviews past work and summarize the outcomes, develop new intuitions, or highlight some shortcomings. 
2. Presents novel perspectives or interpretations of existing machine learning concepts or techniques.
3. Discusses important issues in machine learning, such as reproducibility, from a novel perspective.
4. Analyzes the societal implications of recent advancements in machine learning and AI.
5. Showcases cool research ideas that you tried but did not work out.

**We will not consider politically motivated blogposts for publication.**

If you are unsure about the content of your post you can reach us at [iclr-blogpost-track@googlegroups.com](mailto:iclr-blogpost-track@googlegroups.com).

Past blog posts can be accessed here: [2022](https://iclr-blog-track.github.io/home/#accepted-posts), [2023](https://iclr-blogposts.github.io/2023/about#accepted-posts), [2024](https://iclr-blogposts.github.io/2024/about#spotlight).


<!-- A very influential initiative of this kind happened after the Second World War in France. Because of the lack of up-to-date textbooks, a collective of mathematicians under the pseudonym Nicolas Bourbaki [[Halmos 1957]](#Halm), decided to start a series of textbooks about the foundations of mathematics [[Bourbaki, 1939]](#Bour). 
In the same vein, we aim to provide a new way to summarize scientific knowledge in the ML community.  -->

#### Conflict of interest 

The authors of the blog posts will have to declare their conflicts of interest (positive or negative) with the paper (and the paper's authors) they write about. Conflicts of interest include:
-   Recent collaborators (less than 3 years)
-   Current institution -- reviewers will be asked to judge if the submission is sufficiently critical and objective of the papers addressed in the blog post.  
-  **Blog Posts must not be used to highlight or advertise past publications of the *authors or their lab***.

We will only ask the authors to report if they have a conflict of interest. If so, reviewers will be asked to judge if the submission is sufficiently critical and objective of the papers addressed in the blog post. 



## Publication 

The posts will be created and published under a unified template; see [the submission instructions]({{ '/submitting' | relative_url }}) and the [sample post]({% post_url 2025-04-28-distill-example %}) hosted on the blog of this website.

#### Poster

Additionally, accepted posts will have the option to present their work as a poster during the main poster session. For more information about the main poster session (time, poster format, etc.) please refer to the ICLR homepage.

## Submissions

Our goal is to avoid heavily engineered, professionally-made blog posts ---Such as the “100+ hours” mentioned as a standard by the [Distill guidelines](https://distill.pub/journal/)---to entice ideas and clear writing rather than dynamic visualizations or embedded javascript engines.
Please check our [submission instructions]({{ '/submitting' | relative_url }}) for more details.
We accept submissions in both Markdown and HTML. We believe this is a good trade-off between complexity and flexibility. 

**Submit** your blogpost on [Openreview](https://openreview.net/group?id=ICLR.cc/2025/BlogPosts&referrer=%5BHomepage%5D(%2F))

## Contact

For any technical issues with the blog post repository (for example, blog posts not displaying correctly or issues while following the [submission instructions](https://iclr-blogposts.github.io/2025/submitting/#creating-a-blog-post)), please open an [issue in our github repository](https://github.com/iclr-blogposts/2025/issues).

For other inquiries, reach us via email at: [iclr-blogpost-track@googlegroups.com](mailto:iclr-blogpost-track@googlegroups.com)

## Organizers

<div class="row row-cols-2 projects pt-3 pb-3">
  {% include people_horizontal.html name="Leo Schwinn" affiliation="Technical University of Munich" url="https://schwinnl.github.io//" img="assets/img/organizers/ls.jpg" %}
  {% include people_horizontal.html name="David Dobre" affiliation="Mila, Université de Montréal" url="" img="assets/img/organizers/dd.jpg" %}
  {% include people_horizontal.html name="Nicholas Gao" affiliation="Techical University of Munich" url="" img="assets/img/organizers/ng.jpg" %}
  {% include people_horizontal.html name="Sophie Xhonneux" affiliation="Mila, Université de Montréal" url="" img="assets/img/organizers/sx.jpg" %}
  {% include people_horizontal.html name="Jonas Köhler" affiliation="CuspAI" url="" img="assets/img/organizers/jk.jpg" %}
</div>

## References

<a name="Litt">Michael L Littman. Collusion rings threaten the integrity of computer science research. Communications of the ACM, 2021.</a>

<a name="Tran">David Tran, Alex Valtchanov, Keshav Ganapathy, Raymond Feng, Eric Slud, Micah Goldblum, and Tom Goldstein. An open review of OpenReview: A critical analysis of the machine learning conference review process. arXiv, 2020. </a>

<a name="Lin">Hsuan-Tien Lin, Maria-Florina Balcan, Raia Hadsell, and Marc’Aurelio Ranzato. What we learned from NeurIPS 2020 reviewing process. Medium https://medium.com/@NeurIPSConf/what-we-learned-from-neurips-2020-reviewing-process-e24549eea38f, 2020. </a>

<a name="Brow">Eryn Brown and Chris Woolston. Why science blogging still matters. Nature, 2018.</a>

<a name="Halm">Paul R Halmos. Nicolas Bourbaki. Scientific American, 1957.<a>

<a name="Bour">Nicolas Bourbaki. Elements of mathematics. Éditions Hermann, 1939.</a>


