---
layout: distill
title: Sample Blog Post
description: Our blog post will focus on \textbf{optimizing the serving of large-scale language models in distributed systems}, with an emphasis on improving memory efficiency and reducing latency. We will discuss strategies for optimizing memory layout, execution scheduling, and batching to enhance the throughput of AI model inference. Additionally, the post will examine the role of SmartNICs in offloading certain tasks in data centers, reducing CPU load, and improving communication between compute nodes. Through this, we aim to highlight the importance of networking optimizations for efficient ML serving in real-world systems.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Bae Junhyeong
    url: "https://github.com/20190511"
    affiliations:
      name: POSTECH, Pohang University of Science and Technology
  - name: Gang Sungwook
    url: "https://en.wikipedia.org/wiki/Gang_Sungwook"
    affiliations:
      name: POSTECH, Pohang University of Science and Technology

# must be the exact same name as your blogpost
bibliography: 2025-04-28-Final.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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

## Abstract
- "what problem is this work trying to tackle?"
- "how new is this effort?" (소개, 개요요)
## Background

## Main 설명 (제목 바꿀 것, 여러개 있어도됨)
- "what contributions did this work make, and what impact should this work have?"
- "how new is this effort?"

## Results (논문 실험결과 담아도되고 안담아도 되고..)
- "what are the limitations of this work?"
## Conclusion
- 어떤 노력이 있었으며, 어떤식으로 최적화할 것인가?

## Citation (bibs 로 올릴 것이니까 생각 나는 논문들만 정리해둘것.)