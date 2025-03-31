---
layout: distill
title: 'Factual Context Validation and Simplification: A Scalable Method to Enhance GPT Trustworthiness and Efficiency'
description: 'As the deployment of Large Language Models (LLMs) like GPT expands across domains, mitigating their susceptibility to factual inaccuracies or hallucinations becomes crucial for ensuring reliable performance. This blog post introduces two novel frameworks that enhance retrieval-augmented generation (RAG): one uses summarization to achieve a maximum of 57.7% storage reduction, while the other preserves critical information through statement-level extraction. Leveraging DBSCAN clustering, vectorized fact storage, and LLM-driven fact-checking, the pipelines deliver higher overall performance across benchmarks such as PubMedQA, SQuAD, and HotpotQA. By optimizing efficiency and accuracy, these frameworks advance trustworthy AI for impactful real-world applications.'
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Tianyi Huang
    url: "https://www.linkedin.com/in/tianyi-huang-36ba49280/"
    affiliations:
      name: App Inventor Foundation
      url: https://www.appinventorfoundation.org/

bibliography: 2025-04-28-factual-validation-simplification.bib

toc:
  - name: Introduction
  - name: The Challenge of Factual Accuracy in AI
    subsections:
    -   name: '1. Dynamic and Unstructured Contexts'
    -   name: '2. Compounded Errors in Reasoning'
  - name: Limitations of Existing Approaches
    subsections:
    -   name: Fine-Tuning
    -   name: Retrieval-Augmented Generation (RAG)
    -   name: Post-Hoc Validation
  - name: Theoretical Foundations
    subsections:
    -   name: 'Factual Validation: Core Principles'
    -   name: The Role of Vectorization
    -   name: Combining Granularity and Vectorization
  - name: 'Our Proposal: Fact-Based Validation for GPT Systems'
  - name: Implementation Details
    subsections:
    -   name: New Data Input System
    -   name: LLM Validity Judger
  - name: Experimentation and Results
    subsections:
    -   name: Factual Accuracy
    -   name: RAG Effectiveness
    -   name: Storage Efficiency
    -   name: Validity Judgment with Chain of Thought
  - name: 'Alternative Solution: Statement Extraction'
    subsections:
      - name: Extended Benchmarks
      - name: Realistic Usage
  - name: Broader Implications
  - name: Limitations and Future Work
  - name: Reproducibility
  - name: Conclusion
  
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
  .box-important {
    background: linear-gradient(135deg, #f8f9fa, #e6f7f9);
    border: 1px solid #76b7c7;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    color: #2a3e4b;
    line-height: 1.7;
    border-radius: 0px;
  }
  .box-two {
    border-left: 5px solid #4A90E2;
    background: rgba(74, 144, 226, 0.1);
    padding: 15px;
    margin: 20px 0;
    font-family: "Roboto", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    border-radius: 5px;
  }
  .matrix-box {
    border: 1px solid #d3d3d3;
    border-radius: 10px;
    padding: 15px;
    margin: 20px auto;
    background-color: #fefefe;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    display: inline-block;
  }
  .box-future {
    border: 2px solid #007BFF;
    background: #F8FAFF;
    padding: 25px;
    margin: 20px 0;
    font-family: "Roboto", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    border-radius: 8px;
    box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.08);
  }
  summary-math {
    text-align: center;
    color: black;
  }
  [data-theme="dark"] summary-math {
    color: white;
  }
  d-article {
    overflow-x: visible;
  }
  pre {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

---

## Introduction
Large Language Models (LLMs), such as GPT, have advanced natural language processing by offering incredible fluency and adaptability <d-cite key="brown2020"></d-cite>. Yet, these models are susceptible to **"hallucinations"**---outputs that are linguistically coherent but factually incorrect <d-cite key="ji2022, maynez2020"></d-cite>. This issue arises because LLMs optimize for contextual fluency rather than accuracy, a trade-off that becomes increasingly problematic in applications where precision or reliability is necessary <d-cite key="holtzman2019, esteva2019, ullah2024"></d-cite>.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/hallucination.png' | relative_url }}" style="width: 85%;">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 1.</strong> An example of hallucination in ChatGPT. This figure demonstrates a fabricated response confidently generated by ChatGPT, showing how large language models (LLMs) can produce coherent but factually incorrect outputs. The highlighted sections indicate fictional and misleading content, emphasizing the challenges in ensuring factual accuracy in LLM-generated responses.
    </figcaption>
</figure>

Existing solutions, including fine-tuning, retrieval-augmented generation (RAG), and post-hoc validation, address specific aspects of this problem but often at the cost of scalability or computational efficiency <d-cite key="guu2020"></d-cite>. These methods frequently fail to account for the growing demand to process unstructured data efficiently while maintaining factual accuracy.

In this blog post, we propose an enhancement to RAG pipelines that integrates LLM summarization, clustering with DBSCAN, and vectorized fact storage to tackle the challenges of managing unstructured data <d-cite key="ester1996"></d-cite>. Our framework further innovates a fact-verification system that splits LLM responses into granular components for individual fact-level validation. By processing expansive datasets into concise, verifiable formats, this approach achieves significant improvements in storage efficiency without sacrificing factual accuracy. Evaluated on the PubMedQA dataset, the proposed method demonstrates how preprocessing and factual validation can enhance RAG workflows, improve scalability, and maintain retrieval precision, offering new possibilities for real-world applications <d-cite key="jin2019"></d-cite>.

## The Challenge of Factual Accuracy in AI

<span style="font-size: 20px; font-weight: bold;">Hallucinations and Trust Deficit</span>

Hallucinations in LLMs arise from their probabilistic design, which prioritizes predicting plausible sequences over verifying factual accuracy <d-cite key="radford2019"></d-cite>. This limitation is magnified by two interconnected factors:

<span style="font-size: 20px; font-weight: bold;">1. Dynamic and Unstructured Contexts</span>

LLMs are trained on vast corpora of text that often lack semantic organization. Moreover, the facts embedded within these datasets are static—reflecting the state of the world at the time of training. However, real-world knowledge evolves dynamically, leaving models unable to adapt without post-training updates <d-cite key="mousavi2024"></d-cite>. 

#### Formalizing the Knowledge Divergence
Let the model's knowledge at time $$t$$ be denoted as $$K(t)$$. We can express the knowledge state as:
$$
K(t) = \textcolor{#5DADEC}{K(t_0)} + \textcolor{#E67E22}{\Delta K(t)},
$$
where:
- $$\textcolor{#5DADEC}{K(t_0)}$$ represents the knowledge at the time of training.
- $$\textcolor{#E67E22}{\Delta K(t)}$$ captures incremental updates post-deployment.

In most deployed LLMs:
$$
\textcolor{#E67E22}{\Delta K(t)} = 0,
$$
indicating that the knowledge remains frozen at $$\textcolor{#5DADEC}{K(t_0)}$$. Over time, as $$t \to \infty$$, the divergence between $$K(t)$$ and real-world knowledge widens, resulting in increasingly outdated or inaccurate outputs.
<div class="box-important" markdown="1">
#### Real-World Implications
This static nature is especially problematic in domains such as:

- Medicine, where evolving treatment protocols and research are critical <d-cite key="ji2022"></d-cite>.
- Law, where changes in regulations or precedents can render older knowledge obsolete.

Without mechanisms to dynamically update $$\Delta K(t)$$, LLMs fail to provide accurate, timely information in these high-stakes environments.
</div>

<span style="font-size: 20px; font-weight: bold;">2. Compounded Errors in Reasoning</span>

In multi-step reasoning, LLMs encounter a **cascading error problem**, where inaccuracies at each step propagate and amplify in subsequent steps <d-cite key="bender2021"></d-cite>. 

#### Quantifying Error Propagation
Let $$P(x_n)$$ denote the probability of correctness at reasoning step $$n$$. The overall probability of correctness across $$N$$ steps is given by:
$$
P_{\text{total}} = \prod_{n=1}^N \textcolor{#0f941b}{P(x_n)}.
$$
Since $$\textcolor{#0f941b}{P(x_n)} < 1$$ in most real-world scenarios, $$P_{\text{total}}$$ decays exponentially as $$N$$ increases. This exponential decay illustrates how longer reasoning chains magnify errors, making LLMs less reliable for complex tasks <d-cite key="wei2022"></d-cite>.

#### Modeling Error Interactions

Error propagation can also be modeled using second-order interactions through the Hessian matrix <d-cite key="bishop1992"></d-cite>:
$$
H_{ij} = \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_i \partial x_j}. 
$$

To represent this interaction in matrix form, the Hessian matrix can be visualized as:

<div class="matrix-box">
$$
H = 
\begin{bmatrix}
\frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_1^2} & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_1 \partial x_n} \\
\frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_2 \partial x_1} & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_2^2} & \cdots & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_n \partial x_1} & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 \textcolor{#E74C3C}{E}}{\partial x_n^2}
\end{bmatrix}
$$
</div>

where:
- $$H_{ij}$$ quantifies how an error in dimension $$x_i$$ interacts with errors in dimension $$x_j$$.
- $$\textcolor{#E74C3C}{E}$$ is the error function dependent on reasoning dimensions $$x_i$$.

This matrix representation highlights the symmetry of the Hessian ($$H_{ij} = H_{ji}$$), demonstrating how second-order effects capture complex interactions between dimensions. By modeling these interactions, we can better understand how errors propagate and amplify through reasoning pathways. This indicates that inaccuracies are not isolated; instead, they cascade through interconnected reasoning pathways, compounding the overall error and making multi-step reasoning increasingly unreliable in LLMs.
## Limitations of Existing Approaches

<span style="font-size: 20px; font-weight: bold;">Fine-Tuning</span>

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/finetuning.png' | relative_url }}">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 2.</strong> Workflow showing LLM fine-tuning with domain-specific data, integrating fine-tuned datasets to produce specialized responses.
    </figcaption>
</figure>

Fine-tuning involves retraining LLMs on domain-specific datasets, improving their factual accuracy within specific contexts <d-cite key="howard2018"></d-cite>. However, it has inherent trade-offs:

- High resource requirements make it impractical for many organizations and individuals <d-cite key="zhang2022"></d-cite>.
- Static knowledge freezes models in time, necessitating constant retraining to stay relevant <d-cite key="bommasani2021"></d-cite>.
- Specialization often comes at the expense of general-purpose utility, limiting versatility <d-cite key="ruder2019"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Retrieval-Augmented Generation (RAG)</span>

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/rag.png' | relative_url }}">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 3.</strong> Workflow showing external data retrieval and integration with the LLM to generate grounded responses.
    </figcaption>
</figure>

RAG pipelines integrate external knowledge retrieval into the LLM workflow, providing factual grounding by leveraging curated databases or search engines <d-cite key="ibm2023"></d-cite>. While effective in many cases, RAG systems face challenges:

- Retrieved information may be incomplete, outdated, or biased, directly affecting output reliability <d-cite key="lewis2020"></d-cite>.
- RAG systems retrieve information but do not inherently validate it, leaving unresolved conflicts between retrieved facts and the LLM’s internal predictions <d-cite key="karpukhin2020"></d-cite>.
- As query complexity increases, retrieved information can become overly sparse, and the increasing contextual length can diminish its enhancement for the LLM, ultimately restricting scalability <d-cite key="laban2024"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Post-Hoc Validation</span>

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/posthoc.png' | relative_url }}">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 4.</strong> Workflow showing validation of LLM-generated responses against external datasets, with flagged issues reprocessed for accuracy.
    </figcaption>
</figure>

Post-hoc validation attempts to correct inaccuracies after generation by cross-referencing outputs with external datasets or models <d-cite key="zhong2024"></d-cite>. While valuable, it suffers from inefficiencies:

- It introduces latency, making it unsuitable for real-time applications.
- Validating every output, even when unnecessary, wastes computational resources <d-cite key="guu2020"></d-cite>.
- It provides limited feedback for refining the underlying generative process, addressing symptoms but not root causes <d-cite key="li2024"></d-cite>.

## Theoretical Foundations

<span style="font-size: 20px; font-weight: bold;">Factual Validation: Core Principles</span>

Factual validation ensures that LLM outputs are verifiable and contextually accurate. This approach is guided by two main principles:

- **Granularity**: Responses are decomposed into discrete factual units for precise validation, reducing the likelihood of undetected inaccuracies <d-cite key="maynez2020"></d-cite>.

  Mathematically, a generated response $$R$$ is represented as:
  $$
  R = \{\textcolor{#1f77b4}{F_1}, \textcolor{#ff7f0e}{F_2}, \dots, \textcolor{#2ca02c}{F_n}\},
  $$
  where each $$F_i$$ represents a discrete factual unit, such as a statement or claim.  

  #### Example:
  
  $$
  R = \{\text{"Paris is the capital of France"}, \text{"The Eiffel Tower is in Paris"}\}.
  $$

  Validation is performed by comparing each $$F_i$$ against a knowledge base $$K$$, which contains validated information:
  
  $$
  K = \{\text{"Paris is the capital of France"}, \text{"The Eiffel Tower is in Paris"}\}.
  $$
  
  A factual unit $$F_i$$ is considered valid if it exists in $$K$$:
  $$
  V(F_i) = 
  \begin{cases} 
  1 & \text{if } F_i \in K, \\
  0 & \text{otherwise.}
  \end{cases}
  $$

  The overall validation score for the response is calculated as:
  $$
  V(R) = \frac{\sum_{i=1}^{n} V(F_i)}{n},
  $$
  where $$V(R) \in [0, 1]$$ quantifies the proportion of factual units in $$R$$ that are verified by $$K$$.

- **Scalability**: Validation systems leverage vectorized representations to efficiently compare outputs against large, structured knowledge bases <d-cite key="karpukhin2020"></d-cite>. Instead of exact string matching, factual units $$F_i$$ are mapped into a high-dimensional vector space for efficient comparisons. 

<span style="font-size: 20px; font-weight: bold;">The Role of Vectorization</span>

Vectorization encodes atomic facts into high-dimensional representations (**embeddings**) that facilitate efficient storage, retrieval, and comparison <d-cite key="johnson2017"></d-cite>.  

Each factual unit $$F_i$$ is transformed into an embedding:
$$
\mathbf{v}_i = \textcolor{#1f77b4}{\text{Encode}(F_i)},
$$
where $$\text{Encode}(\cdot)$$ maps textual data to numerical vectors.

#### Example:

$$
F_i = \text{"Paris is the capital of France"}
$$
$$
\mathbf{v}_i = [0.12, 0.45, 0.67, \dots, 0.89]
$$

The knowledge base $$K$$ is also represented as a set of embeddings:
$$
K = \{\mathbf{k}_1, \mathbf{k}_2, \dots, \mathbf{k}_m\}.
$$

To validate $$F_i$$, the similarity between $$\mathbf{v}_i$$ and $$\mathbf{k}_j$$ is measured using cosine similarity:
$$
\text{Sim}(\mathbf{v}_i, \mathbf{k}_j) = \frac{\mathbf{v}_i \cdot \mathbf{k}_j}{\|\mathbf{v}_i\| \|\mathbf{k}_j\|}.
$$

**Cosine Similarity**: Determines how closely aligned two vectors are in high-dimensional space (range: -1 to 1):
- $$1$$: Identical vectors.
- $$-1$$: Completely opposite vectors.

A factual unit $$F_i$$ is considered valid if:
$$
\max_{j} \text{Sim}(\mathbf{v}_i, \mathbf{k}_j) \geq \tau,
$$
where $$\tau$$ is a predefined similarity threshold <d-cite key="reimers2019"></d-cite>. This ensures that validation considers **semantic similarity** rather than exact matches.

<span style="font-size: 20px; font-weight: bold;">Combining Granularity and Vectorization</span>

Factual validation combines the precision of granularity with the efficiency of vectorization to determine the validity of responses. The process involves:

1. **Decompose Response**: Break $$R$$ into:
   $$
   R = \{\textcolor{#1f77b4}{F_1}, \textcolor{#ff7f0e}{F_2}, \dots, \textcolor{#2ca02c}{F_n}\}.
   $$

2. **Encode Units**: Transform each $$F_i$$ into its high-dimensional embedding:
   $$
   \mathbf{v}_i = \text{Encode}(F_i).
   $$

3. **Compare with Knowledge Base**: Use cosine similarity to validate $$\mathbf{v}_i$$ against $$K$$:
   $$
   \max_{j} \text{Sim}(\mathbf{v}_i, \mathbf{k}_j) \geq \tau.
   $$

The overall validation score is computed as:
$$
V(R) = \frac{\sum_{i=1}^{n} \mathbb{1} \left( \max_{j} \text{Sim}(\mathbf{v}_i, \mathbf{k}_j) \geq \tau \right)}{n},
$$
where $$\mathbb{1}(\cdot)$$ is the indicator function that outputs 1 if the condition is satisfied and 0 otherwise.

**This approach ensures**:
1. **Precision** at the **component level** (granularity).  
2. **Scalability** across **large knowledge bases** (vectorization) <d-cite key="guu2020"></d-cite>.

## Our Proposal: Fact-Based Validation for GPT Systems
To address the challenges of ensuring factual accuracy in GPT systems, we propose a framework that preprocesses data into structured and verifiable units for use in fact verification with the aim to enhance the reliability of LLM-generated responses while optimizing storage and retrieval efficiency.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/methodology.png' | relative_url }}" style="width: 100%;">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 5.</strong> This diagram illustrates the workflow of the LLM Validity Judger. The system fetches relevant facts from a vector database, generates responses based on this context, and simplifies outputs into factual statements. It validates these statements by comparing them with the vector database, generating an overall validity score and explanation. The process repeats with feedback until a valid result is achieved. The diagram also includes the New Data Input System, which independently adds summarized and verified facts into the vector database, ensuring its relevance and accuracy. For clarity, the fetching system mirrors standard vector database retrieval processes.
    </figcaption>
</figure>

<span style="font-size: 20px; font-weight: bold;">System Architecture</span>

- **New Data Input System**: Fact Extraction and Storage  
  - **Input**: Multiple text chunks containing factual information.  
  - **Output**: Extracted facts embedded as vector representations, stored in a vector database.

- **LLM Validity Judger**: Prompt and Response Validation  
  - **Input**: A GPT prompt and its corresponding response.  
  - **Output**: An overall validity rating between 0 and 1 indicating the accuracy of the prompt/response pair, along with optional reasoning.

## Implementation Details
<span style="font-size: 25px; font-weight: bold;">New Data Input System</span>

**Fetching Contexts**: All contexts are fetched from the PubMedQA dataset to serve as the input data for preprocessing <d-cite key="jin2019"></d-cite>.
 ```python
  dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")
  data = dataset.select_columns(["question", "context", "pubid"])
 ```
**Summarization and Compression**: Each context is summarized using `GPT-4o-mini`, compressing the content into concise paragraphs. This step ensures efficient downstream processing and storage <d-cite key="gpt4omini"></d-cite>.
 ```python
  new_contexts = [
    str(x)
    for x in await async_gpt_calls(
      [get_summarize_prompt(context) for context in raw_contexts]
    )
  ]
 ```
**Embedding with OpenAI Models**: The summarized contexts are embedded into high-dimensional vector representations using OpenAI's `text-embedding-3-large` model <d-cite key="openaidocs"></d-cite>.
 ```python
  result = client.embeddings.create(
    input=text,
    model="text-embedding-3-large",
  )
  return EmbeddingResponse(
      result.data[0].embedding,
      result.model,
      result.usage,
  )
 ```
**Clustering with DBSCAN**: The embeddings are grouped into clusters using DBSCAN, which identifies similar contexts without requiring predefined categories. This step is critical for organizing related data efficiently <d-cite key="ester1996"></d-cite>.
 ```python
  def cluster(vectors: list[Vector], eps=45.0, min_samples=1) -> list[list[str]]:
    data = StandardScaler().fit_transform(
      np.array([vector["vector"] for vector in vectors])
    )
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(data)
    clusters: dict[str, list[str]] = {}
    for item, label in zip(vectors, db.labels_):
      clusters.setdefault(label, []).append(item["id"])
    return [vectors for label, vectors in clusters.items() if label != -1]
 ```
**Context Combination and Summarization**: Context summaries within each cluster are combined and re-summarized into unified representations using `GPT-4o-mini`, further reducing redundancy and optimizing coherence <d-cite key="gpt4omini"></d-cite>.
 ```python
  def combine_and_summarize(cluster: list[str]):
    combined_context = " ".join(cluster)
    return summarize_context(combined_context)
 ```
**Final Embedding and Storage**: The new summaries are re-embedded with `text-embedding-3-large` <d-cite key="openaidocs"></d-cite>. These embeddings, along with any unclustered individual summaries, are stored in PineconeDB for retrieval <d-cite key="pinecone"></d-cite>.
 ```python
  vectors = [
    {"id": id, "values": embedding.vector, "metadata": {"content": content}}
    for id, embedding, content in zip(ids, embeddings, contexts)
  ]
  index.upsert(vectors=vectors, namespace="pubmed_summarized")
 ```

<span style="font-size: 25px; font-weight: bold;">LLM Validity Judger</span>

**Decomposing Responses**: GPT processes the prompt and its corresponding response, breaking them into discrete factual statements. Each statement is assigned an importance score (0–1), indicating its relevance to the prompt.
 ```python
  def decompose_response(response: str) -> list[dict]:
    sentences = response.split(".")
    return [{"statement": s.strip(), "importance": 0.5} for s in sentences if s.strip()]
 ```
**Proximity Search**: For each statement, the system retrieves nearby facts from the vector database through a semantic proximity search <d-cite key="johnson2017"></d-cite>. These facts serve as the context for validation.
 ```python
  def proximity_search(statement_embedding, top_k=5):
    response = index.query(vector=statement_embedding, top_k=top_k)
    return response.matches
 ```
**Validity Scoring**: Each statement is evaluated against its contextual facts using GPT, which assigns a validity score between 0 and 1 based on alignment. This evaluation determines how well the statement aligns with retrieved facts <d-cite key="kryscinski2019"></d-cite>.
 ```python
  def evaluate_validity(statement: str, context: str):
    prompt = f"Evaluate: '{statement}' based on: '{context}'. Assign a validity score (0-1)."
    response = client.chat.completions.create(model="gpt-4o-mini", prompt=prompt, max_tokens=10)
    return float(response.choices[0].text.strip())
 ```
**Weighted Average Calculation**: The validity scores are aggregated into a weighted average, where the weights are derived from the importance scores of the statements. This weighted score represents the overall factual accuracy of the response.
 ```python
  def calculate_weighted_average(scores: list[float], weights: list[float]):
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
 ```
**Explainability**: If required, GPT generates a rationale for the validity rating, outlining which facts were used, how alignment was determined, and any potential conflicts. This reasoning enhances interpretability and transparency.
 ```python
  def generate_explanation(statement: str, context: str):
    prompt = f"Explain validity of: '{statement}' in context: '{context}'"
    response = client.chat.completions.create(model="gpt-4o-mini", prompt=prompt, max_tokens=150)
    return response.choices[0].text.strip()
 ```

<div class="box-two" markdown="1">
#### Advantages of This System
The proposed system introduces several features that enhance the reliability and adaptability of existing RAG pipelines:

- **Innovative Design**: By integrating techniques like LLM summarization, DBSCAN clustering, and vectorized fact storage, the system introduces a new framework for precise fact validation and efficient data storage reduction.

- **Streamlined Architecture**: The system’s design supports simple integration with existing RAG systems while reducing storage needs and retrieval token usage. This adaptability ensures suitability and long-term applicability for large-scale applications across various domains.

- **Granular Fact Verification**: By decomposing responses into discrete factual units and validating each against a vectorized knowledge base, the system achieves significantly higher precision compared to traditional document- or paragraph-level validation methods, reducing inaccuracies and improving trustworthiness.
</div>

<span style="font-size: 20px; font-weight: bold;">Why Not KNN for Clustering?</span>

K-nearest neighbor (KNN) was considered for fact clustering but was ultimately not used due to its reliance on predefined parameters, such as the number of clusters or neighbors—constraints that are impractical for real-world datasets with high diversity <d-cite key="taunk2019"></d-cite>. In contrast, DBSCAN identifies clusters based on density, eliminating the need for prior knowledge of the dataset's structure <d-cite key="ester1996"></d-cite>. This makes DBSCAN more effective for processing unstructured or unknown data distributions. Additionally, KNN's approach risks significant data abstraction or loss when attempting to condense diverse facts into singular representations, worsening the system's ability to maintain factual granularity <d-cite key="cunningham2020"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Additional Note</span>

If the dataset is preprocessed to isolate one specific fact per entry, we recommend bypassing that step. Instead, the data can directly enter our model, which dynamically groups related facts. This approach preserves the richness of the dataset and streamlines the overall workflow.

## Experimentation and Results

We evaluated our proposed pipeline by benchmarking it against a **Traditional Retrieval-Augmented Generation (RAG) Pipeline** using the PubMedQA dataset <d-cite key="jin2019"></d-cite>. The evaluation focused on three key metrics: **factual accuracy, RAG effectiveness,** and **storage efficiency,** collectively measuring response quality, retrieval precision, and data storage optimization.

<span style="font-size: 20px; font-weight: bold;">Traditional RAG Pipeline</span>

The traditional RAG pipeline was tested under ideal conditions, embedding and retrieving the labeled contexts directly from the PubMedQA dataset <d-cite key="jin2019"></d-cite>. This setup provided perfect access to the correct answer contexts, offering a significant advantage. Despite this, our proposed pipeline—which applies summarization and compression—demonstrates performance comparable to this baseline, emphasizing its effectiveness.

The traditional pipeline workflow includes the following steps:
- Embed each context using OpenAI's `text-embedding-3-large` model.
- Store the embeddings in PineconeDB for retrieval.

<span style="font-size: 20px; font-weight: bold;">Comparative Metrics</span>

| **Metric**           | **Traditional Pipeline** | **Proposed Pipeline** | **Difference**            |
|-----------------------|-------------------------|-----------------------|---------------------------|
| Factual Accuracy      | **71.7%**               | 71.2%                 | -0.5%                    |
| RAG Effectiveness     | **99.2%**               | 98.9%                 | -0.3%                    |
| Storage Efficiency    | 1,351 KB                | **571 KB**            | -57.7% (Reduction)       |

**Table 1.** Comparison of factual accuracy, RAG effectiveness, and storage efficiency between the traditional pipeline and the proposed pipeline. The proposed pipeline achieves comparable performance in accuracy and effectiveness while significantly reducing storage requirements by 57.7%.

<span style="font-size: 20px; font-weight: bold;">Factual Accuracy</span>

We tested factual accuracy to measure how well the system addressed prompts by integrating all processes, including context retrieval, summarization, and LLM response generation. Using the PubMedQA dataset with 1,000 labeled context-question-answer groups, we queried each question and deemed responses correct if they matched the dataset’s answer (yes, no, or maybe) <d-cite key="jin2019"></d-cite>. The traditional pipeline achieved 71.7% accuracy, while our pipeline achieved 71.2%, a negligible difference of 0.5%. This suggests that summarizing contexts did not hinder the LLM’s ability to generate correct answers. Further improvements in prompt engineering or summary generation could potentially surpass the traditional pipeline’s accuracy.

<span style="font-size: 20px; font-weight: bold;">RAG Effectiveness</span>

RAG effectiveness was evaluated to determine how well each pipeline retrieved the most relevant context for a given query. The PubMedQA dataset contexts were queried using labeled questions, and a retrieval was marked correct if the top result matched the labeled correct context <d-cite key="jin2019"></d-cite>. The traditional pipeline achieved 99.2% RAG effectiveness, while our pipeline achieved 98.9%, a minor 0.3% difference. The minimal reduction shows that summarization and clustering do not compromise retrieval quality. Additionally, granting the proposed pipeline access to original embeddings instead of summaries for labeling could eliminate this gap entirely, further reinforcing its effectiveness.

<span style="font-size: 20px; font-weight: bold;">Storage Efficiency</span>

We measured storage efficiency by calculating the total size of stored contexts in the vector database (excluding vector embeddings). The traditional pipeline required 1,351 KB, whereas our pipeline used only 571 KB, a reduction of 57.7%. This demonstrates the significant compression achieved through summarization and clustering. The benefits are particularly pronounced in unstructured datasets, where redundant data can be consolidated. While PubMedQA is a structured dataset, in more diverse real-world datasets, the proposed pipeline would likely achieve even greater storage savings. This reduced footprint allows for the storage of larger datasets and faster query times, providing scalability without sacrificing usability.

<span style="font-size: 20px; font-weight: bold;">Validity Judgment with Chain of Thought</span>

In addition to the metrics discussed above, we evaluated the integration of a validity judgment mechanism in both the traditional and proposed pipelines. The PubMedQA dataset, designed for direct question-answer accuracy, presents inherent limitations for chain-of-thought reasoning <d-cite key="jin2019"></d-cite>. Despite this, the validity judger paired with the traditional pipeline achieved a factual accuracy of 72.4%, demonstrating its effectiveness in assessing and verifying LLM responses.

When integrated into the proposed pipeline, the accuracy decreased to 68.9%, showcasing areas for refinement in our system rather than inherent shortcomings in the validity judger. This drop is attributed to two primary reasons:

- **Skewed Scoring Distribution**: Validity and importance scores were overly concentrated near 1, limiting their ability to distinguish subtle differences in factual accuracy.  
- **Excessive Context Retrieval**: The RAG system retrieved more context than necessary, introducing noise to the validation process.

While the PubMedQA dataset is not well-suited for chain-of-thought reasoning, the validity judger still demonstrated its capability to decompose LLM responses into granular statements and retrieve relevant contexts for validation. These findings demonstrate its potential as a strong component for factual validation, particularly on tasks requiring multi-step reasoning. Addressing scoring and retrieval challenges, along with testing on more reasoning-intensive datasets, will likely see significant improvements in both accuracy and efficiency.

## Alternative Solution: Statement Extraction

One of the main limitations of the proposed framework (summarization-based) is the potential loss of important contextual details—particularly concerning in high-stakes domains such as finance, law, or medicine. To mitigate this risk and retain comprehensive coverage of critical information, we introduce an Alternative Pipeline that avoids excessive compression by converting the input data into standalone, verifiable statements. When an input passage can’t safely be condensed, the system simply reproduces statements identical to the source text, thus ensuring no crucial content is omitted. Rather than aiming for a prescribed compression metric, the pipeline focuses on statement-level extraction, preserving essential facts while eliminating redundancies and inaccuracies.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-factual-validation-simplification/alternative.png' | relative_url }}" style="width: 75%;">
    <figcaption style="font-size: 15px; text-align: left; margin: 0 auto; max-width: 800px;">
        <strong>Figure 6.</strong> This diagram illustrates the workflow of the Statement Extraction pipeline, which adds extracted and compressed statements into a vector database. When data is added to the system, the text is converted into statements, clustered and combined based on similarity, embedded, and entered into a vector database. When data is fetched from the system, it fetches the closest embeddings and statements to the query embedding and determines their sufficiency in answering the prompt. If the context is insufficient, it creates new queries based on the information it needs and repeats the process until it has enough information. Then, it returns the statements that it fetched. The boxes highlighted with blue dashed borders indicate modifications to the summarization-based pipeline.
    </figcaption>
</figure>

## Extended Benchmarks

To evaluate this alternative pipeline, experiments were conducted on two widely used QA datasets: Stanford Question Answering Dataset (SQuAD) <d-cite key="rajpurkar2016"></d-cite> and HotpotQA <d-cite key="yang2018"></d-cite>. We compare factual accuracy and storage size between this Alternative Pipeline and the Traditional Pipeline that stores and retrieves paragraph-level contexts under ideal conditions with RAG.

#### SQuAD

| **Metric**          | **Traditional Pipeline** | **Statement Extraction** | **Difference** |
|---------------------|--------------------------|--------------------------|----------------|
| Factual Accuracy    | 87.3%                   | **89.7%**                | +2.4%          |
| Storage Size        | 1.4 MB                  | **1.1 MB**               | -21.43%        |

**Table 2.** On the SQuAD dataset, statement extraction yields a 2.4% increase in factual accuracy while reducing storage size by 21.43%.

#### HotpotQA

| **Metric**          | **Traditional Pipeline** | **Statement Extraction** | **Difference** |
|---------------------|--------------------------|--------------------------|----------------|
| Factual Accuracy    | 92.0%                   | **93.3%**                | +1.3%          |
| Storage Size        | 763 KB                  | **701 KB**               | -8.12%         |

**Table 3.** On the HotpotQA dataset, the statement extraction pipeline achieves a 1.3% higher accuracy while reducing storage by 8.12%. Though less significant than SQuAD, the improvement demonstrates that statement-level coverage can enhance reliability even in multi-hop question answering tasks.

Notably, in comparison to our proposed pipeline, the storage savings here are not as large. However, this Alternative Pipeline ensures no critical details are inadvertently removed. Just as with the proposed approach, it integrates easily into existing RAG setups, requiring minimal computational overhead. DBSCAN and cosine similarity are relatively lightweight, and the LLM calls typically cost only around $0.01 per API call.

## Realistic Usage

Most real-world data comes from unstructured sources, including blog posts, news reports, or technical articles. Below, we demonstrate a small-scale test with two AI-related news articles—one from Google DeepMind <d-cite key="deepmind"></d-cite> and another from Wired <d-cite key="wired"></d-cite>. Both discuss the Gemini 2.0 model releases, showcasing how statement extraction saves storage while retaining necessary context.

| **Metric**              | **Google Deepmind** | **Wired Article** | **Combination** |
|-------------------------|---------------------|-------------------|-----------------|
| Raw Size               | 3.0 KB             | 5.4 KB            | 8.4 KB          |
| Statement Extraction Size | **2.2 KB**       | **2.3 KB**        | **3.3 KB**      |
| Percentage Reduction     | 26.67%            | 57.41%            | 60.71%          |

**Table 4.** Storage sizes before and after statement extraction. Combining both articles after statement extraction yields a 3.3 KB storage size, significantly smaller than the raw 8.4 KB.

Through DBSCAN clustering, near-duplicate statements—such as multiple introductions of “Gemini 2.0” from different perspectives—are merged into a single statement without risking the loss of important data. In high-stakes settings, this pipeline can similarly maintain critical details (e.g., disclaimers, side effects, or legal clauses). Consequently, while it may not compress as aggressively as the proposed pipeline, the Alternative Pipeline provides stronger guarantees of context sensitivity and factual completeness.

## Broader Implications

The results of these two pipelines highlight their potential to improve how RAG systems handle unstructured and large-scale datasets. By either compressing data via summarization or preserving complete details through statement extraction, it expands the capacity of vector databases, allowing systems to manage larger and more diverse datasets without sacrificing query performance. Such scalability is critical for real-world applications and for increasing context windows, enabling more complex multi-step reasoning tasks and better contextual understanding <d-cite key="yao2024"></d-cite>.

Furthermore, by leveraging modular components like clustering and vectorized storage, it enables future systems to integrate real-time updates, ensuring that models stay relevant as knowledge evolves without requiring full retraining. The design also ensures seamless integration with existing RAG and GPT systems, supporting easy implementation while allowing for domain-specific customization. As a result, organizations and researchers can tailor these methods to build specialized QA systems, bias detection tools, or other applications where data quality is essential.

## Limitations and Future Work

1. **Suboptimal Fit**:
- The data retrieved may not always align perfectly with the prompt, leading to potential mismatches in utility. 
- **Mitigation**: Introduce additional retrieval steps that identify and remove irrelevant facts, guided by LLM evaluations.

2. **Context Relevancy**:
- Statements sometimes refer to their original context, making them unhelpful when used by themselves. 
- **Mitigation**: Better prompt engineering for statement generation or additional agents to ensure statements are standalone before entry into the database.

3. **Potential Source Bias**:
- Bias in the original source documents can propagate into the system, affecting the impartiality of responses. 
- **Mitigation**: Increase the number of trusted source documents to prevent bias by having a higher quantity of unbiased statements overrule the lower quantity of biased statements.

<div class="box-future" markdown="1">
#### Future Direction: Concept-Based Fact Representation

Building on the strengths of the current system, a future enhancement could involve representing facts as structured relationships between concepts (e.g., `CONCEPT 1, CONNECTION, CONCEPT 2`). By storing concepts as Word2Vec embeddings and dynamically mapping their relationships in a database, this approach could enable more granular validation and reasoning <d-cite key="mikolov2013"></d-cite>. This would extend our existing pipeline by not only validating individual facts but also examining their interconnectedness, facilitating a richer understanding of context and relationships.

The potential benefits of this approach include:
- **Improved Contextual Validation**: Exploring relationships between concepts could strengthen the system’s ability to handle complex queries, reducing the risk of context misalignment and improving factual accuracy.
- **Enhanced Scalability**: Dynamically storing concepts and their connections would allow the system to adapt to new knowledge over time, enhancing long-term usability and robustness.
- **Greater Explainability**: Converting retrieved connections into human-readable formats could improve transparency, offering clearer and more interpretable rationales for decisions made by the system.

While this concept represents a forward-looking direction, it builds naturally on the goals of the current system. By enhancing scalability, reliability, and transparency, it offers a pathway for advancing the framework in future iterations.
</div>

## Reproducibility

We have open-sourced all code for our framework at [Tonyhrule/Factual-Validation-Simplification](https://github.com/Tonyhrule/Factual-Validation-Simplification). The repository contains complete scripts and instructions for replicating the results from this work. We hope this open-source release facilitates public auditing, broader adoption of our methods, encourages community-driven improvements, and enables new research on trustworthy LLMs.

## Conclusion

In this blog post, we explored two approaches to improving the scalability and reliability of RAG pipelines by addressing challenges in storage efficiency and factual accuracy. The first method uses summarization, clustering, and vectorized fact storage to optimize data size while maintaining strong retrieval performance. The second method—the Alternative Pipeline—avoids summarization and instead extracts standalone statements, offering higher fidelity in high-stakes domains where preserving every nuance of context is critical.

Through experimentation with PubMedQA, SQuAD, HotpotQA, the pipelines demonstrated competitive performance relative to traditional methods. The summarization-based pipeline excels in reducing storage space, while statement extraction preserves sensitive details, improving factual accuracy and removing redundant information. Looking ahead, by grounding outputs in verified information and ensuring greater transparency in decision-making, this framework takes a step forward in building trustworthy and explainable AI systems.
