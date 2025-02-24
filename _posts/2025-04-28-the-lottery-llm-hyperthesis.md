---
layout: distill
title: The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve?
description: Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods.


date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Zhenheng Tang
    url: "https://scholar.google.com/citations?user=FlYcrEcAAAAJ"
    affiliations:
      name: The Hong Kong University of Science and Technology
  - name:  Xiang Liu
    url: "https://scholar.google.com/citations?user=VtK5lwUAAAAJ"
    affiliations:
      name: The Hong Kong University of Science and Technology (Guangzhou)
  - name: Qian Wang
    url: "https://scholar.google.com/citations?user=KAGrBdoAAAAJ"
    affiliations:
      name: National University of Singapore
  - name: Peijie Dong
    url: "https://scholar.google.com/citations?user=TqS6s4gAAAAJ"
    affiliations:
      name: The Hong Kong University of Science and Technology (Guangzhou)
  - name: Bingsheng He 
    url: "https://scholar.google.com/citations?user=RogYLKYAAAAJ"
    affiliations:
      name: National University of Singapore
  - name: Xiaowen Chu
    url: "https://scholar.google.com/citations?user=v4rX24EAAAAJ"
    affiliations:
      name: The Hong Kong University of Science and Technology (Guangzhou)
  - name: Bo Li
    url: "https://scholar.google.com/citations?user=SLgWQLEAAAAJ"
    affiliations:
      name: The Hong Kong University of Science and Technology


# must be the exact same name as your blogpost
bibliography: 2025-04-28-the-lottery-llm-hyperthesis.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Current Efforts on Compressing LLMs and KV Cache
  - name: Tackling Redundant and Unreal Knowledge of LLMs with Knowledge Retrival
  - name: External Tools
  - name: Computational Expressivity of LLMs
  - name: Multi-step Reasoning
  - name: Lottery LLM Hypothesis
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

## Current Efforts on Compressing LLMs and KV Cache
LLMs have demonstrated remarkable proficiency in natural language processing, enabling sophisticated interactions and understanding of human language<d-cite key="openai2023gpt4"></d-cite>. To learn the tremendous knowledge in the training datasets, the current advanced LLMs like GPT-4 <d-cite key="openai2023gpt4"></d-cite> and Llama-3 <d-cite key="touvron2023llama"></d-cite> have enormous parameters, ranging from $7 to \750$ billion. Training such an LLM requires extensive computational resources, often measured in enormous GPU days using advanced NVIDIA GPUs<d-cite key="touvron2023llama"></d-cite>. This results in substantial electricity consumption, impacting both economic and energy costs<d-cite key="samsi2023words, tangDVFS"></d-cite>, and raising concerns regarding sustainable computing<d-cite key="wilkins2024hybrid"></d-cite>. Furthermore, providing inference services for LLMs necessitates numerous GPUs and incurs additional energy costs<d-cite key="samsi2023words, tangDVFS"></d-cite>, making it a significant challenge for widespread deployment<d-cite key="10.1145/3620666.3651329"></d-cite>.

**Compression methods.** To this end, both academic researchers and industrial engineers are trying to compress model parameters and reduce the model to a smaller one while keeping its performance unchanged. The typical compression algorithms include pruning<d-cite key="Sun2023ASA_wanda, Frantar2023SparseGPTML, dongpruner"></d-cite> and quantization<d-cite key="Yao2022ZeroQuantEA, Dettmers2022TheCF, dong2024stbllm"></d-cite> of LLM parameters, as well as KV cache compression<d-cite key="zhang2024h2o, xiao2024efficient"></d-cite>. However, most of the current methods that compress LLMs and KV cache only show guaranteed performance in terms of perplexity on some basic language tasks like Wikitext2<d-cite key="merity2016pointer"></d-cite> and PTB<d-cite key="marcus1993building"></d-cite>, common sense knowledge QA tasks<d-cite key="hendrycks2021measuring, talmor-etal-2019-commonsenseqa"></d-cite>, and basic arithmetic reasoning tasks<d-cite key="cobbe2021training"></d-cite> in small-scale evaluations but not in real-world industrial scenarios.

**Missed aspects.** Some recent studies show that LLMs may lose their crucial advanced abilities under compression, such as long-context retrieval, long-context generation, and long-document reasoning<d-cite key="jaiswal2024compressing"></d-cite>. Additionally, the long-context understanding ability of LLMs is significantly reduced under KV cache compression<d-cite key="yuan-etal-2024-kv"></d-cite>.

In the following sections, we examine recent advancements in retrieval-augmented generation, the utilization of external tools, and multi-step reasoning, all of which markedly enhance the performance of LLMs. Subsequently, we introduce the lottery LLM hypothesis, which posits that for a specific LLM and task, a smaller lottery LLM can achieve equivalent performance to the original LLM, aided by multi-step reasoning and external tools. Drawing from the review of current LLM advancements, we discuss and outline the critical capabilities that the lottery LLM and KV cache compression should encompass, which are currently neglected in existing methodologies.




## Tackling Redundant and Unreal Knowledge of LLMs with Knowledge Retrival
**Redundant Knowledge.** In contemporary applications, many individuals utilize LLMs as encyclopedic resources or to verify news and academic research, akin to an Internet search engine. Recent studies indicate that LLMs exhibit varying performance in knowledge retrieval, contingent upon the popularity of the information<d-cite key="PopQA"></d-cite>. Specifically, a small subset of real-world question-answer (QA) pairs constitutes the majority of interactions, while a limited number of QAs receive frequent attention, demonstrating a long-tail distribution in their popularity<d-cite key="PopQA"></d-cite>. LLMs tend to perform better on high-popularity QAs compared to those with lower popularity.

**Hallucinated Knowledge.** LLMs often generate unreal outputs rather than factual knowledge, a phenomenon known as hallucination<d-cite key="huang2023survey"></d-cite>. This issue has garnered significant attention from researchers<d-cite key="huang2023survey"></d-cite>. There is ongoing debate regarding the feasibility of completely eliminating hallucinations<d-cite key="farquhar2024detecting"></d-cite>. Some studies suggest that hallucinations are inevitable, as they are a byproduct of the model's reasoning and generalization abilities<d-cite key="banerjee2024llms, xu2024hallucination"></d-cite>.

**Retrieval Augmented Generation (RAG).** Large Language Models (LLMs) exhibit robust in-context learning capabilities, enabling them to respond to queries using prompts rather than relying solely on their internal knowledge encoded within model parameters. Consequently, external knowledge sources such as scholarly articles, web pages, books, and other documents can be integrated into prompts to facilitate the retrieval of additional factual information<d-cite key="yao2022react"></d-cite>, thereby mitigating the occurrence of hallucinations<d-cite key="yao2022react"></d-cite>. This approach raises significant research questions:
<blockquote>
Is it necessary to store all knowledge within LLM parameters if RAG can accurately retrieve factual information from external knowledge bases? If not, which knowledge should be stored and which should not?
</blockquote>

Considering two extreme scenarios:
- Storing all knowledge in **model parameters**: If all knowledge is stored within model parameters, LLMs function as oracle machines, obviating the need for RAG. However, training such an LLM is nearly impossible because not all knowledge can be collected and never become outdated<d-cite key="xu2024hallucination, banerjee2024llms"></d-cite>. Moreover, deploying such a large model is inefficient.
- Storing all knowledge in **external knowledge bases**: If all knowledge is stored externally, LLM parameters could potentially be reduced significantly, allowing for the retrieval of factual information during inference.

Nevertheless, LLMs require foundational common knowledge to perform tasks such as reasoning and accurate retrieval. This issue will be further explored in subsequent sections. Thus, compressing all knowledge into external knowledge bases is not feasible. Investigating the nature of learned knowledge and identifying which knowledge triggers the grokking phenomenon in LLMs remains an open research question<d-cite key="Grokking"></d-cite>.

**Trade-off between model size and knowledge base.** Some studies indicate that adaptive knowledge retrieval is a promising direction to enhance the performance of LLMs and may help to find an optimal trade-off between the knowledge base and model size<d-cite key="jeong2024adaptive"></d-cite>. The adaptive RAG<d-cite key="soudani2024fine, jeong2024adaptive"></d-cite> suggests that popular knowledge can be stored in the model parameters, while less popular knowledge can be stored in the external knowledge base.

The core idea of adaptive RAG appears to be related to a classic efficient data structure, **Huffman coding**<d-cite key="moffat2019huffman"></d-cite>. Specifically, the cost of knowledge retrieval can be viewed as the prompt length (since the retrieved knowledge will be inserted into the prompts). Storing knowledge in the model parameters results in a shorter prompt length because LLMs can directly respond to questions without needing to retrieve knowledge from the external knowledge base. Conversely, storing knowledge in the external knowledge base results in a longer prompt length, implying higher retrieval operations and longer context lengths, which incur greater computational and storage costs during inference<d-cite key="xiao2024efficient"></d-cite>. Therefore, the popularity of the knowledge can be seen as the appearance probability, as in Huffman coding. Storing popular knowledge in the model parameters is more efficient.

**Finetuning vs. retrieval.** Another related question is whether finetuning should be used to enhance the performance of LLMs in specific application domains such as legal, finance, and medical fields<d-cite key="hendrycks2021measuring, talmor-etal-2019-commonsenseqa"></d-cite>. Finetuning may lead to the forgetting problem and additional training overheads, sparking debate on whether finetuning should be employed to improve LLM performance or if reliance on RAG can achieve the same goal<d-cite key="jeong2024adaptive"></d-cite>. Recent studies demonstrate that RAG can significantly enhance LLM performance in specific domains such as legal<d-cite key="pipitone2024legalbench"></d-cite>, medical<d-cite key="jeong2024improving"></d-cite>, and finance<d-cite key="li-etal-2024-alphafin"></d-cite>.

**Beyond the RAG.** Document-based knowledge retrieval primarily assists LLMs in retrieving knowledge of triplets consisting of entity, relation, and object<d-cite key="chen-etal-2024-retrieval"></d-cite>. However, the capabilities and exceptional performance of LLMs extend beyond retrieving triplet knowledge. LLMs also exhibit remarkable abilities such as solving arithmetic problems, playing chess, and coding, which are not simple triplet knowledge retrieval tasks<d-cite key="chen-etal-2024-retrieval"></d-cite>. Ensuring the reasoning performance of smaller LLMs is crucial and cannot be easily addressed by document-based knowledge retrieval.


## External Tools
Advanced Large Language Models (LLMs) demonstrate remarkable capabilities in function calling, which involves invoking external tools for addressing specific tasks. These external tools may include Internet search engines<d-cite key="ToolLLM"></d-cite>, arithmetic calculation functions<d-cite key="NEURIPS2023_d842425e"></d-cite>, system operations<d-cite key="LLM-as-OS, AIOS"></d-cite>, game interfaces, and more. These are formulated into programming function calls<d-cite key="Granite-Function"></d-cite> and conveyed to LLMs via prompts. Based on the function descriptions, LLMs determine which function to call to resolve the given problems<d-cite key="Granite-Function"></d-cite>.

**Arithmetic Function Calls.** To solve arithmetic problems, LLMs are trained on arithmetic datasets<d-cite key="cobbe2021training"></d-cite>. However, simple errors often occur during the arithmetic reasoning process, such as LLMs erroneously determining that 9.11 is greater than 9.9<d-cite key="choi2024automatic"></d-cite>. To mitigate this, some studies propose enabling LLMs to generate programs that include arithmetic operations and use an external Python interpreter to solve these problems<d-cite key="pmlr-v202-gao23f"></d-cite>. Additionally, some research suggests leveraging arithmetic function calls to solve arithmetic problems<d-cite key="he23solving"></d-cite>. Experimental results indicate that arithmetic function calling can significantly enhance the performance of LLMs on arithmetic tasks<d-cite key="pmlr-v202-gao23f, NEURIPS2023_e3936777"></d-cite>.

**Internet Search Engine.** To augment LLM knowledge with online and dynamically updated external information, the Internet search engine is employed as an external tool<d-cite key="yao2022react, FreshLLMs"></d-cite>. Experimental results demonstrate that interacting with an Internet search engine, such as a simple Wikipedia API, can significantly improve LLM performance on knowledge retrieval tasks<d-cite key="yao2022react"></d-cite>.

**LLM Operating System (OS).** By conceptualizing LLM calls as system calls akin to traditional operating systems, recent studies propose developing a new *LLM-as-OS* framework<d-cite key="LLM-as-OS"></d-cite>, which allows LLMs to invoke external tools like applications in an OS. Recent studies also propose the AIOS framework<d-cite key="AIOS"></d-cite> to decouple LLM calls from system calls and implement various managers to enhance AIOS efficiency. The optimized agent framework from the OS perspective significantly improves both the efficiency and performance of LLM calls.

**Logic Solver.** There is ongoing debate regarding whether LLMs can perform logical reasoning akin to humans<d-cite key="GSM-Symbolic, CanLLMReason, valmeekam2022large, 10.1145/3627673.3679832, xu2023large, arkoudas2023gpt4cantreason"></d-cite>. Recent studies suggest that to enhance the reasoning capabilities of LLMs, external logic solvers can be used to solve logical reasoning problems<d-cite key="RecursiveReasoning"></d-cite>. In some frameworks, LLMs are tasked with transforming natural language sentences into logical forms, while logic solvers are responsible for solving the logical reasoning problems<d-cite key="han2022folio, pan-etal-2023-logic, wang2024symbolic"></d-cite>. Other frameworks propose allowing LLMs to summarize sentences into premises and conclusions, then aggregate this extracted information into another prompt to enable logic inference<d-cite key="sun-etal-2024-determlr, wang2024symbolic, Xu2024FaithfulLR"></d-cite>.

## Computational Expressivity of LLMs

**Basic Transformer Architecture.** Basic transformers, devoid of intermediate decoding steps, exhibit limited computational expressivity<d-cite key="10.1162/tacl_a_00562, chiang2023tighter"></d-cite>, aligning with the relatively small circuit complexity class $TC^0$<d-cite key="10.1162/tacl_a_00562"></d-cite>. These basic transformers fall short of Turing completeness, as they are incapable of solving problems that are complete for classes larger than $TC^0$, such as simulating automata, which is $NC^1$-complete.

**Decoding-based Transformers.** Decoding-based transformers generate output sequentially, word by word, rather than producing a single answer. This approach enhances their computational expressivity compared to basic transformers, with expressivity increasing in tandem with the length of the decoding steps<d-cite key="merrillexpressive"></d-cite>. This phenomenon elucidates why the Chain-of-Thought (CoT) reasoning process<d-cite key="CoT"></d-cite> augments the computational expressivity of LLMs<d-cite key="Reveal-CoT"></d-cite>. Some studies demonstrate that with linear steps, transformers equipped with projected-norm can theoretically simulate a Turing automaton<d-cite key="merrillexpressive"></d-cite>. Recent research indicates that autoregressive decoding, which facilitates the processing of arbitrarily long input strings, can simulate a universal Turing machine<d-cite key="Autogressive-Turing"></d-cite>.

**Decoding with External Memory.** Research suggests that external memory can enhance the computational expressivity of LLMs<d-cite key="deletang2023neural"></d-cite>, potentially endowing them with approximate Turing completeness<d-cite key="JMLRv2220-302"></d-cite>. Recent advancements have introduced the Stack-Attention mechanism to further enhance the reasoning capabilities of LLMs<d-cite key="Stack-Attention"></d-cite>. With the integration of external memory and simple regular expression parsers, transformers can simulate the execution of a universal Turing machine, specifically $U_{15,2}$<d-cite key="Memory-Augmented-Turing"></d-cite>.





## Multi-step Reasoning

The Chain-of-Thought (CoT) reasoning paradigm demonstrates that engaging in detailed, step-by-step reasoning can significantly enhance the performance of Large Language Models (LLMs) compared to single-step reasoning<d-cite key="CoT"></d-cite>. This improvement arises because single-step reasoning may overlook crucial intermediate steps that are instrumental in problem-solving<d-cite key="CoT"></d-cite>. The multi-step reasoning process, inspired by human cognitive processes, can substantially elevate the performance of LLMs<d-cite key="CoT"></d-cite>.

**Single LLM Call.** CoT exemplifies a single LLM call, utilizing the model once. Beyond explicit prompting to initiate detailed reasoning, recent studies propose enabling LLMs to execute advanced search algorithms during the decoding process, such as Monte-Carlo Tree Search (MCTS)<d-cite key="Decoding"></d-cite> or Q-star search<d-cite key="chakraborty2024transfer"></d-cite>. Additionally, some research suggests employing backtracking algorithms to allow LLMs to reconsider previous decisions, thereby enhancing final performance<d-cite key="fubreak"></d-cite>.

**Multiple LLM Calls.** Some approaches advocate for multiple LLM calls, which operate independently of each other, potentially yielding correct answers across these calls<d-cite key="brown2024large"></d-cite>. Beyond the single CoT call, CoT-SC proposes multiple CoT-based LLM calls, selecting the optimal answer to improve final outcomes<d-cite key="wangself"></d-cite>. However, these answers exhibit direct dependencies. To optimize scheduling and decomposition of the reasoning process, Tree-of-Thought (ToT) reasoning<d-cite key="ToT"></d-cite> and Graph-of-Thought (GoT) reasoning<d-cite key="GoT"></d-cite> have been introduced, structuring reasoning steps in tree-like or graph-like configurations. Some studies also suggest integrating knowledge graphs, enabling LLMs to reason within graph structures to enhance reasoning capabilities<d-cite key="luoreasoning, sunthink"></d-cite>. Structuring prompts into triplets using LLMs can further bolster reasoning abilities<d-cite key="jiang2023structgpt"></d-cite>. In the absence of a centralized controller, some research proposes simulating multiple agents with LLMs to collaboratively address problems<d-cite key="li2023camel, hong2024metagpt, liang2023encouraging, duimproving"></d-cite>.

**Planning and Scheduling.** The essence of multi-step reasoning lies in decomposing the original problem into multiple sub-problems and addressing them sequentially. This process involves planning and scheduling. To facilitate autonomous planning and scheduling, recent studies propose employing LLMs as meta-agents to orchestrate planning and scheduling, wherein the original problem is decomposed, and the meta-agent delegates sub-problems to other LLMs based on the schedule<d-cite key="hong2024metagpt, wu2024autogen, zhoulanguage, wangvoyager"></d-cite>. With the aid of external symbolic reasoning, LLMs can also engage in planning and scheduling to resolve problems<d-cite key="RecursiveReasoning"></d-cite>.

















## Lottery LLM Hypothesis

Consider an original language model $f_\theta$ parameterized by the $\theta \in \mathbb{R}^{k_{\theta}}$, capable of processing input of token length $n$, and an input problem $q \in \mathbb{R}^{m\times h}$ with token length $m < n$ and ground truth $\mu \in \mathbb{R}^{l\times h}$. The problem $q$ is a question consisting of a sequence of words. And the $\mu$ is also a sequence of words representing the answer to the question $q$. $h$ is the dimension of the word embedding. The performance of the model is evaluated using a performance measure $P(\cdot)$, expressed as $P(f_\theta(q), \mu)$ which maps its inputs as a scalar value. We hypothesize the existence of a smaller language model $g_\phi$ with parameters $\phi \in \mathbb{R}^{k_{\phi}}$ ($k_{\phi} < k_{\theta}$) and the same input length $n$, which can solve the problem $q$ with performance comparable to $f_\theta$, such that:

$$P(f_\theta(q), \mu) \leq P( \mathcal{A}_{g_\phi, \mathcal{D}, \mathcal{R}, \mathcal{C}, \mathcal{M}}(q), \mu),$$

where $\mathcal{A}$ represents a reasoning algorithm that may involve one or multiple invocations of $g_\phi$ with various inputs, including the original problem $q$, documents $d \in \mathcal{D}$ retrieved from the external knowledge base $\mathcal{D}$, or function calls $c \in \mathcal{C}$ retrieved from external tools $\mathcal{C}$ using the retriever $\mathcal{R}$. Each document $d \in \mathbb{R}^{n_d\times h}$ is a vector of words. While the function calls $c: \mathbb{R}^{n_c^i\times h} \to \mathbb{R}^{n_c^o\times h}$ is a provided function. The knowledge base $\mathcal{D}$ is a vector database storing vector documents as key-value pairs, and $\mathcal{M}$ denotes the external memory that stores intermediate results. All $\mathcal{D}$, $\mathcal{C}$, and $\mathcal{M}$ are sets. And items in $\mathcal{D}$ and $\mathcal{C}$ are key-value pairs depending on the specific tasks, like vector database<d-cite key="pan2024survey"></d-cite>. The retriever $\mathcal{R}$ is a function that retrieves the required documents or function calls from the $\mathcal{D}$ or $\mathcal{C}$ based on the request. And its specific implementation can be various <d-cite key="gao2023retrieval"></d-cite>.

The reasoning algorithm $\mathcal{A}$ is described as Algorithm 1, employing a divide-and-conquer strategy to solve the original problem $q$. This dynamic divide-and-conquer methodology is versatile and applicable to numerous contemporary reasoning algorithms.



<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-the-lottery-llm-hyperthesis/algorithm.png' | relative_url }}" width="200">
    <figcaption style="font-size: 1em;">Figure 1: A general pseudo code of the reasoning algorithm $\mathcal{A}$.</figcaption>
</figure>






**Recursive and Dynamic Scheduling.** Algorithm 1 can encompass tree-based reasoning methods such as Tree-of-Thought (ToT)<d-cite key="zhoulanguage, ToT"></d-cite>, due to its recursive design that facilitates tree search and allows the branch-or-solve mechanism to be dynamically determined by LLMs. Additionally, Algorithm 1 is applicable to graph-based reasoning methods like Graph-of-Thought (GoT)<d-cite key="GoT, luoreasoning, sunthink"></d-cite>, as the interaction between different LLMs and the external memory $\mathcal{M}$ can be conceptualized as a combination in GoT, where outputs from various nodes are integrated to construct the graph structure.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-the-lottery-llm-hyperthesis/graph-structure.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 2: The problem-solving process of the multi-step reasoning with external tools (the interaction with the external memory and the verification are not shown in the figure).</figcaption>
</figure>




**External Knowledge and Tools.** During each phase of problem-solving, Algorithm 1 initially assesses whether the problem can be directly addressed using the external knowledge base $\mathcal{D}$ or external tools $\mathcal{C}$. If so, Algorithm 1 utilizes $g_\phi$ to evaluate the problem $q$ and ascertain the necessary knowledge or tools required for its resolution. Subsequently, based on the generated requests, the retriever $\mathcal{R}$ searches for external knowledge $d \in \mathcal{D}$ or tool $c \in \mathcal{C}$ to provide the requisite results. These supplementary results are then integrated with the problem $q$ for resolution by the model $g_\phi$. This framework facilitates the application of Retrieval Augmented Generation (RAG) and external tools, such as arithmetic calculation functions, Internet search engines, and logic solvers, to effectively address the problem $q$.

**External Memory.** The external memory $\mathcal{M}$ functions as a repository for storing intermediate results throughout the reasoning process. When tackling various sub-problems, intermediate results can be stored in the external memory for reuse in subsequent steps. By interacting with the external memory, Algorithm 1 can emulate reasoning methods that utilize working memory<d-cite key="wang2024symbolic"></d-cite>. The structure of the `Divide_and_Conquer` function in Algorithm 1 is not constrained. Through careful design and programming, the recursive mechanism can execute fundamental operations such as MOV, COPY, JUMP, and WRITE and READ from the external memory, thereby simulating a Turing machine<d-cite key="Memory-Augmented-Turing"></d-cite>, as depicted in Figure 3.





<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-the-lottery-llm-hyperthesis/Automata.png' | relative_url }}" width="50">
    <figcaption style="font-size: 1em;">Figure 3: Simulating the Turing machine with LLMs and the external memory<d-cite key="Memory-Augmented-Turing"></d-cite>.</figcaption>
</figure>


Most of previous model compression<d-cite key="Sun2023ASA_wanda, Frantar2023SparseGPTML"></d-cite> <d-cite key="Yao2022ZeroQuantEA, Dettmers2022TheCF"></d-cite> and KV cache compression methods<d-cite key="zhang2024h2o, xiao2024efficient"></d-cite>only focus on the guaranteeing the model performance on the perplexity metric<d-cite key="merity2016pointer"></d-cite> or some downstream tasks like the common sense knowledge<d-cite key="hendrycks2021measuring, talmor-etal-2019-commonsenseqa"></d-cite> and the basic arithmetic problems<d-cite key="cobbe2021training"></d-cite>. From the above analysis and the procedures of Algorithm 1, we can see that there are some other crucial abilities that the lottery LLM and other compression methods must take into consideration. We summarize the crucial abilities that the lottery LLM should have as follows.

**Ability 1: Retrieval from prompts.** Obviously, the useful information in the prompts related to addressing the problem $q$ is crucial for the lottery LLM. After collecting the required external results into the prompt, the LLM $g_\phi$ needs to be able to retrieve the required information from the prompt and avoid the interruption of some irrelevant information. This is related to the retrieval ability of the LLM and its measurement test is like the well-known needle-in-the-haystack(NIAH) test<d-cite key="needle"></d-cite>. We show that there is a simple and interesting method to endow the LLM with advanced retrieval ability with preprocessing prompts, by applying the embedding to retrieve the related information about the question in problem $q$ and combine them with the question to prompt the LLM $g_\phi$ rather let the LLM $g_\phi$ to process the original long context information of problem $q$.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-the-lottery-llm-hyperthesis/Meta-Llama-3-8B-Instruct_evaluated.jpg' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 4: Vanilla NIAH results of LLaMA3-8B-Instruct.</figcaption>
</figure>

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-the-lottery-llm-hyperthesis/Meta-Llama-3-8B-Instruct-retrieval_evaluated.jpg' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 5: NIAH results of LLaMA3-8B-Instruct with preprocessing prompts.</figcaption>
</figure>


The figures illustrate that preprocessing prompts markedly enhance the performance of LLMs on the NIAH test. Importantly, even when the input length surpasses the model's context size (8K tokens for LLaMA3-8B-Instruct), there is no observed degradation in performance. This indicates the potential of utilizing preprocessed prompts to augment the retrieval capabilities of LLMs.

**Ability 2: Identification of Required External Resources.** To effectively determine which external resources to utilize, such as knowledge databases or external tools, the LLM $g_\phi$ must possess the capability to comprehend and correlate the problem $q$ and its associated sub-problems with the relevant resources. Consequently, $g_\phi$ should have foundational knowledge of the problem $q$ and the external resources. Additionally, it must exhibit a strong ability to associate queries with the available resources. When external tools are adeptly employed, the performance of smaller LLMs can be significantly enhanced. The subsequent table presents the results of arithmetic problem-solving using various LLMs and methodologies. The PAL<d-cite key="pmlr-v202-gao23f"></d-cite> approach, which employs external arithmetic calculation functions, demonstrates a substantial improvement in the performance of smaller LLMs.


|        | GSM8K | SVAMP | ASDIV |  ADDSUB | MULTIARITH |
|--------|:------:|:-----:|:-----:|:------:|:----------:|
| DIRECT Codex | 19.7  |69.9  | 74.0  |  90.9   | 44.0       |
| CoT UL2-20B  | 4.1   |12.6  | 16.9  | 18.2   | 10.7       |
| CoT LaMDA-137B | 17.1  | 39.9  | 49.0  |  52.9   | 51.8       |
| CoT Codex    | 65.6  | 74.8  | 76.9  |  86.0   | 95.9       |
| CoT PaLM-540B | 56.9  | 79.0  | 73.9  |  91.9   | 94.7       |
| CoT Minerva 540B | 58.8  | -     | -     | -      | -          |
| PAL<d-cite key="pmlr-v202-gao23f"></d-cite>            | **72.0**  | **79.4**  | **79.6**  | **92.5**   | **99.2**       |

Besides, with provided the external documents, following results<d-cite key="asai2024selfrag"></d-cite> show that the small LLM (Llama-3-Ins8B) show the superb performance in many QA tasks<d-cite key="mallen2023not, kwiatkowski2019natural, stelmakh2022asqa"></d-cite> than the large LLMs (Llama-3-Ins70B and ChatGPT-4oMINI).


| Method | LLM | PopQA (acc) | NQ (acc) | ASQA (str-em) | ASQA (hit) | 
|--------|-----|-------------|----------|---------------|-------------|
| CoT without RAG | Llama-3-Ins8B | 24.8 | 44.0 | 28.8 | 7.8 |
| CoT without RAG  | Llama-3-Ins70B | 31.6 | 54.4 | 36.4 | 11.2 |
| CoT without RAG | ChatGPT-4oMINI | 32.4 | 53.2 | 32.4 | 8.0 |
| With RAG | Llama-3-Ins8B | **59.8** | **54.0** | **38.8** | **14.0** |




**Ability 3: Planning and Scheduling.** To effectively decompose the problem $q$ into multiple sub-problems and address them sequentially, the LLM $g_\phi$ must possess robust planning and scheduling capabilities. This competency is essential for the lottery LLM to tackle complex problems efficiently. Consequently, the LLM $g_\phi$ should have a comprehensive understanding of both the primary problem $q$ and its constituent sub-problems. However, the intricate details of solving these sub-problems may not be necessary for the LLM $g_\phi$, as external resources can be leveraged to resolve them. Moreover, proficient scheduling is crucial for the lottery LLM to enhance reasoning efficiency.

The table below illustrates the performance of LLMs using simple inference compared to those employing a strategy of decomposing the problem into sub-problems and utilizing external logic solvers, such as Logic-LM<d-cite key="pan-etal-2023-logic"></d-cite>. The five datasets are commonly used in logical reasoning tasks. Notably, we emphasize the results<d-cite key="pan-etal-2023-logic"></d-cite> (simple inference/with Logic-LM) of GPT-3.5, which, despite being less advanced than GPT-4, demonstrates comparable performance to GPT-4 (GPT-3.5 with Logic-LM compared with GPT-4 with simple inference). Thus, with advanced reasoning algorithms, the weaker LLMs can outperform the stronger LLMs in advanced tasks.




| Dataset            | ChatGPT (gpt-3.5-turbo) | GPT-3.5 (text-davinci-003) | GPT-4 (gpt-4) |
|--------------------|-------------------------|----------------------------|---------------|
| PrOntoQA           | 47.40 / 61.00           | 51.80 / **85.00**          | 77.40 / 83.20 |
| ProofWriter        | 35.50 / 58.33           | 36.16 / **71.45**          | 52.67 / 79.66 |
| FOLIO              | 45.09 / **62.74**           | 54.60 / 61.27          | 69.11 / 78.92 |
| LogicalDeduction   | 40.00 / **65.67**           | 41.33 / 62.00              | 71.33 / 87.63 |
| AR-LSAT            | 20.34 / 26.41           | 22.51 / **25.54**              | 33.33 / 43.04 |



**Ability 4: Precise Approximation of Fundamental Operations.** As discussed in the section on the computational expressivity of LLMs, achieving (approximate) Turing completeness necessitates that the LLM $g_\phi$ precisely approximates fundamental operations such as MOV, COPY, JUMP, and WRITE and READ from external memory<d-cite key="Autogressive-Turing, Memory-Augmented-Turing"></d-cite>. Although these operations may not be directly employed in problem-solving, they are essential for the lottery LLM to function as a potential meta-agent<d-cite key="hong2024metagpt"></d-cite>.

**Ability 5: Long-Context Reasoning.** In single-step reasoning, an extended context length allows the LLM $g_\phi$ to access and utilize more information for problem-solving. In multi-step reasoning, the prompt serves as a form of working memory for the meta-agent, or planner (controller). Each result from solved sub-problems should be incorporated into the prompt for subsequent steps. As problem complexity increases, so does the depth of the sub-problem tree. Therefore, the LLM $g_\phi$ must possess the ability for extended contextual reasoning to support deep tree reasoning<d-cite key="merrillexpressive, Reveal-CoT"></d-cite>.

## Conclusion

This blog aims to elucidate the potential of the lottery LLM and to summarize the essential capabilities that the lottery LLM should possess, which are currently lacking in existing methods of LLM and KV cache compression. The discussion on redundant knowledge within LLMs also highlights the trade-off between knowledge storage and reasoning capabilities. With the development of the lottery LLM, alongside external tools, knowledge bases, and a robust algorithm $\mathcal{A}$, there is potential for the lottery LLM to function as a meta-agent akin to human cognition. Its external memory could serve as long-term memory, the prompt as short-term memory, and the LLM inference process $g_\phi$ as the fundamental cognitive process. External tools and knowledge bases can be considered as supplementary tools commonly used in daily life. Deploying the lottery LLM could significantly reduce energy and resource consumption in large-scale LLM-driven applications. Future research on LLM compression, KV cache compression, and other efficient LLM methodologies should address both efficiency and the essential capabilities of LLMs.




**Acknowledgements**

This work was partially supported by National Natural Science Foundation of China under Grant No. 62272122, the Guangzhou Municipal Joint Funding Project with Universities and Enterprises under Grant No. 2024A03J0616, Guangzhou Municipality Big Data Intelligence Key Lab (2023A03J0012), and Hong Kong CRF grants under Grant No. C7004-22G and C6015-23G, contract R6021-20, and RGC GRF grants under the contracts 16200221, 16207922 and 16207423, the MOE Academic Research Fund (AcRF) Tier 1 Grant in Singapore (Grant No. T1 251RES2315).



















































