---
layout: distill
title: "Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations"
description: "Vision language models (VLMs), such as GPT-4o, have rapidly evolved, demonstrating impressive capabilities across diverse tasks. However, much of the progress in this field has been driven by engineering efforts, with a limited understanding of how these models work. The lack of scientific insight poses challenges to further enhancing their robustness, generalization, and interpretability, especially in high-stakes settings. In this work, we systematically review the use of mechanistic interpretability methods to foster a more scientific and transparent understanding of VLMs. Specifically, we examine five prominent techniques: probing, activation patching, logit lens, sparse autoencoders, and automated explanation. We summarize the key insights these methods provide into how VLMs process information and make decisions. We also discuss critical challenges and limitations that must be addressed to further advance the field."
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Yiming Liu*
    url: "https://leo1oel.github.io/"
    affiliations:
      name: Tsinghua University
  - name: Yuhui Zhang*
    url: "https://cs.stanford.edu/~yuhuiz/"
    affiliations:
      name: Stanford University
  - name: Serena Yeung-Levy
    url: "https://ai.stanford.edu/~syyeung/"
    affiliations:
      name: Stanford University

bibliography: 2025-04-28-vlm-understanding.bib  

toc:
  - name: Introduction
  - name: Current Methods
    subsections:
    - name: Probing
    - name: Activation Patching
    - name: Logit Lens
    - name: Sparse Autoencoders
    - name: Automated Explanation
  - name: Future Directions
    subsections:
    - name: From Single Model to Multiple Models
    - name: From Small Models to Large Models
    - name: From Language-Centric to Vision-Centric
    - name: From Static Processes to Dynamic Processes
    - name: From Micro-Level to Macro-Level
  - name: Conclusion

_styles: >
  
  .center {
      display: block;
      margin-left: auto;
      margin-right: auto;
  }

  .framed {
    border: 1px var(--global-text-color) dashed !important;
    padding: 20px;
  }

  d-article {
    overflow-x: visible;
  }

  .underline {
    text-decoration: underline;
  }

  .todo{
      display: block;
      margin: 12px 0;
      font-style: italic;
      color: red;
  }
  .todo:before {
      content: "TODO: ";
      font-weight: bold;
      font-style: normal;
  }
  summary {
    color: steelblue;
    font-weight: bold;
  }

  summary-math {
    text-align:center;
    color: black
  }

  [data-theme="dark"] summary-math {
    text-align:center;
    color: white
  }

  details[open] {
  --bg: #e2edfc;
  color: black;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  font-size: 80%;
  line-height: 1.4;
  }

  [data-theme="dark"] details[open] {
  --bg: #112f4a;
  color: white;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  font-size: 80%;
  }
  .box-note, .box-warning, .box-error, .box-important {
    padding: 15px 15px 15px 10px;
    margin: 20px 20px 20px 5px;
    border: 1px solid #eee;
    border-left-width: 5px;
    border-radius: 5px 3px 3px 5px;
  }
  d-article .box-note {
    background-color: #eee;
    border-left-color: #2980b9;
  }
  d-article .box-warning {
    background-color: #fdf5d4;
    border-left-color: #f1c40f;
  }
  d-article .box-error {
    background-color: #f4dddb;
    border-left-color: #c0392b;
  }
  d-article .box-important {
    background-color: #d4f4dd;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article .box-note {
    background-color: #555555;
    border-left-color: #2980b9;
  }
  html[data-theme='dark'] d-article .box-warning {
    background-color: #7f7f00;
    border-left-color: #f1c40f;
  }
  html[data-theme='dark'] d-article .box-error {
    background-color: #800000;
    border-left-color: #c0392b;
  }
  html[data-theme='dark'] d-article .box-important {
    background-color: #006600;
    border-left-color: #2bc039;
  }
  d-article aside {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
    font-size: 90%;
  }
  .caption { 
    font-size: 80%;
    line-height: 1.2;
    text-align: left;
  }
---

<div style="display: none">
$$
\definecolor{input}{rgb}{0.42, 0.55, 0.74}
\definecolor{params}{rgb}{0.51,0.70,0.40}
\definecolor{output}{rgb}{0.843, 0.608, 0}
\def\mba{\boldsymbol a}
\def\mbb{\boldsymbol b}
\def\mbc{\boldsymbol c}
\def\mbd{\boldsymbol d}
\def\mbe{\boldsymbol e}
\def\mbf{\boldsymbol f}
\def\mbg{\boldsymbol g}
\def\mbh{\boldsymbol h}
\def\mbi{\boldsymbol i}
\def\mbj{\boldsymbol j}
\def\mbk{\boldsymbol k}
\def\mbl{\boldsymbol l}
\def\mbm{\boldsymbol m}
\def\mbn{\boldsymbol n}
\def\mbo{\boldsymbol o}
\def\mbp{\boldsymbol p}
\def\mbq{\boldsymbol q}
\def\mbr{\boldsymbol r}
\def\mbs{\boldsymbol s}
\def\mbt{\boldsymbol t}
\def\mbu{\boldsymbol u}
\def\mbv{\boldsymbol v}
\def\mbw{\textcolor{params}{\boldsymbol w}}
\def\mbx{\textcolor{input}{\boldsymbol x}}
\def\mby{\boldsymbol y}
\def\mbz{\boldsymbol z}
\def\mbA{\boldsymbol A}
\def\mbB{\boldsymbol B}
\def\mbE{\boldsymbol E}
\def\mbH{\boldsymbol{H}}
\def\mbK{\boldsymbol{K}}
\def\mbP{\boldsymbol{P}}
\def\mbR{\boldsymbol{R}}
\def\mbW{\textcolor{params}{\boldsymbol W}}
\def\mbQ{\boldsymbol{Q}}
\def\mbV{\boldsymbol{V}}
\def\mbtheta{\textcolor{params}{\boldsymbol \theta}}
\def\mbzero{\boldsymbol 0}
\def\mbI{\boldsymbol I}
\def\cF{\mathcal F}
\def\cH{\mathcal H}
\def\cL{\mathcal L}
\def\cM{\mathcal M}
\def\cN{\mathcal N}
\def\cX{\mathcal X}
\def\cY{\mathcal Y}
\def\cU{\mathcal U}
\def\bbR{\mathbb R}
\def\y{\textcolor{output}{y}}
$$
</div>

# Introduction



Vision language Models (VLMs), such as GPT-4V <d-cite key="2023GPT4VisionSC"></d-cite> or LLaVA <d-cite key="liu2023visual,liu2023improved,liu2024llavanext,li2024llavaonevision"></d-cite>, have achieved remarkable success across a wide range of tasks, including including Image Captioning <d-cite key="vinyals2014show"></d-cite>, Visual Question Answering (VQA)  <d-cite key="agrawal2015vqa"></d-cite>, and Multimodal Reasoning <d-cite key="NEURIPS2022_11332b6b"></d-cite>. These advancements have driven innovation in diverse fields, such as virtual assistants <d-cite key="wu2023visual"></d-cite>, autonomous robotics <d-cite key="driess2023palme"></d-cite>, and medical diagnostics <d-cite key="singhal2023towards"></d-cite>. However, despite their rapid adoption, the internal mechanisms of these models remain largely opaque, raising significant concerns about their reliability, robustness, and interpretability—particularly in high-stakes applications <d-cite key="kolicic2024inherently"></d-cite>.

Interpretability research offers a promising path to address these challenges. Mechanistic interpretability, in particular, seeks to uncover the inner processes of neural networks and explain how specific outputs are generated <d-cite key="saphra2024mechanistic,hastingswoodhouse2024introduction"></d-cite>. By applying these techniques to VLMs, researchers can gain valuable insights into how these models represent, process, store, and integrate visual and linguistic information, advancing both theoretical understanding and practical utility.

In this work, we examine how mechanistic interpretability methods can illuminate the inner workings of VLMs. We review five key techniques—probing <d-cite key="alain2016understanding,hewitt-manning-2019-structural"></d-cite>, activation patching <d-cite key="NEURIPS2020_92650b2e, NEURIPS2022_6f1d43d5"></d-cite>, logit lens <d-cite key="alignmentforumorg2024interpreting"></d-cite>, sparse autoencoders <d-cite key="bricken2023monosemanticity,DBLP:conf/iclr/HubenCRES24"></d-cite>, and automated explanations <d-cite key="singh2023explaining,bills2023language"></d-cite>—detailing their mechanisms, applications, and the insights they provide through concrete examples. These methods help answer critical questions, such as what information is encoded in VLM representations <d-cite key="cao2020behind"></d-cite>, how and when visual and linguistic modalities are integrated <d-cite key="Palit_2023_ICCV,neo2024towards"></d-cite>, and how individual neurons contribute to the model’s decision-making process <d-cite key="huo2024mmneuron,huang2024miner"></d-cite>.

Additionally, we discuss the limitations of current interpretability methods and highlight five key directions for future research: developing approaches that are more generalizable, scalable, vision-centric, dynamic, and capable of macro-level analysis. For instance, the heterogeneity of VLMs calls for interpretability methods that can adapt across diverse models; the micro level of mechanistic interpretability needs to be complemented by a macro-level perspective for a broader understanding. By addressing these challenges, we aim to pave the way for more transparent, reliable, and capable vision language models.


<br><br><br>
# Current Methods

In this section, we review mechanistic interpretability methods applied to vision language models (VLMs), which aim to uncover the internal processes of these VLMs process visual and language information and explain how they produce specific outputs. Key techniques discussed include probing, activation patching, logit lens analysis, sparse autoencoders, and automated explanations.



<br>
## Probing

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/probing.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Illustration of probing in neural networks: a simple classifier is trained on intermediate representations of a vision language model to predict target properties, revealing the model’s encoding strengths and limitations. In the diagram, $h_0$, $h_1$, $h_2$ represent different attention heads.
</div>
<br>

### What is Probing

*Probing* <d-cite key="alain2016understanding,hewitt-manning-2019-structural"></d-cite> is a diagnostic technique used to analyze the internal representations of neural networks. It helps researchers identify **whether specific types of information are encoded within the model** by training auxiliary classifiers—referred to as probes—on the model's intermediate outputs. This method is particularly useful for understanding what a model has learned and how it organizes information across its layers.

### How Probing Works

Probing involves training supervised classifiers, typically simple ones like linear probes, to predict specific properties from the internal representations of a model. These properties can include linguistic, visual, or multimodal features. The probe’s performance indicates whether the target information is accessible in the model's representations:
- **High accuracy**: Suggests the property is well-encoded.
- **Low accuracy**: Indicates the property may be absent or deeply entangled.

Linear probes are often preferred because their simplicity ensures that high accuracy reflects the quality of the model’s representations, rather than the complexity of the probe itself <d-cite key="belinkov2021probing"></d-cite>.



<details markdown="1">
<summary><b>Example</b></summary>
To illustrate, consider a vision language model analyzing the caption: *"A cat sitting on a mat."*
- A probe could be trained to predict spatial relationships (e.g., object positioning) from intermediate representations of the model.
- Another probe might predict visual attributes, such as "furry" or "striped," encoded in the text embeddings.

If the probe achieves high accuracy, it suggests that these properties (spatial relationships or visual attributes) are captured in the model's representations. If accuracy is low, the information may be missing or insufficiently disentangled to be easily accessed.
</details>
<br>

### Key Findings from Existing Works

Most research on probing tasks in VLMs focuses on two primary objectives: **identifying the concepts these models struggle to capture** and **assessing the relative importance of visual and linguistic modalities** <d-cite key="golovanevsky2024vlms"></d-cite>. 

Cao et al. <d-cite key="cao2020behind"></d-cite> introduced the VALUE (Vision-And-Language Understanding Evaluation) framework, which developed a set of probing tasks to explain individual layers, heads, and fusion techniques. This study reveals several important insights: pre-trained models often prioritize language over vision in multimodal tasks (**modality prioritization**); specific attention heads are effective at capturing interactions between visual and linguistic modalities (**cross-modal interactions**); and visualization of attention mechanisms has revealed interpretable relationships, such as object-object interactions in images (**attention visualization**). 

Studies have also explored diverse model capabilities, such as visual semantics <d-cite key="dahlgren-lindstrom-etal-2020-probing"></d-cite>, verb processing <d-cite key="hendricks2021probing,beňová2024beyond"></d-cite>, numerical reasoning <d-cite key="kajic2022probing"></d-cite>, and spatial reasoning <d-cite key="pantazopoulos2024lost"></d-cite>. A notable line of research compared representations at different training stages, such as pre-training versus fine-tuning, using carefully designed datasets to minimize biases <d-cite key="Salin_Farah_Ayache_Favre_2022"></d-cite>.



### Method Variants and Limitations

**Limitations**:
- **Correlation vs. Causation**: High probe accuracy indicates correlation, not causation; the model may encode the information but not actively use it for predictions <d-cite key="elazar2021amnesic"></d-cite>.
- **Task Design**: Probing tasks must be carefully designed to avoid confounding factors or misleading results <d-cite key="vatsa2023adventures"></d-cite>.
- **Model-Specific Dependencies**: Probing results are often architecture-dependent, limiting their generalizability across models.

<aside class="l-body box-note" markdown="1">
Key Takeaways:
- Probing is a powerful tool for interpreting neural networks, offering insights into the types of information encoded in their representations.
- For vision language models, probing has revealed critical findings on modality interactions, representation priorities, and encoding patterns.
</aside>


<br>
## Activation Patching

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/activation.svg' | relative_url }}" type="image/svg+xml" width="100%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Activation patching compares model behavior under clean, corrupted, noising, and denoising conditions. The results highlight how noising disrupts and denoising restores key logits (A and B values), demonstrating the method's utility in identifying critical layers and components.
</div>
<br>

### What is Activation Patching

*Activation patching* <d-cite key="NEURIPS2020_92650b2e, NEURIPS2022_6f1d43d5"></d-cite> (also known as causal tracing or causal mediation analysis) is an interpretability technique for neural networks. It selectively modifies internal activations while keeping others constant, allowing researchers to investigate **how specific components contribute to model behavior**. This method provides causal insights, helping identify critical components and potential interventions to improve performance and robustness.

### How Activation Patching Works

The activation patching process typically involves five steps:

1. **Save Activations:** Record the internal activations of a model when processing clean and corrupted inputs.
2. **Select Target Activations:** Identify the specific activations to modify.
3. **Patch Activations:** Replace activations from one input (e.g., corrupted) with those from another (e.g., clean).
4. **Rerun the Model:** Run the model with patched activations and observe behavioral changes.
5. **Analyze Results:** Infer the role of specific components based on how the output changes.

There are two primary ways to apply activation patching <d-cite key="heimersheim2024use"></d-cite>:

- **Denoising Analysis** involves taking a corrupted prompt, such as one where Gaussian noise has been added to key embeddings, and replacing its activations with those from a clean prompt. By observing which patched activations restore the clean behavior, researchers can identify the components that are **sufficient to correct** the corrupted behavior. For example, this technique can reveal layers where key information is integrated or restored during processing.
- **Noising Analysis**, on the other hand, starts with a clean prompt and replaces its activations with those from a corrupted prompt. By determining which patches disrupt the clean behavior, this method pinpoints the components **necessary to maintain** the correct output. This analysis is particularly useful for identifying which layers or activations play a critical role in preserving the model's functionality.

<details markdown="1">
<summary><b>Example</b></summary>
Imagine analyzing a Vision language Model (VLM) tasked with identifying objects in an image:
- **Clean Input:** An image of "a cat sitting on a mat."
- **Corrupted Input:** The same image with Gaussian noise added to the "cat" region.

Steps:
1. Run both inputs through the model and save activations for all layers.
2. Replace specific layer activations in the corrupted input with those from the clean input.
3. Rerun the model with these patched activations.
4. Observe if replacing activations in certain layers restores the ability to correctly identify "cat."

If restoring activations in a specific layer consistently fixes errors, this suggests that layer plays a critical role in object recognition.
</details>
<br>

### Key Findings from Existing Works

1. **Visual-Linguistic Integration**
- **Layer-Specific Processing in BLIP**: Palit et al. <d-cite key="Palit_2023_ICCV"></d-cite> used Gaussian noise patching to analyze BLIP's processing patterns. They found that image information primarily influence the model's outputs in specific layers: layer 11 of the question encoder and layers 9-11 of the answer decoder. This observation suggests two possibilities:
  - The model might primarily combine visual and text information in its later layers
  - Later layers might play a more decisive role in the final output, while earlier layers provide supporting information
- **Visual-to-Language Transformation in LLaVA**: Neo et al. <d-cite key="neo2024towards"></d-cite> examined how LLaVA processes visual information. They found that representations at visual token positions change systematically across layers, gradually aligning with interpretable textual concepts. This indicates that VLMs can naturally transform visual information into language-like representations, even without specific visual pretraining.
- **Architectural Differences**: Golovanevsky et al. <d-cite key="golovanevsky2024vlms"></d-cite> developed a new method called *Semantic Image Pairs (SIP)* - a method where they make concept changes to images (e.g., changing a "cat" to a "dog") to understand how VLMs process meaning. Their analysis revealed:
  - Cross-attention serves three functions: object detection, suppression, and outlier suppression
  - Different architectures have distinct characteristics: (1) LLaVA lacks "text-only" attention heads; (2) BLIP has no "vision-only" heads; (3) Both models use universal heads for cross-modal integration

2. **Layer-wise Information Processing**
- **Early vs. Late Layer Functions**: Basu et al. <d-cite key="basu2024understanding"></d-cite> used causal tracing to show that LLaVA primarily retrieves visual information in early layers (1-4), followed by consistent summarization in final visual tokens. Neo et al. <d-cite key="neo2024towards"></d-cite> further investigated this through attention knockout experiments, they found that:
  - Layers 1-10 process broader contextual information
  - Layers 15-24 focus on extracting specific object details
  - Notably, they found that blocking visual token attention to the last row had minimal impact, challenging previous theories about intermediate summarization steps
- **Layer Importance for Model Performance**: Initial studies by Gandelsman et al. <d-cite key="gandelsman2023interpreting"></d-cite> on CLIP showed that final layers have significant direct effects on model accuracy, while early layer modifications (like removing multihead attention) have minimal impact. Balasubramanian et al. <d-cite key="balasubramanian2024decomposing"></d-cite> later extended these findings across a broader range of Vision Transformers, confirming the critical role of the final four layers in model performance.

3. **Analytical Tools**
- Recent analytical tools have significantly enhanced our understanding of VLMs. Ben et al. <d-cite key="Ben_Melech_Stan_2024_CVPR"></d-cite> developed LVLM-Interpret, an interactive tool that combines attention knockout with relevancy mapping and causal graph construction to visualize information flow patterns and identify critical image regions.

### Method Variants and Limitations

- **Variants:**
  - **Direct Ablations** <d-cite key="DBLP:conf/iclr/NandaCLSS23"></d-cite>: A simpler variant where activations are replaced with zeros or dataset means. While zero ablation shows components critical for network behavior, mean ablation is a more natural version of zero ablation.
  - **Path Patching** <d-cite key="goldowsky-dill2023localizing"></d-cite>: An extension that traces specific causal pathways through the network, helping understand how information flows between different model components. 
  - **Attention Knockout** <d-cite key="geva2023dissecting"></d-cite>: A specialized form focused on analyzing attention mechanisms by selectively blocking attention patterns between tokens.

- **Creating Corrupted Inputs:**
  - **Text Inputs:** Introduce *Gaussian Noise (GN)* or use *Symmetric Token Replacement (STR)*, which replaces tokens with semantically similar alternatives. STR is often preferred as GN disrupts model internals.
  - **Image Inputs:** Apply Gaussian noise or use Semantic Image Pairs (SIP) to modify concepts (e.g., change "cat" to "dog") <d-cite key="golovanevsky2024vlms"></d-cite>.




<aside class="l-body box-note" markdown="1">
Key Takeaways:
- Activation patching provides causal insights into model behavior by modifying specific internal activations while keeping others constant
- Different variants (e.g., direct ablation, attention knockout) offer complementary perspectives on information flow and processing
- Key findings reveal:
  - Cross-modal integration occurs primarily in late layers, with visual information gradually evolving towards language-like representations
  - VLMs show hierarchical processing: early layers handle context with minimal direct impact, while final layers are crucial since they extract information more related to the task
  - Different architectures exhibit distinct patterns in how they handle cross-modal information
</aside>



<br>
## Logit Lens

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/logit_lens.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Logit lens uses the model's unembedding matrix to extract and interpret predictions from intermediate layers, providing insights into how the model refines its understanding at each stage.
</div>
<br>

### What is Logit Lens

*Logit lens* <d-cite key="alignmentforumorg2024interpreting"></d-cite> is an analytical method used to understand **how neural networks refine their predictions layer by layer.** By applying the model’s final classification layer (unembedding matrix) to intermediate activations, it projects these activations into vocabulary space. This allows researchers to analyze intermediate predictions, offering insights into the model's evolving understanding of multimodal inputs.

### How Logit Lens Works

The logit lens maps intermediate activations to a sequence of "snapshots" of predictions as they develop across the network’s layers. The process involves:
1. Extracting activations from each layer of the model.
2. Applying the unembedding matrix to transform these activations into vocabulary distributions.
3. Observing how prediction probabilities change from one layer to the next, revealing the model's internal decision-making process.

<details markdown="1">
<summary><b>Example</b></summary>

Consider a vision language model analyzing an image of "a dog chasing a ball in a park." Using the logit lens, the prediction evolution might look like this:
- **Early layers:** Predictions are highly uncertain, with terms like "dog," "animal," and "pet" receiving similar probabilities.
- **Middle layers:** The model begins refining predictions, focusing on "dog" while maintaining context-related terms such as "park" and "ball."
- **Final layers:** The model confidently predicts specific relationships like "dog chasing ball" and integrates objects into a coherent scene.

This example illustrates how the logit lens tracks the progression from basic feature recognition to high-level understanding.
</details>
<br>

### Key Findings from Existing Works

1. **Concept Distribution Patterns**
- MMNeuron <d-cite key="huo2024mmneuron"></d-cite> applies logit lens to analyze hidden states of multimodal models like LLaVA-NeXT and InstructBLIP. Through their analysis of decoded vocabulary distributions, they reveal that image tokens generate notably sparser distributions than text tokens. This observation suggests that **image representations are encoded as mixtures of concepts rather than direct word mappings.**

2. **Representations Evolution**
- By examining the entropy of these distributions across layers, Huo et al. <d-cite key="huo2024mmneuron"></d-cite> uncover a distinctive three-stage pattern: **initial feature alignment with high entropy, followed by information processing with sharply declining entropy in middle layers, and finally token selection with slight entropy increase.** More recent work <d-cite key="neo2024towards"></d-cite> further explores how representations at visual token positions evolve through the layers in LLaVa 1.5, finding that **activations in the late layers at each visual token position correspond to token embeddings that describe its original patch object.**

3. **Reduce Hallucinations**
- Building on these insights, Jiang et al. <d-cite key="jiang2024interpreting"></d-cite> demonstrate practical applications of the logit lens by using it to spatially localize objects and perform targeted edits to VLM's latent representations. Their approach effectively reduces hallucinations without compromising the model's overall performance, showcasing how understanding internal representations can lead to concrete improvements in model reliability.

### Method Variants and Limitations

- **Limitations:**
   - The logit lens can be brittle, as it assumes hidden states remain linearly interpretable across all layers, which may not hold for heavily tuned or non-linear models <d-cite key="belrose2023eliciting"></d-cite>.
   - It is less effective for analyzing tasks requiring complex reasoning or long-term dependencies.

<aside class="l-body box-note" markdown="1"> 
Key Takeaways: 
- The logit lens enables detailed tracking of prediction evolution across model layers.
- Key findings highlight how VLMs process images as distributed concept mixtures and exhibit a distinct three-stage representation evolution.
- Practical applications, such as reducing hallucinations, demonstrate its value in improving model performance.
</aside>


<br>
## Sparse Autoencoders

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/sae.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
The sparse autoencoder works by mapping input activations into a high-dimensional sparse representation using an encoder and reconstructing the original input through a decoder.
</div>
<br>

### What Are Sparse Autoencoders

*Sparse Autoencoders (SAEs)* are a neural network-based method designed to **disentangle complex internal representations in neural networks** by addressing the *superposition* problem <d-cite key="arora2016linear,olah2020zoom"></d-cite>. In superposition, neurons encode multiple overlapping features, which makes interpretation challenging. SAEs mitigate this by mapping representations into a higher-dimensional, sparsely activated space, enabling the extraction of distinct, interpretable features.

### How Sparse Autoencoders Work

SAEs use an encoder-decoder structure to transform and reconstruct input representations while enforcing sparsity. Given activations $$ z \in \mathbf{R}^d $$ from a neural network, the encoder transforms these into a sparse, high-dimensional representation $$ h $$:

$$
h = \text{ReLU}((z - b_{dec})W_{enc} + b_{enc})
$$

Here, $$ b_{dec} $$ normalizes hidden states, and $$ W_{enc} $$ maps the activations. The decoder reconstructs the input from the sparse representation:

$$
\text{SAE}(z) = h W_{dec} + b_{dec}
$$

The training objective minimizes the following loss:

$$
\cL(z) = \|\mbz - \text{SAE}(\mbz)\|_2^2 + \alpha \|\mbh(\mbz)\|_1
$$

The first term ensures accurate reconstruction, while the $$ L_1 $$ norm encourages sparsity, making each dimension correspond to a distinct feature.

<details markdown="1">
<summary><b>Example</b></summary>

Consider a vision language model where internal activations encode multiple concepts (e.g., visual concepts and language semantics). For instance, an activation might simultaneously encode features like “cat” (visual concept) and “playful” (language concept), making interpretation difficult. By applying a Sparse Autoencoder (SAE), these entangled representations are transformed into a high-dimensional sparse space, where each dimension uniquely captures a specific feature, such as “furry animal” or “expressive tone.”

</details>
<br>

### Key Findings from Existing Works

- **Language Models**: SAEs have been successfully applied to large language models like Claude 3 <d-cite key="templeton2024scaling"></d-cite>, GPT-4 <d-cite key="gao2024scaling"></d-cite> and LLaMA-3.1 <d-cite key="he2024llama"></d-cite>, enabling the discovery of distinct patterns in how these models encode syntax, semantics, and other linguistic features.
- **Vision Transformers (ViTs)**: Researchers have begun using SAEs to analyze ViTs <d-cite key="joseph2023vit,DBLP:conf/eccv/RaoMBS24"></d-cite>. Early results suggest that SAEs can extract interpretable image features, such as object boundaries and textures, using less data compared to their application in language models.

However, Sparse Autoencoders have not yet been applied to vision language models.

### Method Variants and Limitations

- **Variants**: 
	- TransCoders <d-cite key="dunefsky2024transcoders"></d-cite> and CrossCoders <d-cite key="lindsey2024sparse"></d-cite> extend SAEs by incorporating cross-layer and cross-model feature analysis, enabling comparisons both within layers and across different models.
- **Limitations**: 
	- Applying SAEs to large-scale models is computationally expensive due to the increased dimensionality of the sparse space.

<aside class="l-body box-note" markdown="1">
Key Takeaways:
- Sparse Autoencoders offer a powerful method for addressing the superposition problem in neural networks by transforming complex representations into sparse, interpretable features.
- They have been successfully applied to both language models and vision transformers, uncovering hidden patterns in neural representations, but have yet to be explored in vision language models.
</aside>



<br>
## Automated Explanation

### What is Automated Explanation

*Automated explanation* methods aim to make neural networks more interpretable by **translating their abstract representations into human-understandable concepts**. Unlike traditional methods that emphasize identifying important features in the input space, automated explanations focus on uncovering the meaning behind these features. These methods minimize reliance on manual analysis, bridging the gap between mathematical representations and human intuition.

### How Automated Explanation Works

There are two primary approaches to automated explanation:

- **Text-image space alignment**: This type of method establishes connections between visual features and natural language descriptions by **mapping activations into a shared semantic space**. This enables the discovery of interpretable concepts that explain model behavior.

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/concept.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Text-image space alignment aims to find concepts to match with the model's internal representations.
</div>
<br>

<details markdown="1">
<summary><b>Example: TextSpan <d-cite key="gandelsman2023interpreting"></d-cite></b></summary>

- **Goal**: To reveal interpretable text-labeled bases for outputs of vision encoder.
- **Process**:
  1. Cache vision encoder attention head outputs.
  2. Use a predefined text bank to greedily select text descriptions that maximize explained variance.
  3. Analyze discovered attention heads to identify interpretable properties like "color" or "counting."
</details>
<br>

- **Data distribution-based analysis**: This type of method explores patterns in neuron activations across diverse input types to reveal specialized neurons or components. This approach uses either supervised or unsupervised to explain the underlying distribution of neural activations.

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/automated.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Data distribution-based analysis uses supervised or unsupervised methods to explain a distribution of most activating examples into natural language concepts.
</div>
<br>

<details markdown="1">
<summary><b>Example</b></summary>

**Supervised Approaches**  

Supervised methods use concept-labeled data to guide the interpretation of neural network components. These methods identify components that consistently activate strongly when presented with specific input types. For example:

- A neuron that activates strongly for images of cats but remains inactive for other inputs can be classified as specialized for detecting "cat" features.

While supervised approaches provide clear and verifiable concept mappings, their reliance on labeled data limits scalability and may miss concepts not included in the predefined set.

**Unsupervised Approaches**  

Unsupervised methods take a data-driven approach, discovering meaningful patterns in neural activations without requiring labeled data. These techniques use clustering or dimensionality reduction to group similar activation patterns and identify components’ functions. 

Recent advances integrate language models or vision language models to automatically generate natural language descriptions of discovered patterns, offering greater flexibility in concept discovery <d-cite key="DBLP:conf/blackboxnlp/HuangGDWP23"></d-cite>. 

However, ensuring the meaningfulness and reliability of these concepts remains challenging for practical applications.

</details>
<br>

### Key Findings from Existing Works

Automated explanation methods have led to several notable discoveries:

- **TextSpan**: Identified specialized attention heads in vision encoders responsible for features like "color" and "counting." This enabled targeted interventions, such as reducing spurious correlations and improving property-based image retrieval <d-cite key="gandelsman2023interpreting"></d-cite>. Building upon this foundation, Balasubramanian et al. <d-cite key="balasubramanian2024decomposing"></d-cite> extended TEXTSPAN's applicability beyond CLIP to ViTs. Their proposed automated representation decomposition method to analyze the computational graph generated during the forward pass. Using this method, they break down internal contributions of models into their final representation and mapping these components to CLIP space, where they then use TEXTSPAN for text-based interpretation.

- **Neuron Specialization**:
   - **Language-Specific Neurons**: Tang et al. <d-cite key="tang2024languagespecific"></d-cite> discovered neurons in LLMs that activate exclusively for language-related tasks.
   - **Domain-Specific Neurons**: MMNeuron <d-cite key="huo2024mmneuron"></d-cite> revealed neurons specialized for particular domains in vision language models. Interestingly, deactivating these neurons often had minimal effect on task performance, suggesting that VLMs rely on generalized representations.
   - **Modality-Specific Neurons**: Miner <d-cite key="huang2024miner"></d-cite> further refined the methodology to find modality-specific neurons. They also reveal that modality-specific neurons are primarily concentrated in shallow layers, with most modality information remaining within its original token set.

- **Sparse Autoencoders (SAEs)**: Rao et al. <d-cite key="DBLP:conf/eccv/RaoMBS24"></d-cite> used cosine similarities between decoder weights and word embeddings to map neural features to human-understandable concepts, providing interpretable sparse representations.

- **SpLiCE**: Bhalla et al. <d-cite key="bhalla2024interpreting"></d-cite> introduced sparse mappings that align neural features with a comprehensive semantic vocabulary in the CLIP embedding space, ensuring concise yet informative concept representations.

- **MAIA**: Shaham et al. <d-cite key="DBLP:conf/icml/ShahamSWRHA024"></d-cite> developed an automated framework for hypothesis-driven interpretability. MAIA iteratively tested hypotheses, answering queries such as identifying neurons selective for specific features like "forested backgrounds."


### Method Variants and Limitations

- **Limitations**: 
	- Ensuring that discovered concepts are both meaningful and reliable for practical applications is challenging, as concepts that seem plausible to humans may not faithfully reflect internal processing mechanisms <d-cite key="atanasova-etal-2023-faithfulness,agarwal2024faithfulness,madsen2024selfexplanations"></d-cite>.
	- Many methods are developed and tested on specific architectures (e.g., CLIP, ViTs). Their applicability to other architectures remains underexplored.

<aside class="l-body box-note" markdown="1">
Key Takeaways:
- Automated explanation methods connect neural network representations with human concepts, reducing reliance on manual analysis.
- Two main approaches—text-image space alignment and data distribution-based analysis—form the foundation of this field.
- Recent advances, such as TextSpan and MAIA, demonstrate the potential of these methods to uncover nuanced model behaviors and improve interpretability.
</aside>





<br><br><br>
# Future Directions

  

While the above mechanistic interpretability studies have provided significant insights into how vision language models (VLMs) function, several challenges remain. This section discusses and summarizes these challenges and proposes potential directions for future research.

  

### From Single Model to Multiple Models

  

**Current Situation**: Unlike large language models (LLMs), vision language models (VLMs) exhibit much greater heterogeneity in terms of architectures, data, and training paradigms. For instance, VLMs can differ significantly in their vision encoders <d-cite key="li2023monkey,ye2023ureader,xue2024xgenmm"></d-cite>, language models <d-cite key="laurençon2024matters"></d-cite>, and the connectors between them—ranging from simple linear layers to visual resamplers or cross-modal attention mechanisms <d-cite key="liu2023visual,NEURIPS2022_960a172b,awadalla2023openflamingo,DBLP:conf/nips/LaurenconSTBSLW23,dubey2024llama"></d-cite>. They also vary in their training data, which may include image-captioning datasets, visual instruction tuning data, or interleaved image-text datasets <d-cite key="laurençon2024building"></d-cite>. Additionally, their training paradigms differ, such as whether they perform alignment <d-cite key="rafailov2023direct,sun2023aligning,yu2023rlhfv,chen2023dress"></d-cite>, or whether the vision encoder is frozen or fine-tuned during training <d-cite key="Qwen-VL,Qwen2VL,lu2024deepseekvl"></d-cite>. This substantial heterogeneity may limit the transferability of findings if interpretability studies are only conducted on a single model.

  

**Path Forward**: Conducting cross-model analyses is essential to verify conclusions across different VLMs and ensure their generalizability. This approach can help identify universal principles applicable across various VLMs, as well as model-specific insights that could lead to tailored improvements.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Perform cross-model analyses to validate findings across different VLMs and ensure their generalizability.

</aside>

  
  

### From Small Models to Large Models

  

**Current Situation**: Current interpretability research in VLMs primarily focuses on smaller-scale models, such as those with 2B or 7B parameters. However, larger VLMs often exhibit emergent capabilities that are absent in smaller models. These new capabilities may pose unique challenges for applying interpretability tools to larger models.

  

**Path Forward**: Scaling up interpretability studies to include larger models is critical for understanding how these tools perform at scale and what new insights they might uncover <d-cite key="templeton2024scaling,gao2024scaling"></d-cite>. This effort can deepen our understanding of emergent behaviors and inform the development of interpretability methods suitable for larger models.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Scale up interpretability studies to include larger VLMs to understand emergent capabilities and challenges.

</aside>

  

### From Language-Centric to Vision-Centric

  

**Current Situation**: VLMs differ from LLMs in their handling of visual information. While many LLM interpretability tools have been successful in explaining text-based mechanisms <d-cite key="neo2024towards"></d-cite>, applying these tools directly to VLMs may not suffice due to the richer, more ambiguous nature of visual information <d-cite key="joseph2023vit"></d-cite>. Furthermore, VLMs incorporate vision encoders, language models, and connectors between them, adding layers of complexity to interpretability studies.

**Path Forward**: Developing tools specifically designed for visual contexts is necessary to address the unique challenges posed by vision-based features<d-footnote>Although in the past many interpretability methods have focused on vision models like CNNs, VLMs have distinct characteristics.</d-footnote>. Meanwhile, these tools should consider the intricate architectures of VLMs and prioritize analyzing the vision components and vision language connectors, ensuring that interpretations are accurately attributed to the visual inputs. Additionally, input data used for interpretability should emphasize vision-centric tasks that cannot be easily solved by text-only models, ensuring meaningful insights into how VLMs process visual inputs.

  
<aside class="l-body box-note" markdown="1">

Summary:

- Develop interpretability tools tailored for visual contexts and apply them to vision components to understand how VLMs process visual information.

</aside>

  
  

### From Static Processes to Dynamic Processes

  

**Current Situation**: Interpretability studies often focus on a single checkpoint of a model, ignoring the dynamic nature of information flow during training. For example, VLM training typically involves multiple stages, such as initial alignment using image-captioning data (where only the vision language connector is tuned) followed by end-to-end fine-tuning with diverse instruction-tuning data. These stages may include phase changes <d-cite key="wei2022emergent"></d-cite> where models gain new capabilities or behaviors, such as transitioning from unimodal pre-trained models to multimodal systems.

  

**Path Forward**: Studying the dynamics of VLM training is crucial for uncovering novel insights. Applying interpretability tools at different checkpoints during training can shed light on phase changes and the evolution of information flow. Insights from these dynamic studies could also resonate with cognitive science research, such as experiments on restoring vision in previously blind individuals.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Study the dynamics of VLM training to understand phase changes and information flow evolution.

</aside>

  
  

### From Micro-Level to Macro-Level

  

**Current Situation**: Interpretability research often focuses on micro-level phenomena, such as individual neurons or layers, to understand how VLMs process information. However, these findings are rarely connected to macro-level behaviors, such as performance variations across tasks or model designs. For example, recent studies show that CLIP/SigLIP vision encoders pre-trained on image-text data outperform those trained purely on images such as DINO when building VLMs <d-cite key="tong2024eyes"></d-cite>. However, the underlying reasons for these differences remain unclear. Similarly, VLMs can struggle with seemingly simple vision-centric tasks like image classification, despite their vision encoders excelling in such tasks <d-cite key="zhang2024visuallygrounded"></d-cite>.

  

**Path Forward**: Bridging the gap between micro-level findings and macro-level behaviors is essential for driving advancements in VLM development. Applying interpretability tools to investigate unresolved macro-level questions—such as why certain vision encoders perform better or why VLMs struggle with specific tasks—can yield actionable insights. For example, probing tools have been employed to link VLM failures on vision-centric tasks to limitations in the vision encoder. Such findings can inform the design of improved vision encoders, potentially combining the strengths of models like CLIP and DINO to overcome these shortcomings.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Connect micro-level insights from interpretability studies to macro-level behaviors to inform VLM design and development.

</aside>

<br><br><br>
# Conclusion

This work provides a comprehensive review of studies leveraging mechanistic interpretability tools to analyze vision language models (VLMs), including probing techniques, activation patching, logit lenses, sparse autoencoders, and automated explanation methods. These tools have greatly enhanced our understanding of how VLMs represent, integrate, and process multimodal information. Despite these advancements, several key challenges remain. These include the need for validation across a wider range of VLM architectures and training paradigms, a deeper exploration of information flow dynamics throughout training stages, and a stronger alignment between micro-level insights and macro-level behaviors. Addressing these challenges will pave the way for developing more robust and effective VLMs, advancing both their design and practical applications.

