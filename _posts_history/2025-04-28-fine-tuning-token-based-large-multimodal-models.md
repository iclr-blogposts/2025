---
layout: distill
title: "Fine-Tuning Token-Based Large Multimodal Models: What Works, What Doesn’t and What's Next"
description: In this blog post, we explore the advancements and challenges in fine-tuning unified token-based large multimodal models, focusing on the Chameleon architecture and its fine-tuned variant, Anole. Released in 2024, these models exemplify a modern approach for integrating various data modalities through tokens, simplifying modal fusion and leveraging established techniques from large language models. The post details our research efforts to reveal what is important, what is mistaken, and what is worth exploring in future research during the fine-tuning process.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Zhulin Hu
    affiliations:
      name: GAIR Lab, Shanghai Jiao Tong University
  - name: Yan Ma
    affiliations:
      name: GAIR Lab & Fudan University
  - name: Jiadi Su
    affiliations:
      name: GAIR Lab & Fudan University
  - name: Ethan Chern
    affiliations:
      name: GAIR Lab, Shanghai Jiao Tong University
  - name: Pengfei Liu
    affiliations:
      name: GAIR Lab & Shanghai Innovation Institute 
    


# must be the exact same name as your blogpost
bibliography: 2025-04-28-fine-tuning-token-based-large-multimodal-models.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Background
  - name: Introduction to the architecture 
  - name: Explorations in Token-based Large Multimodal Model Fine-Tuning
  - name: Fine-tuning LM_head of the model
    subsections:
    - name: What is LM_head and why we chose to fine-tune it exclusively
    - name: Exp 1-Eliciting image generation capability of Chameleon
    - name: Exp 2-Exploring the Scaling Law Phenomenon in token-based multimodal models
    - name: Exp 3-Attempting to further improve image generation capability of the model by fine-tuning LM_head
  - name: Fine-tuning full parameters of the model
    subsections:
    - name: Advantages and new challenges of full parameter tuning
    - name: Exp 4-Exploring the impact of different data proportions on fine-tuning
    - name: Exp 5-Exploring the impact of Z-loss on training stability
    - name: Exp 6-Exploring improving a model's image generation capability by enhancing the loss function
  - name: Quick start
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
In 2024, a number of works on unified, token-based large multimodal models have emerged, including Chameleon <d-cite key="chameleonteam2024chameleonmixedmodalearlyfusionfoundation"></d-cite>, Transfusion <d-cite key="zhou2024transfusionpredicttokendiffuse"></d-cite>, and EMU3  <d-cite key="wang2024emu3nexttokenpredictionneed"></d-cite>. Modeling various modalities using tokens has become a major research hotspot in large multimodal models. This model architecture offers a more convenient approach to modal fusion, facilitates easier training and fine-tuning, and allows the transfer of established techniques from large language models for optimization. It presents vast research prospects. We delve into this field, focusing on improving image generation quality and enhancing model training stability when fine-tuning. We compiled our research into this blog. In this blog, we aim to:
1. **Introducing the architecture** of a typical unified token-based large multimodal model, Chameleon, and a fine-tuned model based on it, Anole. 
2. **Share our intriguing findings and experiences** during the fine-tuning process of this model.
3. **Highlight the challenges** we encountered in our research and **propose several questions** worth exploring in the context of unified token-based multimodal models.
4. Provide a clear and concise command-line guide for researchers interested in this field to **quickly get started with using the model for inference or training**.

We aim for our blog to be a valuable resource for both newcomers and seasoned researchers in the field. Our goal is to help newcomers quickly get up to speed while offering in-depth insights for those already engaged in this area. Additionally, we hope this work will raise awareness about unified token-based large multimodal models and promote collaboration and openness in large multimodal model research.

## Background
In recent years, large multimodal models that integrate images and text have achieved significant advances in the field of artificial intelligence. These models exhibit remarkable capabilities in tasks related to cross-modal understanding and generation. By effectively combining visual and linguistic information, they adeptly handle complex tasks such as image captioning, visual question answering, and cross-modal retrieval. The primary concept behind these models is the alignment of image and text representations <d-cite key="radford2021learningtransferablevisualmodels"></d-cite>. A notable example of a large multimodal model architecture is LLaVA <d-cite key="liu2023visualinstructiontuning"></d-cite>. This model extracts features from input images using the CLIP visual encoder, projects these features into the language feature dimensions, and then processes them using a large language model.

Recently, unified, token-based model architectures have gained significant attention, leading to the development of innovations like Chameleon <d-cite key="chameleonteam2024chameleonmixedmodalearlyfusionfoundation"></d-cite>, Transfusion <d-cite key="zhou2024transfusionpredicttokendiffuse"></d-cite>, Show-o <d-cite key="xie2024showosingletransformerunify"></d-cite>, DreamLLM <d-cite key="dong2024dreamllmsynergisticmultimodalcomprehension"></d-cite>, MIO <d-cite key="wang2024miofoundationmodelmultimodal"></d-cite>. Unlike architectures such as LLaVA, the token-based multimodal model does not convert other modalities into the text modality. Instead, it represents all modalities using tokens and employs a unified model to generate new tokens, thus effectively achieving seamless modality fusion. This approach provides a straightforward method for modality integration and could potentially be a key pathway towards creating the ultimate large multimodal model. 

Moreover, in this architecture, large multimodal models, similar to large language models, utilize the "next token prediction" paradigm. This allows established practices from large language models, such as inference acceleration techniques, supervised fine-tuning, and direct preference optimization, to be directly applied to these new models. Consequently, this model architecture presents numerous avenues for in-depth research and exploration.

## Introduction to the architecture
In our research, we selected the Chameleon model <d-cite key="chameleonteam2024chameleonmixedmodalearlyfusionfoundation"></d-cite> as our foundational architecture. Released in May 2024, Chameleon represents a significant advancement in unified, token-based large multimodal models. Its architectural backbone mirrors the auto-regressive model structure of the LLaMa series <d-cite key="touvron2023llama2openfoundation"></d-cite>, making it an ideal choice for those seeking to delve into unified token-based multimodal models. Figure 1 illustrates the architecture of the Chameleon model, which utilizes VQGAN <d-cite key="esser2021tamingtransformershighresolutionimage"></d-cite> to translate input images into discrete tokens. This approach unifies the modeling of both text and image modalities, enabling the model to generate text and image tokens in an autoregressive manner akin to a language model. Subsequently, these generated tokens are decoded by the pertinent decoder, allowing for the seamless generation of content across both modalities.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 1: The whole achitecture of Chameleon
</div>
The current open-source version of the Chameleon model lacks the capability to generate interleaved text-image, as noted in the [issues of its official GitHub repository](https://github.com/facebookresearch/chameleon/issues). This limitation has dampened the enthusiasm of many researchers who were eager to explore its potential. In response, we have successfully unlocked the Chameleon model's image generation capability without compromising its performance in language and image understanding, which we'll elaborate on in the next section. Our enhanced version, named Anole, features the ability to create stunning images and seamlessly interweave text and image content. We have open-sourced the Anole model code, including all necessary training and fine-tuning scripts, multimodal inference scripts, and comprehensive documentation. Our code is designed to be fully compatible with the Transformers library, ensuring easy integration and accessibility for researchers working within the Transformers ecosystem.

Built upon the Chameleon model architecture, Anole excels in both understanding and generating content by seamlessly integrating text and images. A unique feature that sets Anole apart from other models supporting text-image multimodal generation is its ability to generate visual information directly using VQGAN decoding, without the need for an additional diffusion module. This model is trained natively on multimodal data, employing a token-based approach that enables intuitive modeling in discrete space, akin to large language models. Furthermore, Anole's multimodal generation capability are entirely open-source. Table 1 below compares the Anole model with other models in the realm of text-image multimodal generation.

| Models              | without Diffusion | native | token-based | fully open-sourced |
|---------------------|-------------------|--------|-------------|---------------------|
| MM-Interleaved <d-cite key="tian2024mminterleavedinterleavedimagetextgenerative"></d-cite> | ×                 | ×      | ×           | √                   |
| Emu2 <d-cite key="sun2024generativemultimodalmodelsincontext"></d-cite>            | ×                 | ×      | ×           | ×                   |
| DreamLLM <d-cite key="dong2024dreamllmsynergisticmultimodalcomprehension"></d-cite>       | ×                 | ×      | ×           | ×                   |
| AnyGPT <d-cite key="zhan2024anygptunifiedmultimodalllm"></d-cite>         | ×                 | ×      | √           | √                   |
| Chameleon <d-cite key="chameleonteam2024chameleonmixedmodalearlyfusionfoundation"></d-cite>             | √                 | √      |√           | ×                   |
| Anole                       | √                 | √      | √           | √                   |

**Table 1: The characteristics of the Anole model compared to other popular multimodal models by the end of June 2024**

## Explorations in Token-based Large Multimodal Model Fine-Tuning
While exploring token-based fine-tuning of large multimodal models based on the Chameleon model, we encountered several research questions,which are outlined below:

Q1：**How to elicit Chameleon's image generation capability?**

As mentioned earlier, the open-source Chameleon has capability limitations. Based on our analysis and practice as demonstrated in Exp 1, we ultimately activated the model's image generation capability by fine-tuning the model's LM_head.

Q2: **Do larger foundational models possess a more powerful image generation capability?**

We used the same method to stimulate the image generation capability of the 7B base model and the 30B model. On this basis, we want to explore the possibility of the existence of scaling laws in token-based large multimodal models. In Exp 2, we present a comparison of the differences in the visual modality among models of different sizes.

Q3: **Since the model's ability to generate images has been elicited by fine-tuning the LM_head, is it possible to further enhance the model's image generation capability using this method?**

In Exp 3, we attempted to further enhance the image generation quality of the model by increasing both the quantity and quality of the dataset, while using the same fine-tuning approach. During the experiment, we observed many interesting phenomena. Ultimately, our experiments revealed that merely fine-tuning a small number of parameters in the model might not significantly improve its image generation capability.

Q4: **When fine-tuning token-based large multimodal model with full parameters, have we encountered any notable techniques or discoveries?**

When we discovered that fine-tuning only the LM_head created a bottleneck in improving the model's performance, we began experimenting with full parameter fine-tuning. After a series of experiments, we have shared some findings in full-parameter fine-tuning that can be used to improve training stability and enhance the model's capability. In Exp 4, we found that **data proportion** plays a vital role in the fine-tuning of token-based large multimodal models. In Exp 5, we introduced **Z-loss** <d-cite key="debrébisson2016zlossshiftscaleinvariant"></d-cite> to enhance the stability of the fine-tuning process. In Exp 6, we improved the model's image generation capability after full-parameter fine-tuning by **refining the loss function**.

The subsequent sections of this blog will provide a detailed presentation of our experimental methods and results. We hope that some of these research findings can address the researchers' questions or provide them with new insights.

## Fine-tuning LM_head of the model

### What is LM_head and why we chose to fine-tune it exclusively

The open-source version of the Chameleon model is only capable of generating content in the text modality, which restricts its applicability in multimodal tasks. Therefore, our primary goal is to elicit the model's ability to generate images, enabling it to process and generate visual content. 

In our preliminary tests assessing the model’s capabilities, we found that the Chameleon model demonstrates an understanding of multimodality by effectively using the text modality to describe given images. This suggests that, within an early-fusion large multimodal model, the model's visual modality encoder and Transformer layers are operating correctly.

Based on these observations, we hypothesize that the **LM_head layer**, located after the Transformer layers, could be **the key factor restricting its ability** to generate in a multimodal fashion. The LM_head layer is a **linear layer** that maps the hidden states produced by the Transformer into vocabulary logits. Its weights constitute a two-dimensional matrix, as illustrated in Figure 2. Tokens numbered from 4 to 8197 are designated as image tokens, while the remaining tokens represent the text modality. Obviously, the values in this matrix at the image token are noticeably abnormal, as if noise has been added.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure2.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 2: The heatmap of Chameleon's LM_head and Anole's LM_head. We found a numerical anomaly issue in the LM_head layer of the open-source Chameleon model.
</div>

### Exp 1-Eliciting image generation capability of Chameleon

We focused our efforts on fine-tuning the model by freezing all parameters except for those in the LM_head. Specifically, we aimed to preserve the integrity of the text modality while enhancing the model's image generation capability. To achieve this, we only fine-tuned the parameters associated with image tokens within the LM_head, as indicated by the red section of the matrix mentioned earlier.

We meticulously selected 6,000 images from the LAION-5B art dataset <d-cite key="schuhmann2022laion5bopenlargescaledataset"></d-cite> for fine-tuning, utilizing cross-entropy loss. After training for three epochs, the loss curve converged, indicating a stable training process. Initially, the LM_head of the Chameleon model exhibited abnormal values at the image token, but these discrepancies were resolved post fine-tuning. Our multimodal inference tests confirm that the model can now successfully generate images.

We also attempted to fine-tune the Chameleon model with **full parameters** under the same configuration. However, the fine-tuned model only produced striped images and failed to fully restore its capability for image generation. This outcome underscores the advantages of our method. A schematic of our approach and examples of the generated images are displayed in Figure 3 and Figure 4 below.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure3.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 3: Diagram illustrating the fine-tuning method used by the Anole model
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure4.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 4: More examples of images generated by Anole, including text-to-image and interleaved text-image
</div>

### Exp 2-Exploring the Scaling Law Phenomenon in token-based multimodal models

Our earlier experiments focused on the Chameleon-7B model, and we extended our investigation to the Chameleon-30B model by utilizing the same fine-tuning configurations to explore its image generation capability. When comparing images produced by the 30B and 7B models using identical prompts, the 30B model delivers noticeably superior quality, marked by richer details. As demonstrated in Figure 5, images generated by the larger model capture more elements from the prompts, such as the sunset in the first image and the rising smoke in the second. Additionally, the larger model tends to create more polished images, as evidenced by the depiction of the coffee cup in the third image. These observations suggest that early-fusion large multimodal models with auto-regressive architectures **might exhibit a phenomenon akin to the scaling law**, a concept extensively validated in the realm of large language models. This hypothesis presents an intriguing avenue for further in-depth research by scholars in the field. We also conducted a simple image generation test comparing the 30B and 7B models under the same training conditions. On the Geneval benchmark <d-cite key="ghosh2023genevalobjectfocusedframeworkevaluating"></d-cite>, the 30B model achieved approximately a 10% overall performance improvement over the 7B model.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure5.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 5: Comparison of images generated by 7B model and 30B model under the same prompt
</div>

### Exp 3-Attempting to further improve image generation capability of the model by fine-tuning LM_head
After initially enhancing the model's image generation capability with 6,000 images, we moved on to fine-tune a few parameters in the LM_head layer. Subsequently, we expanded the dataset to include 30,000 high-quality images for further refinement. **Despite this additional fine-tuning, we observed no noticeable improvement** in the quality of the images produced by the model. As illustrated in Figure 6, with sufficient training, the model is capable of generating visually appealing images. However, it continues to **struggle with effectively following instructions**. For instance, when we aimed to generate a city nightscape that included sidewalks and pedestrians in the second image, the final result failed to depict any pedestrians.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure6.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 6: Examples of images generated by Anole fine-tuned with more images. Although the images still look exquisite, the overall quality has not improved significantly.
</div>

Furthermore, our observations **indicate that insufficiently trained models compromise the completeness of generated images**. As depicted in Figure 7, a model lacking adequate training often produces images with horizontal disruptions. For instance, in the fifth image, where the prompt was "Under the bright sun, dinosaurs strolling in Times Square, with people taking photos and traffic passing by," the generated image clearly displays a dividing line in the middle. Two similarly colored dinosaurs are positioned above and below this line. This issue may stem from the raster scan order of image token generation.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure7.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 7: Examples of images generated by Anole trained insufficiently.
</div>

We are striving not only to expand the dataset's **size** but also to enhance its **quality**. To further boost the model's performance, we incorporated an additional 100,000 images from the JourneyDB <d-cite key="sun2023journeydbbenchmarkgenerativeimage"></d-cite> dataset. Our goal is to improve the model's image generation capability by utilizing the high-quality synthetic data from MidJourney.

Despite a substantial increase in dataset size and diversity, the model’s performance did not meet expectations. Persistent issues, such as difficulty in accurately following instructions and generating low-quality images, remain unresolved.

As demonstrated in Figure 8, the image quality produced by the fine-tuned Anole model has shown some improvement. The model generates more accurately detailed faces and better-shaped waterfalls in the examples. However, it still does not fully meet all the requirements specified in the instructions, such as depicting 'holding a guitar' in the first figure. More concerning, for the high-quality data from JourneyDB (as seen on the right side of the image below), the fine-tuned model **did not exhibit any significant enhancements** in generating images from the training set.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure8.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 8: Comparison of images generated by Anole and Anole fine-tuned with JourneyDB.
</div>

During the fine-tuning process of the LM_head layer of our model, we observed notable results. The model demonstrated an ability to generate detailed images for specific categories, such as animals. However, it showed weaknesses when generating images for certain other categories. We speculate that the frozen model backbone contains a substantial amount of multimodal knowledge, and fine-tuning the LM_head can only elicit this capability without introducing new knowledge. This explains why, after fine-tuning on various datasets, the model consistently performs well on specific categories but struggles with others, including categories present in the fine-tuning dataset.

Thus, we conducted an experiment where we fine-tuned the LM_head layer of the Chameleon model, which originally lacked image generation capability. Using only four facial images for training until convergence, the model produced remarkable outcomes, as shown in Figure 9. Despite the training dataset containing solely human faces, the fine-tuned model was capable of generating images of camels, wine glasses, dogs, and valleys—categories that were not included in the training data.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure9.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 9: We found that even when fine-tuning with only a few facial images above, the model can still generate images of various categories.
</div>

We hypothesize that **during pre-training, the unmodified Transformer layers had already accumulated a substantial amount of multimodal knowledge**. Fine-tuning the LM_head layer serves to elicit the **model's existing image generation capability**, but it is challenging to introduce new knowledge or concepts through this layer alone. Based on this insight, we conducted further experiments, **attempting full parameter fine-tuning of the model**. Our goal was to more comprehensively adjust the internal weight of the model, thereby enhancing its ability for cross-modal generation and knowledge acquisition.

## Fine-tuning full parameters of the model

### Advantages and new challenges of full parameter tuning

Our objective is to refine the model's internal weights through full parameter tuning, enabling it to assimilate new knowledge and improve its performance in multimodal generation tasks. Nevertheless, a significant challenge of full parameter tuning is **ensuring that the model retains its original capabilities** to the greatest extent possible as it **integrates new knowledge**. This challenge is particularly pronounced in multimodal scenarios, where the model must consistently perform well in tasks involving image generation, text generation, and those that combine both. An unstable tuning process could lead to diminished performance in certain tasks, which underscores the importance of achieving stability in the process.

To tackle this issue, we experimented with **adjusting the proportions** of the training data to ensure optimal task performance across different modalities. Additionally, we **implemented the Z-loss method** <d-cite key="debrébisson2016zlossshiftscaleinvariant"></d-cite>. This technique aids in mitigating numerical disparities in logits across modalities, thereby enhancing the stability of the training process. Finally, we also attempted to **improve the loss function** to enhance the quality of images generated during full parameter fine-tuning.

### Exp 4-Exploring the impact of different data proportions on fine-tuning

During the full parameter fine-tuning of multimodal models, we discovered that **using only single-modal data (such as pure text or pure images) significantly weakens the model's generalization ability across other modalities**. This results in bad performance on cross-modal tasks. To maintain the model's competency across various modalities, it is crucial to **include a diverse mix of multiple modalities in the training data**. This should encompass tasks like text-to-text, text-to-image, and image-to-text, among others. Such mixed-modal training enables the model to better understand and leverage the interactions between different modalities, thereby enhancing its performance across a broad spectrum of tasks.

In our experiments, we attempted to fine-tune the Anole model, which has multimodal capabilities, using purely text data. As a result, the fine-tuned model was restricted to performing text-to-text tasks and lost its proficiency in other multimodal functionalities. Conversely, when the model was fine-tuned with a combination of text-to-text and text-to-image tasks, it maintained its capabilities in text-to-text, text-to-image, and image-to-text tasks. However, it was no longer able to generate interleaved text-image content.

**When deciding whether to incorporate interleaved text-image data in training**, it's important to consider the specific requirements of your task. If the task involves generating interleaved content, such as comics or image captions, including this type of data can significantly enhance the model's performance in producing mixed content. Conversely, if interleaved generation is unnecessary for your application, you can forego including this data. It's worth noting that recent research <d-cite key="lin2024vilapretrainingvisuallanguage"></d-cite> indicates that incorporating an excessive amount of interleaved text-image data can **potentially degrade the model's overall capabilities**, especially its zero-shot performance in multimodal contexts.

To scientifically determine the optimal ratio of multimodal training data, **further exploration is necessary**. This is a critical issue that significantly influences model performance. When designing data composition, it is essential to consider not only the specific task requirements but also the potential negative effects of various data types. By assessing the impact of different data types on the model's capabilities, one can ensure both effective generalization and robust task performance.

### Exp 5-Exploring the impact of Z-loss on training stability

We have introduced the **Z-loss** mentioned from the Chameleon paper to address the logit shift issue. The Z-loss is used to regularize the denominator Z of the softmax function (1). The final form of the loss function is as follows.

$$
\sigma(x)_i = \frac{e^{x_i}}{Z}, \quad Z = \sum_{i} e^{x_i} \quad\quad (1)
$$

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cross_entropy}} + 10^{-5} \log^{2} Z \quad\quad (2)
$$

Recent research claims that Z-loss helps the training loss to converge <d-cite key="liu2024luminamgptilluminateflexiblephotorealistic"></d-cite>. When the training data consists solely of a single modality, the model can be optimized smoothly without the introduction of Z-loss, without encountering significant numerical instability issues. Therefore, the role of Z-loss is rather limited in this scenario. However, when the training data includes multiple modalities, such as images and text, the varying magnitude of logits across different modalities can lead to instability if optimized directly. **In such multimodal scenarios, adding Z-loss effectively balances the differences between logits across modalities**, alleviating numerical inconsistencies and enhancing model stability and training performance. This finding is particularly valuable for training large multimodal models and can help improve performance when dealing with heterogeneous data.

Nevertheless, simply adding Z-loss did not effectively enhance the performance of the fine-tuned model in our experiment. As shown in Figure 10, after fine-tuning with 10,000 text-to-text and text-to-image mixed sample in a 1:1 ratio, the model based on Anole with Z-loss did not show a significant decrease in the loss function, nor did it lead to noticeable improvements in image generation quality. The reasons for this experimental failure might be that we did not find more effective or suitable fine-tuning methods, such as precise hyperparameter tuning, insufficient training data, or mismatched data types. Additionally, the Chameleon architecture itself may **exhibit instability during training, making it challenging to train effectively**. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure10.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 10: Comparison of the Anole model's loss curves and generated images before and after adding Z-loss.
</div>

In summary, maintaining training stability during the full parameter fine-tuning process is a challenging and worthwhile area of exploration for a large multimodal model utilizing an early fusion architecture. Coordinating the fusion of multimodal information during fine-tuning and optimizing training strategies to ensure model stability is not only crucial for enhancing the performance of multimodal models but also an important research direction for advancing this field further. Continuing to explore and find more effective fine-tuning methods in future work will have a significant impact on evolution and application of large multimodal models.

### Exp 6-Exploring improving a model's image generation capability by enhancing the loss function
In our experiments focused on enhancing image generation capability, we discovered that fine-tuning only the LM_head layer does not lead to significant improvements in image quality, image-text alignment accuracy, or consistency with the style of images in the training dataset. The underlying reason is that when solely the LM_head layer is fine-tuned, the crucial parameters of the model's Transformer backbone remain unchanged, limiting the model's flexibility and adaptability. This restriction hinders the model's ability to fully explore and optimize its multimodal generation capabilities, ultimately creating a bottleneck in enhancing image quality. To overcome this issue, we proceeded to conduct full parameter fine-tuning on the Anole model.

During full parameter fine-tuning with text-to-image data, we encounter an unexpected challenge: while using standard cross-entropy loss, the training loss decreases significantly, but the quality of generated images deteriorates, and there's a risk of mode collapse. As illustrated in Figure 11, the training loss drops dramatically when compared to fine-tuning only the LM_head, going from 5 to 5e-3. However, this improvement in loss does not translate to better image quality. We hypothesize that the **cross-entropy loss function may be ill-suited for image generation tasks**. Due to its local nature, it can lead to numerous local minima on the loss surface, making the optimization process prone to getting stuck. This limitation hinders the quality of the output images. Consequently, we aim to enhance the model fine-tuning process by **developing a more suitable loss function** for training, ultimately seeking to improve the quality of generated images.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure11.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 11: After full parameter fine-tuning, the images generated by the model are completely chaotic despite the low loss.
</div>

To address the aforementioned issues and improve image generation quality, **we propose a method based on reconstruction loss**. During training, we map the entire generated image into the feature space of VQGAN codebook features. We then use the codebook features of the original images from the training set as supervision labels for supervised learning. This approach allows the model to optimize the loss function not by focusing only on local information, but by making adjustments and optimizations from a global perspective for the entire image generation. Such a holistic method better captures the overall structure and details of the image, thereby enhancing the quality and consistency of the generated images. The specific training process is illustrated in Figure 12, which includes the complete process from feature extraction to loss calculation, ensuring that the model can effectively optimize each generation step during training. The final loss function we use is $$\mathcal{L} = \alpha \mathcal{L}_{\text{reconstruction}} + \mathcal{L}_{\text{cross_entropy}}$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure12.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 12: Diagram illustrating how we introduce the reconstruction loss.
</div>

We conducted experiments using a small dataset, which consists of 2,000 images. A smaller dataset is more likely to increase the risk of overfitting during full parameter fine-tuning of the model. In this setting, conducting experiments makes it easier to observe the effect of adding this loss. The results, as illustrated in Figure 13, show that as the weight of the reconstruction loss increases, the value rapidly decreases, while the cross-entropy loss experiences a slight increase. **When the weight of the reconstruction loss reaches 0.5, the generated images effectively avoid mode collapse**, displaying more diverse and stable image quality. However, if the reconstruction loss weight becomes too high, the images begin to appear overly smooth, resulting in a loss of detail and an overall decline in quality. Therefore, the experiments indicate that setting the reconstruction loss weight within an optimal range can effectively enhance the quality of image generation by the model, avoiding the issues that arise from relying solely on cross-entropy loss.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-fine-tuning-token-based-large-multimodal-models/figure13.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
  Figure 13: Results demonstrating the effect of reconstruction loss.
</div>

Currently, we are further experimenting with the specific values of the weights and planning to expand the dataset to examine whether improving the loss function can significantly enhance image generation performance during full parameter fine-tuning. Through full-parameter fine-tuning, our 7B model achieved a score of over 0.39 on the Geneval benchmark, demonstrating image generation capabilities comparable to those of the unreleased Chameleon 7B model. Future experiments will also focus on the impact of different loss weight combinations on generation quality, aiming to identify the optimal parameter configurations that work well across various tasks and datasets. Improving the loss function is also a worthwhile area to explore in multimodal generation.

Overall, although all modalities are encoded into tokens in a token-based architecture, we can optimize the training process by applying additional loss functions to specific modalities that are more difficult to train. To optimize image generation, some studies have explored new loss functions <d-cite key="li2025imaginereasoningspacemultimodal"></d-cite> or adopted approaches such as diffusion to better model the image generation process. <d-cite key="zhou2024transfusionpredicttokendiffuse"></d-cite>

## Quick start

To enable researchers to more quickly start working with token-based unified multimodal large models, we have open-sourced the weights of the Anole model, along with inference scripts, training scripts, and the dataset for fine-tuning. With the following code, researchers can quickly install, use our model, and perform inference and training.

First, download our [repository](https://github.com/Kenny1004/anole-hf) and install the relevant environment.

```bash
#Download repo and install environments
git clone ...
cd anole-hf
bash install.sh
```

Next, download the Hugging Face weights for the model. You can choose to download either the Chameleon or Anole weights. Due to double-blind reviews, we do not provide the weight download links here.
Now, you can use the model for inference. The code supports input and output in any modality, whether image or text. Below is an example.

```bash
###text to image
python text-image.py -i Instruction -b Batch_size -s Save_dir

###Multimodal in & multimodal out
python interleaved_generation -i Instruction -s Save_dir

#the instruction for mulitimodal is a json file, e.g.
#[
#    {
#        "type": "text",
#        "content": "What can I bake with this? Give me the full recipe and show me an image with the final dish. <image>"
#    },
#    {
#        "type": "image",
#        "content": ["./banana.png"]
#    }
#]
```

Our code also supports both few parameters fine-tuning and full parameter fine-tuning as mentioned in the blog. You can run the command below:

```bash
cd training
#set constants such as ckpt path in constants_training.py
#train.py is the code for full parameters fine-tuning
#finetune.py is the code for fine-tuning model's LM head
#You can determine which file to run based on the model size and the type of fine-tuning.
#An example to fine-tuning 30b model's full parameters:
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_30b.py
```

In addition, you can fine-tune using your own collected dataset.

```bash
cd training
#You need to organize the data format in advance.
python data_tokenization.py
```

## Conclusion

In conclusion, this blog post introduced the Chameleon architecture and the fine-tuned anole model, highlighting it as a prominent example of the popular token-based unified large multimodal model architecture. We also discussed our findings and the challenges we faced during the fine-tuning process. Additionally, we are pleased to announce that we have open-sourced both the weights and code for our fine-tuned Anole model. We hope that by sharing these resources, we can assist fellow researchers in advancing the study of unified large multimodal models, and collectively stimulate progress in this exciting field.