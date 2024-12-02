---
layout: distill
title: "Boosting translation capabilities of low resource languages"
description: Large Language Models (LLMs) are increasingly recognized for their potential in translation tasks. Researchers are actively exploring the most effective architectures, prompting formats, and fine-tuning techniques for these tasks. Further, the use of LLMs in low-resource machine translation is still emerging. This study evaluates methods that combine In-Context Learning (ICL) with fine-tuning scenarios where language data is scarce. Our strategy involves two steps -  first, introducing translation examples from languages closely related to the target language second, using embeddings to include similar examples to the source language for few-shot ICL. We hypothesize that this approach helps LLMs access their hidden translation capabilities. We test our methods on two language clusters Konkani and Tunisian Arabic, using models from two LLM families. The first model is tailored for machine translation tasks, though not specifically with data from the target language clusters. The second model is trained in several languages and is known for its multilingual capabilities. Our evaluations show that models fine-tuned with this technique perform well given these constraints. Through this paper, we hope to make LLMs more accessible to communities speaking low-resource languages.

date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Anonymous

bibliography: 2025-04-28-boosting-translation-capabilities-of-low-resource-languages.bib

toc:
  - name: Introduction
  - name: Related Work
    subsections:
      - name: In-Context Learning
      - name: Fine-Tuning
  - name: Methodology
  - name: Experiments
    subsections:
      - name: Language similarity
      - name: Models
      - name: Datasets
      - name: Trainable parameters
      - name: Tokens
      - name: Parameters
        subsections:
          - name: Training Parameters
          - name: Quantization
          - name: LoRA
          - name: Text generation
      - name: Baselines
  - name: Results
  - name: Discussion
  - name: Appendix

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


***



## Introduction

A variety of research and development efforts are focused on increasing linguistic diversity in research datasets and boosting performance for Natural Language Processing (NLP) tasks for languages in low-resource settings. For example, the recent No Language Left Behind (NLLB) effort <d-cite key="team2022NoLL"></d-cite> seeks to create datasets and models for massively multilingual Machine Translation (MT) with the stated aim of "eradicating language barriers on a global scale." Other efforts, such as the Bloom Library datasets project <d-cite key="leong-etal-2022-bloom"></d-cite> and Masakhane <d-cite key="nekoto2020participatory"></d-cite>, seek to establish the first known baselines for languages with data and tooling developed by speakers of those languages. These and other related efforts generally hope to address the inequalities in the performance of NLP tasks, such as machine translation, across the world’s 7,000+ languages <d-cite key="Eberhard2024Ethnologue"></d-cite>.

At the same time, the explosion of interest and research in generative, LLMs has simultaneously demonstrated: (1) impressive multi-task instruction-following capabilities of text generation models, such as Llama 2 <d-cite key="touvron2023llama2openfoundation"></d-cite> or Mistral <d-cite key="Jiang2023Mistral7"></d-cite>, that are fine-tuned using supervised fine-tuning and preference optimization; and (2) the dominance of a small number of resource-rich languages like English and Mandarin in the data used to train these models. Although an increasing number of region-specific and other-than-English-and-Mandarin LLMs are being released (see, for example, the SeaLLM <d-cite key="Nguyen2023SeaLLMsL"></d-cite> and SEA-LION models <d-cite key="lowphansirikul2021wangchanberta"></d-cite> for Southeast Asian languages), the majority of LLM models driving open access model innovation represent a negligible percentage of living languages.

Rather than focusing on the important research directions of dataset creation (to solve data scarcity issues) or language-specific foundation model creation (to increase linguistic diversity in base LLMs), this paper focuses on leveraging the few-shot instruction-following capabilities of LLMs for the task of machine translation in low-resource settings. In cases where there is a desire to translate from a widely spoken language (supported in commercial and open-source MT systems) to a language without high digital language vitality (and not supported in any known commercial and open-source MT systems), we propose leveraging prompts with few-shot examples of machine translation from the source language to both the target language and a higher-resourced, linguistically related pivot language. These prompts demonstrate the adaptation of a machine translation from the pivot language to the target language "in context," such that for new samples (where the adaptation is not known), the LLM generates a likely adaptation to the target language.

Such prompts could be utilized with existing pre-trained models, which would potentially yield useful results if the instruction-following capabilities of the pre-trained LLMs could perform the adaptation task without further fine-tuning. However, prompts formatted in this manner could also be curated into datasets to fine-tune LLMs for the low-resource machine translation task, with the hope that such prompt curation and formatting could reduce the need for large parallel corpora and/or computationally prohibitive training of massively multilingual models.

We demonstrate the approach for two linguistically distinct language clusters: Arabic and Konkani. We find that pre-trained LLMs, with ICL applied, do not produce useful translation in the low-resource language scenario. However, LLMs fine-tuned on small numbers of ICL prompts outperform LLMs trained with zero-shot translation prompts.

We find that when the models are fine-tuned by constructing the prompt as mentioned, they outperform MT benchmarks for these languages.

***

## Related Work


### In-Context Learning

ICL has gained popularity for improving the accuracy of LLM responses in downstream tasks without updating the model's weights. Unlike traditional fine-tuning or pretraining—where model weights are adjusted—ICL allows LLMs to perform tasks by conditioning on input-output examples provided in the prompt. This process enables LLMs to utilize patterns and structures learned during pretraining to generate appropriate responses. Pretraining focuses on next-token prediction, generating the next word based on the preceding context. Through ICL, latent concepts from the pretraining data are activated, adapting the LLM's next-token prediction capabilities to perform tasks it wasn't originally trained for. The extensive training on vast amounts of text equips the LLM to handle a variety of tasks without further weight adjustments. When presented with an ICL prompt, the LLM "locates" previously learned concepts or knowledge within its training data. These concepts act as latent variables, encapsulating various document-level statistics and information that the model was exposed to during its training phase <d-cite key="xie2021explanation"></d-cite>.

However, while ICL has shown promise in data used during pretraining, its effectiveness in low-resource languages that the model was not exposed to during pretraining remains underexplored. Since low-resource languages are underrepresented in pretraining data, LLMs may lack the necessary latent concepts to perform tasks in these languages effectively. Our research aims to address this gap by leveraging fine-tuning techniques in conjunction with ICL to enhance translation quality for low-resource languages like Konkani and Tunisian Arabic. By incorporating linguistically related pivot languages and embedding-based example selection, we seek to activate latent translation capabilities within LLMs, even when data is scarce.

ICL has proven to be useful in translation tasks, particularly through the use of few-shot ICL for translation tasks <d-cite key="moslem2023adaptive"></d-cite>. This approach improves real-time adaptive machine translation and provides a way to enhance translations. They show that the translation quality achieved with few-shot ICL can surpass that of strong encoder-decoder MT systems, especially for high-resource languages. They hypothesize that during the unsupervised pretraining phase, the model develops pattern recognition abilities characteristic of autoregressive decoder-only models. During inference, these patterns are applied to execute translation tasks effectively. They also explore translation from English to Kinyarwanda, which is a low-resource language. This work considers the GPT-3.5 and BLOOM models but does not explore open-source models. They also do not fine-tune the models. They explain that tokenization could be the reason why translations for languages like Arabic and Kinyarwanda do not perform well. In this work, we aim to overcome these limitations by fine-tuning the models and also analyzing how these languages are tokenized.

Translation quality and the type of in-context examples used can significantly impact LLM performance in translation tasks <d-cite key="agrawal2022context"></d-cite>. Their study indicates that a single-shot, noisy, or unrelated example can degrade output quality, while including more relevant examples—selected based on n-gram overlap with the test source—substantially enhances translation quality. This method outperforms a strong kNN-MT baseline in half of the tested out-of-domain datasets. Similar examples to the test source are retrieved from a datastore containing pairs of source text and their corresponding translations. Retrieval is facilitated by BM25, an unsupervised and efficient method that provides additional context to the model. In their work, they mostly focus on selecting relevant examples for few-shot demonstrations and translations to and from German and Russian, which are high-resource languages. In our work, we aim to expand on this by adding a pivot language and further fine-tuning decoder-only models.

### Fine-Tuning

LLMs demonstrate strong translation capabilities even though they are not trained with parallel corpora in the pretraining stage. In the previous section, we primarily focused on improving translations using ICL examples. However, translation performance can also be enhanced by fine-tuning models. The ability of LLMs to execute translation tasks depends on their comprehension of translation instructions and the degree of alignment between different languages <d-cite key="li2023eliciting"></d-cite>. By applying multilingual fine-tuning alongside translation instructions, LLMs can effectively learn to translate even between language pairs that were not seen during the instruction tuning phase. Their fine-tuning approach not only improves translation capabilities for language pairs included in the training but also for new, unseen language pairs. In their work, they use open-source LLM models and limit their data to 1,000 examples per language pair. They also make use of a pivot language, but in a different sense. For example, they use English as a pivot while translating between unseen language pairs to improve translation performance during fine-tuning. In our work, we explore the scenario where the target is an unseen language and the pivot language is a high-resource language. We also propose adding few-shot demonstrations to establish alignment between language pairs.

The technique of including ICL examples during fine-tuning has also proven effective beyond translation tasks. It has been shown that in question-answering tasks, retrieving similar document chunks from embeddings and including them in the prompt of a fine-tuned LLM yields better performance than either method alone <d-cite key="fatehkia2024t"></d-cite>. We aim to apply this technique by retrieving similar examples using a sentence transformer model <d-cite key="reimers-2019-sentence-bert"></d-cite> to determine whether this approach will improve language alignment and, ultimately, translation quality.

***

## Methodology

Our main goal here is to determine if adding a pivot language translations, a language that is similar to the low resource target language into the few shot demonstration will improve translation performance.

{% include figure.html path="assets/img/2025-04-28-boosting-translation-capabilities-of-low-resource-languages/Flowcharts.png" class="img-fluid" %}

We began by evaluating the performance of pretrained language models on the translation tasks for our target languages without any fine-tuning or prompt adjustments. This initial assessment established a baseline and highlighted the limitations of these models in handling low-resource languages. Recognizing the potential of ICL to improve translation tasks, we implemented few-shot learning by including relevant examples in the input prompts during inference. Our goal was to determine whether adding similar sentences as examples would enhance the models' translation performance. To select the most relevant few-shot examples, we employed the following approach. We used the all-MiniLM-L12-v2 sentence transformer model <d-cite key="reimers-2019-sentence-bert"></d-cite> to generate vector embeddings for all sentences in our training dataset. This model was chosen for its balance between computational efficiency and embedding quality. The embeddings were stored in a vector database using the lancedb. For each source sentence to be translated, we generated its embedding and calculated cosine similarity scores with the embeddings in the vector database. We retrieved the top five most similar sentences based on cosine similarity scores. These sentences, along with their translations, were included in the prompt as few-shot examples, formatted consistently with the models' requirements. We utilized two advanced language models for our experiments. Unbabel's Tower models <d-cite key="alves2024tower"></d-cite> are context-aware translation models fine-tuned from Llama 2. Designed to handle contextual information effectively, they are well-suited for translation tasks involving nuanced language structures. Nous Research's Hermes-2-Pro-Llama-3-8B<d-cite key="Hermes-2-Pro-Llama-3-8B"></d-cite> is a multilingual model capable of handling multiple languages, including those not seen during pretraining. Its architecture allows for effective adaptation to new language pairs. When it comes to translation we determine if a model instruction tuned specifically for translation task can perform better than a multilingual model trained on several languages. For each language pair (English-Konkani and English-Tunisian Arabic), we compiled a dataset of 1,000 sentence pairs. The sentences were sourced from publicly available parallel corpora56, ensuring a diverse representation of language use. The training data was formatted according to the prompt structures recommended for each model. Examples of these prompt templates are provided in the Appendix. During fine-tuning, the retrieved similar sentences were included in the prompts to reinforce the models' ability to translate sentences similar to those in the training data. The fine-tuning was conducted on Google Colab Pro using NVIDIA A100 GPUs making this approach accessible.
We used BLEU scores <d-cite key="papineni-etal-2002-bleu"></d-cite>, calculated using SacreBLEU, to assess the translation quality of each model. BLEU scores provide a standard metric for comparing the similarity between machine-generated translations and reference translations.

***

## Experiments

### Language similarity
In this work, we hypothesize that for low-resource target languages, adding a high-resource pivot language—one that the model is likely trained on and shares the same script—can help the model produce more higher quality translations. The choice of a suitable pivot language may depend on factors such as data availability, whether the model has been pretrained on the language, and whether it belongs to the same language family as described in Ethnologue. Quantitatively measuring the similarity between the pivot and target languages to ensure that the data added during in-context learning is relevant, as noisy data can degrade translation quality <d-cite key="agrawal2022context"></d-cite>.

Jaccard Similarity can be used to measure the similarity between the parallel corpus. This metric measures the similarity between two sets, often represented as the ratio of the intersection (shared elements) to the union (total unique elements) of those sets. In this case the measure can be used to check the overlap of words between the pivot language and the target language. Jaccard similarity quantifies the vocabulary or structural overlap, yielding a value between 0 and 1. High Jaccard similarity between two languages typically indicates shared vocabulary. We hypothesize that leveraging this overlap, along with selecting pivot languages from language families classified by Ethnologue, increases the likelihood of accurate translations. 

Although other measures of lexical similarity exists, Jaccard similarity is interpretable and allows to measure the similarity between the parallel corpus for low resource languages where the regionally satandardized wordlist is not available <d-cite key="Eberhard2024Ethnologue"></d-cite>. 

In our case, a Jaccard similarity score of 62 between Marathi and Konkani suggests a substantial overlap, with the two languages sharing about 62% of their linguistic elements, such as words, phrases, or structures. We also calculated the Jaccard similarity between Tunisian Arabic and Modern Standard Arabic, which resulted in a score of 56%. While Jaccard similarity is not an essential criterion, it supports our selection of effective pivot and source languages. In future work, we aim to explore whether correlation between lexical similarity and translation performance exists.

### Models
In this experiment, we evaluate the performance of two models: Unbabel/TowerInstruct-7B-v0.2 <d-cite key="alves2024tower"></d-cite>, and NousResearch/Hermes-2-Pro-Llama-3-8B <d-cite key="Hermes-2-Pro-Llama-3-8B"></d-cite>. 

The Hermes-2-Pro-Llama-3 model is a instruction-tuned version of Llama 3 <d-cite key="dubey2024llama3herdmodels"></d-cite>, known for its multilingual capabilities. Llama 3 supports 8 languages — English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai, although the underlying foundation model has been trained on over 176 languages. Our aim is to see whether we can make use of the latent knowledge alignemnt of the model while translating to low resource languages

The TOWERINSTRUCT model is finetuned from TOWERBASE model. The TOWERBASE model is continuously pretrained from from the Llama 2 model with a mixture of monolingual and parallel data with 20B tokens. The TOWERBASE model is the finetuned with instruction dataset that is relevant to translation process. Some of these tasks include Automatic Post Edition, Context-aware Translation, Named-entity Recognition etc. The languages supported by the TOWERBASE model are  English (en), German (de), French (fr), Dutch (nl), Italian (it), Spanish (es), Portuguese (pt), Korean (ko), Russian (ru), and Chinese (zh). Although TOWERINSTRUCT performs well on translation tasks, it is not expected to excel in languages it was not exposed to during training.

Our strategic selection of these models is designed to assess their effectiveness in translating languages outside their initial training sets. Despite TOWERINSTRUCT’s specialized training in translation, it has not been directly exposed to the specific low-resource languages focused on in this experiment, offering a unique test of its adaptability to unseen languages.

### Datasets
For this experiment, we focused on two low-resource languages. The first is Konkani, an Indian language spoken in the western part of India, with approximately 2.35 million speakers as of 2011. The second is Tunisian Arabic, spoken in Tunisia, with around 12 million speakers as of 2021.

The Konkani parallel corpus was constructed using a dataset open-sourced by AI4Bharat, which also contributed to the training set for the IndicTrans2 model <d-cite key="gala2023indictrans2"></d-cite>. This corpus includes English, Marathi, and Konkani. Despite having access to a larger dataset, we intentionally limited the training set to approximately 900 records to simulate a low-resource language scenario.

Given the experiment’s reliance on a pivot language included in the ICL examples, Marathi was selected as the pivot language due to its wider prevalence in western India and its linguistic similarity to Konkani. The test set for Konkani consisted of 200 records.

For Tunisian Arabic, the corpus was derived from the work described in <d-cite key="bouamor2014multidialectal"></d-cite>, with Modern Standard Arabic chosen as the pivot language. The parallel corpus for Tunisian Arabic contained 1,000 records, with 900 used in the training set and 100 used in the test set.

Our aim was to replicate the translation performance for low-resource languages that typically have limited data available. An additional motivation for selecting these languages was their use of non-Latin scripts. By working with a small training or ICL set of approximately 1,000 records, we aimed to explore whether it is possible to enhance translation performance under such resource constraints.

### Trainable parameters
This work deals with scenarios where only few hundred rows of data is available. In such scenarios we try to see what the model size, finetuning technique should be. With limited data, finetuning methods like prompt tuning (where embeddings are adjusted) or LoRA (Low-Rank Adaptation) prove particularly effective <d-cite key="zhang2024scaling"></d-cite>. With Parameter-Efficient Fine-Tuning (PEFT), even increasing the data yielded modest performance improvements. For instance, using LoRA on the Llama 3 8B model brought the trainable parameters down to 176,242,688, or just 2% of the model’s total parameters. In this work we mainly focus on LORA for finetuning. 

### Tokens 
In this section, we analyze token-related metrics in the training data that may influence model performance for different languages. The token-per-word ratio is an important metric in understanding how effectively a tokenizer captures the linguistic structure of a language, which can influence model performance <d-cite key="remy2024trans"></d-cite>.

In the first table, we observe that for English, the token-per-word count is approximately 1 for both models, as expected. However, for Marathi and Konkani, the TOWERINSTRUCT model exhibits a significantly higher token-per-word count compared to the Nous Hermes Llama 3 model. This suggests that the Nous Research model may produce better translations, which aligns with the findings discussed in the results section.

| Language         | Model                              | Model Total Tokens | Model Unique Tokens | Model Avg Token Length |
|------------------|------------------------------------|--------------------|----------------------|------------------------|
| English          | NousResearch/Hermes-2-Pro-Llama-3-8B | 27,136            | 7,624               | 1.341                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 31,823            | 6,496               | 1.573                 |
| Marathi          | NousResearch/Hermes-2-Pro-Llama-3-8B | 73,641            | 892                 | 4.144                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 139,905           | 128                 | 7.873                 |
| Konkani          | NousResearch/Hermes-2-Pro-Llama-3-8B | 68,777            | 984                 | 4.050                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 128,651           | 132                 | 7.576                 |


For the Arabic language cluster, token counts and other metrics are as follows. Similar to the Marathi and Konkani findings, the Nous Research's Hermes-2-Pro-Llama-3-8B finetune demonstrates a lower token-per-word ratio compared to the TOWERINSTRUCT model, suggesting potentially better translation performance.


| Language         | Model                              | Model Total Tokens | Model Unique Tokens | Model Avg Token Length |
|------------------|------------------------------------|--------------------|----------------------|------------------------|
| English          | NousResearch/Hermes-2-Pro-Llama-3-8B | 13,185            | 2,829               | 1.024                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 14,197            | 2,675               | 1.102                 |
| Modern Standard Arabic              | NousResearch/Hermes-2-Pro-Llama-3-8B | 18,365            | 1,625               | 2.008                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 41,359            | 83                  | 4.523                 |
| Tunisian Arabic  | NousResearch/Hermes-2-Pro-Llama-3-8B | 17,418            | 1,585               | 2.051                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 39,834            | 78                  | 4.752                 |

Across languages, we can observe that the dataset for Konkani translation is richer compared to the dataset for Tunisian Arabic, as evidenced by the higher English token counts. This richness of the dataset will likely influence model performance.

Analyzing these metrics is critical to assess the quality of the dataset and its potential impact on translation accuracy. In future work, it would be valuable to explore the relationship between token counts in the dataset and model performance, especially for translation tasks to unseen languages. Understanding these dynamics could further improve fine-tuning strategies and dataset construction for multilingual models.


### Parameters
Parameter-Efficient Fine-Tuning (PEFT) is computationally more efficient than pure ICL, which led us to adopt PEFT for our model fine-tuning process <d-cite key="liu2022few"></d-cite>. We used the Huggingface Transformers library and fine-tuned the models with the following hyperparameters: 

#### Trainining Prameters

| Parameter                     | Value             |
|-------------------------------|-------------------|
| `per_device_train_batch_size` | 1                 |
| `num_train_epochs`            | 1.5               |
| `warmup_ratio`                | 0.03              |
| `logging_steps`               | 25                |
| `learning_rate`               | 2e-4              |
| `gradient_accumulation_steps` | 1                 |
| `gradient_checkpointing`      | True              |
| `lr_scheduler_type`           | Cosine            |
| `weight_decay`                | 0.001             |
| `save_strategy`               | No                |
| `optim`                       | `paged_adamw_32bit` |
| `warmup_steps`                | 100               |
| `bf16`                        | True              |

These parameters were selected to optimize both computational efficiency and model performance for fine-tuning on limited data.

#### Quantization
The model was loaded in 4-bit precision using the BitsAndBytes library with the nf4 quantization type. The following table summarizes the configuration used for 4-bit quantization:

| Parameter                 | Value                                       |
|---------------------------|---------------------------------------------|
| `Compute dtype for 4-bit base models`` | float16 (Compute dtype for 4-bit base models) |
| `Quantization type``    | nf4    


#### LoRA 
For fine-tuning, we employed the LoRA configuration, as detailed in the table below:

| Parameter         | Value                                                                  |
|--------------------|------------------------------------------------------------------------|
| `r`               | 64                                                                     |
| `lora_alpha`      | 16                                                                     |
| `lora_dropout`    | 0.1                                                                    |
| `bias`            | none                                                                   |
| `task_type`       | CAUSAL_LM                                                              |
| `target_modules`  | ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj', 'lm_head'] |


#### Text generation
The following parameters were used to generate the outpput from the finetuned model during the evaluation

| Parameter               | Value  |
|-------------------------|--------|
| `do_sample`            | True   |
| `temperature`          | 0.1    |
| `num_return_sequences` | 1      |
| `max_new_tokens`       | 200    |
| `return_full_text`     | False  |



### Baselines

To establish baselines, we used commercially available machine translation (MT) tools. For Konkani, we chose Bing Translate as our baseline. However, Tunisian Arabic was not supported by most commercial MT systems.
We also fine-tuned the No Languages Left Behind (NLLB) 1.3B distilled model to establish baseline performance on the given data. Since Tunisian Arabic is already included in the pre-trained NLLB model, we observed that fine-tuning it on our training data slightly improved performance. Fine-tuning was conducted for 2,000 steps with a learning rate of 1e-4 and a weight decay of 1e-3 after 1,000 steps.

***

## Results

BLEU scores, calculated using SacreBLEU, were obtained for each model on the test set, which comprised 200 sentences. This test set was evaluated using commercially available MT tools. For Konkani, Bing Translate served as the baseline, with direct translation from English to Konkani without a pivot language. 

In the prompt for In-Context Learning (ICL), few-shot examples were constructed by selecting sentences that closely resemble the source sentence for translation—in this case, English. The prompt included 5 examples of translations from English to Marathi and then to Konkani. Initially, we assessed the performance of the pre-trained models.

The BLEU scores were calculated for the two models. Without any fine-tuning and using only similar examples as few-shot demonstrations, it was observed that the model often did not produce high-quality translations. The responses contained Latin characters and occasionally included comments on the translations. In these cases the translations were processed further before evaluation. The models were then fine-tuned both with and without the addition of pivot language translations. It was found that incorporating high-resource pivot translations improved the BLEU score and made the translations less noisy.

The results are highlighted below. The more performant general-purpose model, Nous Research's Hermes-2-Pro-Llama-3-8B model, outperformed TOWERINSTRUCT, a fine-tuned version of Llama 2. Notably, these results are based on only around 800 training samples, which suggests that this technique for translations can be widely accessible. Additionally, Konkani was chosen for this study because its script is non-Latin, making it valuable to investigate how language models trained primarily on Latin-character data handle translations for such languages

| **Model**                                 | **BLEU Score** |
|-------------------------------------------|----------------|
| Baseline (Bing Translate)                 | 40.90         |
| Unbabel/TowerInstruct-7B-v0.2                             | 7.28         |
| Unbabel/TowerInstruct-7B-v0.2  fine-tuned without pivot   | 8.39         |
| Unbabel/TowerInstruct-7B-v0.2  fine-tuned with pivot      | 10.56         |
| NousResearch/Hermes-2-Pro-Llama-3-8B                        | 7.26           |
| NousResearch/Hermes-2-Pro-Llama-3-8B fine-tuned without pivot | 13.22       |
| NousResearch/Hermes-2-Pro-Llama-3-8B fine-tuned with pivot  | 18.50         |

**Table:** Konkani language BLEU scores for various models. This table compares the performance of baseline and fine-tuned models with and without a pivot language.

Tunisian Arabic is supported by NLLB, which was used as the baseline for Tunisian Arabic translations. Similar to the approach taken with Konkani, the performance of Tunisian Arabic translations was initially evaluated using a pre-trained model. Following fine-tuning, the models showed improved performance.



| Model                              | BLEU Score |
|------------------------------------|------------|
| Baseline (NLLB)                    | 4.68       |
| Unbabel/TowerInstruct-7B-v0.2                      | 3.71       |
| Unbabel/TowerInstruct-7B-v0.2  fine-tune without pivot | 4.33      |
| Unbabel/TowerInstruct-7B-v0.2  fine-tune with pivot  | 5.13      |
| NousResearch/Hermes-2-Pro-Llama-3-8B                 |  2.05      |
| NousResearch/Hermes-2-Pro-Llama-3-8B fine-tune without pivot | 5.21 |
| NousResearch/Hermes-2-Pro-Llama-3-8B fine-tune with pivot | 7.00  |

**Table:** Tunisian Arabic language BLEU scores for various models. This table compares the performance of baseline and fine-tuned models with and without adding pivot language.

For the arabic cluster the finetuning has improved performance however we see that the BLEU scores are not very high. This can be attributed to the quality of data. 

***

## Discussion

Much of the existing research has focused on high-resource languages or those closely related to English. In this paper, we examine translation involving two relatively low-resource languages and offer a practical guide for achieving high translation performance with minimal resources. The fine-tuning process described here is computationally efficient, requiring only about 1,000 samples. All that is needed is a Colab notebook, approximately 1,000 rows of data, and minimal computational resources. This approach could make these models more accessible to communities that speak lesser-known languages. Even two language pairs are sufficient, and the third language pair can be constructed using tools like Google Translate. By bridging this gap, we aim to provide a clear methodology that can be effectively utilized in various scenarios.
However, this methodology has certain limitations. One significant challenge is the dependency on identifying a high-resource language that is linguistically similar to the target language. Additionally, building parallel corpora across multiple languages can be challenging due to limited data availability. While we did not extensively examine dataset quality in this study, we observed that training data quality is crucial; ideally, the data should resemble the types of content on which the translations will be tested.
We also acknowledge the limitations of evaluation metrics like the BLEU score. BLEU scores may not fully capture the quality of translations, especially for languages with different grammatical structures or those that are low-resource. Therefore, relying solely on BLEU scores may not provide a complete picture of the model's performance.
The results from the experiment emphasize the importance of using a high-quality, diverse dataset for both fine-tuning and constructing ICL examples, as these factors critically influence the effectiveness of the methodology. By providing a practical and accessible approach, we hope that others can easily adopt this method. All that is required is minimal computational resources and a small dataset, making it feasible for anyone interested in enhancing translation performance for low-resource languages.

***


## Appendix


### Prompt Template


Both the TOWERINSTRUCT model and NousResearch's Hermes-2-Pro-Llama-3-8B model utilize a similar prompt format. The prompt includes the source sentence in English and its translation in a pivot language. For in-context learning, the prompt contains five demonstrations. In each demonstration, the assistant field is pre-filled with the target language translation. These demonstrations are carefully selected sentences that closely resemble the sentence to be translated. In the final instance, the assistant field is left blank. This prompt structure proved to be highly effective for translation tasks of this nature. However, when using this format with the base model, the outputs often included elements like "Note," gibberish, and repetitions. After fine-tuning the model with this format, the generated translations adhered to the expected structure and consistently produced Konkani sentences.

```
<|im_start|>user
APE is a task designed to enhance the quality of the translation by performing minor adjustments
Original (English): [Original text in English]
Translation: [Translaion to  pivot langauge]
Post-edited: 
<|im_end|>
<|im_start|>assistant
[Target language translation generated by the model]
<|im_end|>

```


### Prompt tuning
Prompt tuning, is a mechanism designed to learn soft prompts that condition models to perform specific downstream tasks <d-cite key="lester2021power"></d-cite>. It was observed that when data is limited, prompt tuning can be more effective than traditional finetuning <d-cite key="zhang2024scaling"></d-cite>. However, for low-resource languages, the performance of prompt-tuned models was suboptimal. BLEU scores for translations ranged between 2 and 5, with the base model (without any finetuning) outperforming the finetuned model. 

This performance gap is likely because the model was not sufficiently exposed to these low-resource languages during training. As a result, merely applying prompt tuning was insufficient to yield significant improvements. The trainable parameters accounted for just 0.0004% of the Hermes-2-Pro-Llama-3-8B model. For the TOWERINSTRUCT model it was 0.0005%.

#### Training Parameters
The following parameters were used during the training process:

| Parameter   | Value  | 
|-------------|--------|
| `max_length`| 64     |
| `lr`        | 0.03   |
| `num_epochs`| 1      |
| `batch_size`| 8      |

#### Evaluation Parameters
To generate translations during evaluation, the following parameters were applied:

| Parameter         | Value  |
|-------------------|--------|
| `max_new_tokens`  | 750    |
| `temperature`     | 0.1    |
| `do_sample`       | True   |

These parameters are commonly employed in text generation tasks to control output length, randomness, and termination criteria. 

While prompt tuning shows potential, its effectiveness for low-resource languages remains a challenge. Refining the mechanisms to optimize prompt tuning for such scenarios is left for future work.

## Acknowledgement
I would like to thank David Dale for his critical input in understanding how best to adapt a language model to translate to low resource languages. 
