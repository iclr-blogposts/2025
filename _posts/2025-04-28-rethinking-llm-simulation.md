---
layout: distill
title: Can LLM Simulations Truly Reflect Humanity? A Deep Dive
description: Simulation powered by Large Language Models (LLMs) has become a promising method for exploring complex human social behaviors. However, the application of LLMs in simulations presents significant challenges, particularly regarding their capacity to accurately replicate the complexities of human behaviors and societal dynamics, as evidenced by recent studies highlighting discrepancies between simulated and real-world interactions. This blog rethinks LLM-based simulations by emphasizing both their limitations and the necessities for advancing LLM simulations. By critically examining these challenges, we aim to offer actionable insights and strategies for enhancing the applicability of LLM simulations in human society in the future.
date: 2025-04-28
future: true
htmlwidgets: true
# hidden: false

# # Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Qian Wang
    affiliations:
      name: National University of Singapore
  - name: Zhenheng Tang
    affiliations:
      name: Hong Kong University of Science and Technology
  - name: Bingsheng He
    affiliations:
      name: National University of Singapore
# must be the exact same name as your blogpost
bibliography: 2025-04-28-rethinking-llm-simulation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
    - name: Introduction
    - name: Limitations in Modeling Human
    - name: Absence of Human Incentives
    - name: Bias in Training Data
    - name: Why Use LLM Simulations Despite Their Many Limitations?
    - name: How Can We Align LLMs More Closely with Human Societies?
    - name: LLM-based Simulations in Cryptocurrency Trading
    - name: Concusion
  
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
With the approximate human knowledge, large language models have revolutionized the way of simulations of social and psychological phenomena <d-cite key="Park2023GenerativeAgents, gao2023s"></d-cite>. By processing and generating human-like language, LLMs offer unprecedented opportunities to model complex interactions and behaviors that were previously challenging to simulate. This capability opens doors to exploring societal trends, market dynamics, and individual psychological states through a new lens.

However, there is a notable lack of comprehensive studies **<span style="color:blue">examining whether LLM simulations can accurately reflect real-world human behaviors</span>**. Some studies have explored this dimension from various angles. First, recent studies <d-cite key="wang2023not, wang2024new, wang2025limits"></d-cite> show that the inner knowledge of LLMs show **strong cultural bias**, **decision preference** <d-cite key="huang2024far"></d-cite>, **prior psychological character**<d-cite key="pan2023llms"></d-cite>. Second, the current training datasets of LLMs **lack personal inner psychological states, thoughts and life experiences**. LLM may reflect the common cognition of all humans instead of individual persons. Third, unlike humans who make decisions and act based on motivations from the living, emotions and achievements <d-cite key="felin2024theory"></d-cite>, LLMs **lack intrinsic motivations, emotions, and consciousness**. They operate based on resultant patterns in training data, not from lived experiences. These fundamental differences motivate rethinking how we use LLMs for simulation purposes and to critically assess their ability to replicate the depth and complexity of human society.

In this post, we delve into the limitations of LLM-driven social simulations. We discuss and summarize the challenges these models face in capturing human psychological depth, intrinsic motivation, and ethical considerations. These challenges provide insights for future LLM evaluation and development. Nevertheless, we compare the traditional simulation and the LLM-based simulation, and find that the LLM-based simulation is still a significantly potential direction due to their low costs compared to humans, scalability, and the ability to simulate emergent behaviors. Furthermore, we propose future research directions to better align LLM simulations with humans.

## Limitations in Modeling Human
Some recent works employ LLMs to model human behaviors, such as Simucara, which simulates a town to observe social dynamics <d-cite key="Park2023GenerativeAgents"></d-cite>. This simulation provides intriguing insights, including the emergence of election-like activities driven by interactions within the town. The behaviors of different LLM-simulated agents are generated based on the LLMs themselves. However, the different personlities and characteristics of LLMs are defined by the researchers' prompts. The LLM responses are rooted in patterns derived from the training datasets, but these datasets often lack deep insights into human psychology or individual life. Observing this, we identify several key limitations that significantly impact the effectiveness of LLM simulations, including the lack of access to inner psychological states and the absence of human-like incentives.

- **Training datasets lack inner psychological states.** The training datasets used for LLMs often do not include nuanced representations of inner psychological states. This limitation becomes particularly evident when LLMs are tasked with simulating diverse psychological types or personalities, as they lack intrinsic motivations that drive human decision-making. Humans make decisions based on not only the rationale and logics, but also their personal psychological states. Collecting datasets that accurately represent inner psychological states is challenging in real-world settings. Consequently, LLM training data often lacks the depth needed to capture the complexities of human psychology. Can LLM simulate these states without getting enough data related to them?

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/psychological.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 1: LLMs cannot get the inner psychological states from humans.</figcaption>
</figure>
- **Training datasets lack personal past living experiences.** Additionally, training datasets also lack comprehensive life histories, which significantly impact individual decision-making. For instance, someone with a past experience of betrayal may develop tendencies that influence their future interactions <d-cite key="finkel2002dealing"></d-cite>. 
<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/past_living.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 2: The vast scope of a human's past living experiences makes them difficult to collect comprehensively.</figcaption>
</figure>

- **Not sure whether using the same LLM can simulate different persons.** Using the same LLM model, such as black-box GPT-4, to simulate multiple agents means these agents inherently share the same foundational knowledge, making it challenging to create distinct, authentically varied personalities. The absence of personal psychological states, individual thoughts, and unique life experiences means that LLMs tend to mirror a generalized human cognition rather than capturing distinct individual personalities. Consequently, a critical question arises: **Can a single LLM genuinely simulate diverse psychological profiles?** While prompts might guide an LLM to adopt varied behaviors, the model’s core knowledge remains unchanged, raising doubts about the depth of psychological diversity that can be simulated.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/same_model.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 3: Can we believe that the same LLM can truly simulate different personas?</figcaption>
</figure>

## Absence of Human Incentives
Except for the psychological states, another significant factor that profoundly influences human behaviors is the incentive structure of humans, like survival, financial security, social belonging, emotional fulfillment, and self-actualization—each varying in intensity among individuals. **<span style="color:blue">Human decisions are shaped not only by immediate circumstances but also by intrinsic motivations, goals, and desires that vary widely among individuals <d-cite key="maslow2023motivation"></d-cite>.</span>**


  These incentives are essential for replicating realistic human behavior, as they drive diverse responses to similar situations, enable goal-oriented decision-making, and influence the trade-offs people make based on personal values and life experiences <d-cite key="shen2024motivation"></d-cite>. Even with extensive datasets on human incentives, LLMs face significant challenges in meaningfully incorporating this information due to their lack of intrinsic consciousness, emotions, and personal goals. We envision difficulties of aligning LLMs with the inner incentives of humans as the following.

- **Lacking human incentive datasets**. Similar with the psychological states, collecting the human incentive datasets is difficult. First, people may not be willing to share with their true incentives and personal goals. Second, in different time, humans may have varying goals. Third, many people do not really know what they want, the motivation is hiddened in their subconscious <d-cite key="maslow2023motivation"></d-cite>. It is hard to express them as the natural language to encode into LLMs.

- **Representing incentives with the next-word prediction.** Even we have data about human inner incentives, it is hard to model the relationships between incentives and the decisions using the next-word predition training paradigm <d-cite key="kou2023risks, wang2024research"><\d-cite>. The next-word prediction paradigm is ill-suited for modeling incentive-based behavior. Human incentives involve complex, often subconscious relationships between past experiences, emotions, and anticipated future outcomes, which shape individual decision-making in subtle, dynamic ways. Simulating such intricate, motivation-driven behaviors would require a model capable of understanding and prioritizing internal goals, a capability far beyond current LLMs’ design. Thus, while LLMs offer impressive results in language tasks, their reliance on statistical prediction, rather than intrinsic motivation, creates a gap between simulated and authentic human behavior.

## Bias in Training Data
LLMs provide a unique means to simulate large-scale social processes, such as idea dissemination, network formation, or political movement dynamics. The responses of LLMs represent their knowledge learned from the training datsets. Thus, the bias in the training data of LLMs is a significant concern <d-cite key="lee2024life"></d-cite>, as it affects the fairness and inclusivity of their outputs. One major issue is the lack of representation for certain social groups and cultural practices. We categorize several prevalent biases that significantly influence LLM simulations, including representation bias, cultural bias, and confirmation bias, each of which can distort simulation outcomes, shown in the figure below.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/bias.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 4: Numerous biases in the training data.</figcaption>
</figure>

- **Cultural bias.** For example, training data is predominantly sourced from English-speaking countries, leading to a limited understanding of diverse languages, cultures, and societal norms <d-cite key="wang2023not"></d-cite>. This geographic and cultural imbalance can result in outputs that marginalize or misrepresent non-Western perspectives.

- **Occupational and socioeconomic bias.** Another critical issue is occupational and socioeconomic bias. Workers in industries such as manufacturing or agriculture, who often have limited digital footprints, are frequently excluded from datasets. As a result, the lived experiences of these groups are underrepresented, leading to LLM outputs that fail to reflect their perspectives or address their needs—despite these individuals constituting a significant portion of human society.

- **Gender bias.** Gender bias is also evident in LLM training data, with studies showing that models are more likely to generate male-associated names and roles, reinforcing stereotypes. For example, LLMs are 3-6 times more likely to choose an occupation that stereotypically aligns with a person’s gender <d-cite key="kotek2023gender"></d-cite>. Similarly, class bias emerges in outputs that favor affluent individuals or highlight experiences and values associated with wealth, as data on the Internet disproportionately reflects the views and experiences of those familiar with and active in digital spaces <d-cite key="dai2024bias"></d-cite>.

- **Skewed voice.** These biases stem from the reliance on internet-sourced data, which is inherently skewed toward the voices of digitally literate populations. As a result, LLMs reflect the biases present in the training data, amplifying inequalities and potentially excluding significant portions of human societies from being accurately represented. 

## Why Use LLM Simulations Despite Their Many Limitations?
Despite these limitations, LLMs represent a revolutionary advancement in the field of simulation, offering unique advantages that traditional methods cannot match. Traditional simulations have long been restricted by high costs <d-cite key="gaba2004future"></d-cite>, limited scalability <d-cite key="buyya2009modeling"></d-cite>, and ethical concerns <d-cite key="rokhshad2023artificial"></d-cite>. In contrast, LLM-based simulations present several distinct advantages over traditional methods, including cost efficiency, scalability, and adaptability. For instance, LLMs can generate emergent behaviors in response to diverse scenarios, allowing researchers to explore complex social interactions without the constraints of predefined rules. The following table compares traditional simulations with LLM-based simulations, highlighting key differences in cost, scalability, flexibility, and ethical considerations in detail:  

| **Aspect**                   | **Traditional Simulation**                                                             | **LLM-Based Simulation**                                                |
|-------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Cost**                     | **High**: Requires significant financial and logistical resources, including human participants and infrastructure. | **Low**: Computationally efficient with no need for live participants.      |
| **Scalability**              | **Limited**: Expensive and resource-intensive to scale up.                                | **High**: Can simulate large-scale environments with minimal additional cost. |
| **Flexibility**              | **Rigid**: Constrained by predefined rules and models.                                    | **Adaptive**: Generates emergent behaviors and adapts to diverse scenarios. |
| **Ethical Concerns**         | **High**: Ethical issues arise from involving live participants or animals in sensitive experiments. | **Low**: Avoids ethical concerns by simulating behaviors without real-world involvement. |
| **Bias and Representation**  | **Controlled**: Biases depend on the initial design of the simulation.                    | **High Risk**: Reflects and amplifies biases in training data.              |
| **Data Requirements**        | **Specific**: Requires custom data collection and modeling for each scenario.             | **Broad**: Utilizes vast, pre-trained datasets but lacks scenario-specific granularity. |
| **Interpretability**         | **High**: Clear causal relationships based on predefined rules.                           | **Moderate**: Decisions are derived from complex patterns, making causality harder to trace. |
| **Realism**                  | **Moderate**: Captures predefined behaviors but struggles with emergent phenomena.         | **Variable**: Capable of emergent phenomena but limited by training data and lack of intrinsic motivation. |
| **Use Case Complexity**      | **Limited**: Best suited for scenarios with well-defined rules and parameters.             | **High**: Suitable for complex, open-ended scenarios with adaptive behaviors. |
| **Time to Develop**          | **Long**: Requires significant time to design, test, and validate models.                 | **Short**: Pre-trained LLMs reduce development time, with additional fine-tuning as needed. |
| **Potential for Innovation** | **Moderate**: Limited by predefined parameters and models.                                | **High**: Generates unexpected insights through emergent patterns.          |


### Cost Efficiency and Scalability
Traditional simulations, especially those involving complex human behavior, require significant financial and logistical resources, often involving teams of experts, infrastructure, and, in some cases, live participants. For instance, compensation in Singapore typically ranges from 10 to 30 Singapore dollars per hour per person. Simulating a society with 1,000 individuals would therefore incur costs between 10,000 and 30,000 Singapore dollars, representing a substantial expense. LLM-based simulations, on the other hand, are computationally efficient and can run on a large scale without the need for human participants. This makes them more accessible and affordable for researchers, enabling extensive studies across diverse scenarios and repeated simulations at a fraction of the cost.

### Unexpected and Emergent Results
LLMs have the unique ability to produce "out-of-the-box" results, generating insights that might not emerge in a structured, rule-based simulation <d-cite key="vertsel2024hybrid"></d-cite>. Since LLMs operate on patterns learned from vast datasets encompassing a wide array of human experiences, they can mimic human-like behaviors and interactions in ways that are sometimes surprising, offering novel perspectives or emergent social phenomena. For example, agents in Simulacra spontaneously initiated a mayoral election activity without any supervision <d-cite key="Park2023GenerativeAgents"></d-cite>. This characteristic allows researchers to explore complex social behaviors where unexpected behaviors may arise—for studying social dynamics, market trends, or collective human responses to specific events.

### Simulating Unconventional Scenarios
LLM-based simulations can achieve scenarios that traditional methods struggle to replicate. For example, simulating human society under conditions of anarchy or alien societal structures <d-cite key="jin2024if"></d-cite> is challenging with rule-based simulations that rely on predefined behaviors. LLMs, however, can adapt flexibly to such open-ended scenarios, generating responses and interactions that evolve dynamically based on input prompts. This adaptability allows for the exploration of future societies, governance structures, or extreme social conditions, expanding the boundaries of what simulations can achieve and enabling studies on societal organization and behavior in ways previously unachievable.

### Reduced Ethical Concerns
Traditional human-centered simulations can pose ethical challenges, often requiring participants to experience stress, discomfort, or other adverse conditions for experimental purposes. For example, psychological experiments like the Stanford Prison Experiment <d-cite key="zimbardo1971stanford"></d-cite> or animal-based studies raise ethical concerns due to the distress or harm they may cause participants. LLM simulations sidestep many ethical issues associated with traditional human-centered research, allowing researchers to simulate behaviors and reactions without involving real participants. This ethical advantage enables studies in sensitive areas, such as social conflict or psychological stress, where live participant involvement might be deemed inappropriate or harmful.

### Need of LLM Multi-agent System
There is growing research interest in LLM-based multi-agent systems <d-cite key="wu2023autogen, chen2023agentverse, hong2023metagpt"></d-cite>, driven by their ability to address complex tasks. For example, MetaGPT introduces a meta-programming framework that effectively simulates the software development process <d-cite key="hong2023metagpt"></d-cite>. Additionally, recent studies leverage LLMs’ cognitive capabilities to simulate intricate scenarios, such as large-scale social media simulations involving thousands of agents <d-cite key="guo2024large"></d-cite>. As the demand for simulating increasingly complex human societies grows, it is essential to focus on enhancing LLM simulations to better align with real-world human behaviors and societal dynamics.

In conclusion, despite the notable limitations of LLMs, their strengths in cost efficiency, scalability, and adaptability position them as transformative tools for advancing simulation research across various fields, including sociology, economics, and psychology. Future research should focus on integrating LLMs with agent systems and enhancing their personalization to create more authentic simulations.

## How Can We Align LLMs More Closely with Human Societies?
After highlighting LLM's necessasity in simulating, we discuss on **<span style="color:blue">how to align LLMs more closely with human societies.</span>** Key directions include enriching training data with nuanced psychological and experiential insights, improving the design of agent-based systems, creating realistic and meaningful simulation environments, and externally injecting societal knowledge.

### Enriching Training Data with Personal Psychological States and Life Experiences
One foundational approach is to incorporate data that reflects a broader spectrum of human psychological states, personal thoughts, and lived experiences. While current LLMs are trained on **general information** from diverse sources, this data often lacks depth in representing individual cognition and emotional states. Adding more personalized content, such as reflective diaries or first-person narratives that capture inner motivations, fears, and aspirations, could help the model simulate more realistic human behaviors. Incorporating varied life experiences can also create a richer model that better captures how past events influence decision-making and personality development over time. Personalized LLMs represent a promising direction for simulating more realistic human behaviors by incorporating concrete life experiences and individual psychological profiles <d-cite key="tseng2024two"></d-cite>.

### Improving Agent System Design
If we believe agent-based LLM simulations can simulate complex human societies and finish complex tasks, a crucial area of focus is the design of the agents themselves. Research can aim to develop reward functions that encourage agents to make decisions that mirror human behavior more accurately, and can developing the mechanism how to prevent the malacious actions propagrate, balancing short-term and long-term incentives similar to real human decision-making. Additionally, enhancing agent autonomy—such as allowing agents to learn from simulated life experiences, adapt to new environments, and develop unique 'personalities'—can improve their capacity to replicate diverse behaviors. This could involve adding emotion-like functions or “memories” that allow agents to respond adaptively based on prior interactions, similar to humans. 

### Careful Simulation Environment Design
The design of the simulation environment significantly affects agent behavior and the outcomes of the simulation. By creating environments that reflect the social, economic, and psychological complexities of human societies, agents can be more likely to engage in behaviors that resonate with human decision-making processes. For example, simulations can introduce social roles, resource scarcity, and moral dilemmas that prompt agents to make trade-offs and prioritize long-term goals over short-term gains. Personalized LLMs and retrieval-augmented generation (RAG)-based simulations can be used to dynamically provide agents with relevant information about the simulated society <d-cite key="xu2024genai"></d-cite>, helping them make decisions based on a blend of factual knowledge and social context.

### External Injection of Societal Knowledge and Values
Another promising direction is to externally inject curated societal knowledge and values into LLMs. This could be done through targeted fine-tuning or post-processing steps that embed specific ethical principles, cultural norms, and societal rules within the model’s decision-making framework. Such an approach would require LLMs to access structured knowledge bases and value systems that reflect human societal complexities, allowing them to make decisions aligned with social norms or ethical standards. For example, by integrating modules on ethics, cultural diversity, and societal roles, LLMs could better understand and reflect the diverse values that drive human societies.

### Developing Robust Evaluation Metrics
To ensure that LLMs align closely with human societies, it is essential to develop robust evaluation metrics that assess **not only the accuracy but also the depth and contextual relevance of simulated human behavior**. For instance, metrics could include alignment with established psychological theories, diversity of agent responses, and the stability of social systems over time. Metrics could include factors like alignment with human moral reasoning, diversity of responses across agents, and the stability of simulated social systems over time. Robust benchmarks that measure how closely agents’ actions mirror real-world human behaviors would allow researchers to refine LLMs more effectively, continuously improving their realism and applicability in social simulations.

## LLM-based Simulations in Cryptocurrency Trading
In this section, we analyze a case study of cryptocurrency trading simulations to illustrate the potential and limitations of LLM-based simulations.

### Using LLMs to Simulate Human Buy/Sell Behaviors in a Cryptocurrency Market
CryptoTrade is an LLM-based trading agent designed to enhance cryptocurrency market trading by integrating both on-chain and off-chain data analysis. It leverages the transparency and immutability of on-chain data, along with the timeliness and influence of off-chain signals, such as news, to offer a comprehensive view of the market. CryptoTrade also incorporates a reflective mechanism that refines its daily trading decisions by assessing the outcomes of previous trades. **It simulates the buy and sell behaviors of human traders in the cryptocurrency market.** An overview of this simulation is shown in the figure below <d-cite key="li2024cryptotrade"></d-cite>.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/cryptotrade.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 5: Overview of the CryptoTrade Simulation.</figcaption>
</figure>

And the result of this simulation on the Ethereum market compared with other trading baselines is shown in the figure below <d-cite key="li2024cryptotrade"></d-cite>.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/eth_results.png' | relative_url }}" width="400">
    <figcaption style="font-size: 1em;">Figure 6: Comparison of CryptoTrade with other trading baselines.</figcaption>
</figure>

To gain deeper insights into why CryptoTrade takes specific actions, we extract reasoning process from its simulation logs. These logs reveal how GPT-3.5 and GPT-4o respond to the same news event: Ethereum Shanghai Upgrade.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-rethinking-llm-simulation/reasoning.png' | relative_url }}">
    <figcaption style="font-size: 1em;">Figure 7: Reasoning process of gpt-3.5 and gpt-4o.</figcaption>
</figure>

### Key Observations
We summarize the key observations of the CryptoTrade simulation performance and reasoning processes as follows:

1. **LLM Simulation Can't Outperform Buy and Hold**: In a bear market, CryptoTrade **lags behind the Buy and Hold strategy by approximately 2%**, highlighting a significant limitation. While LLMs are expected to outperform human traders, the results do not align with this expectation.
2. **Inherent Bias**: During trading, CryptoTrade exhibited a tendency to prioritize factual information signals over sentiment-based information. While this approach can be advantageous in a bull market, it proves less effective in a bear market. For instance, in Ethereum trading, CryptoTrade outperformed the Buy and Hold strategy by 3%, likely due to its **inherent factual bias**. However, this bias is less suited for bear markets, where profitability often requires selling assets proactively at the first signs of a downturn in the social media.
3. **Herd Behavior**: When multiple simulation agents in CryptoTrade relied on the same LLM-based models, they often made identical decisions, which could **amplify market movements rather than creating realistic market dynamics**.

### Lessons Learned

This case study provides several insights about LLM simulations:

1. **Hybrid Approaches Needed**: The most effective simulations might combine LLM agents with **some form of human oversight or intervention**, which can be injected as the format of RAG, especially for handling extreme market conditions.

2. **Bias Mitigation**: To enable LLM simulations to better replicate realistic human behaviors, it is essential to address the factual preference biases inherent in LLMs and to incorporate societal knowledge and values into their design and training.

3. **Evaluation Metrics**: Currently, the evaluation metric is solely focused on return-related mathematical metrics in trading. However, what if different individuals prefer different trading styles or strategies? How can we assess the performance of LLM simulations in such scenarios? If we aim to simulate a realistic cryptocurrency market with diverse traders, what evaluation metrics should be used?

## Conclusion
This blog highlights the limitations of LLM simulations in aligning with human behavior, encouraging deeper reflection on their ability to model the complexity of human societies. At present, LLM-based simulations offer significant potential for research, combining cost efficiency, flexibility, and the capacity to model intricate societal dynamics in innovative ways. However, addressing ethical concerns, such as bias and representation, is essential to ensure these simulations contribute **positively and equitably to our understanding of human behavior**. To better align LLM simulations with human societies, future research should focus on mitigating inherent biases, enhancing personalization, creating realistic environments, and developing reliable metrics to produce more authentic and impactful simulations.

### Acknowledgements
This project is supported by the MOE Academic Research Fund (AcRF) Tier 1 Grant in Singapore (Grant No. T1 251RES2315).


