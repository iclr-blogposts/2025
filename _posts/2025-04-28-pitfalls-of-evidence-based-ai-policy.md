---
layout: distill
title: Pitfalls of Evidence-Based AI Policy
description: Evidence is of irreplaceable value to policymaking. However, there are systematic biases shaping the evidence that the AI community produces. Holding regulation to too high an evidentiary standard can lead to systmatic neglect of certain risks. If the goal is evidence-based AI policy, the first regulatory objective must be to actively facilitate the process of identifying, studying, and deliberating about AI risks.

date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

## Anonymize when submitting
#authors:
#  - name: Anonymous
authors:
  - name: Stephen Casper
    url: https://stephencasper.com/
    affiliations:
      name: MIT CSAIL
  - name: David Krueger
    url: https://davidscottkrueger.com/
    affiliations:
      name: Mila
  - name: Dylan Hadfield-Menell
    url: https://people.csail.mit.edu/dhm/
    affiliations:
      name: MIT CSAIL

# must be the exact same name as your blogpost
bibliography: 2025-04-28-pitfalls-of-evidence-based-ai-policy.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Abstract
  - name: How Do We Regulate Emerging Tech?
    subsections:
    - name: "Nope, I am against evidence-based policy."
    - name: A Broad, Emerging Coalition
    - name: A Vague Agenda?
  - name: The Evidence is Biased
    subsections:
    - name: Selective Disclosure
    - name: Easy vs. Hard-to-Measure Impacts
    - name: Precedented vs. Unprecedented Impacts
    - name: Ingroups vs. Outgroups
    - name: The Culture and Values of the AI Research Community
    - name: Industry Entanglement with Research
  - name: Lacking Evidence as a Reason to Act
    subsections:
    - name: Substantive vs. Process Regulation
    - name: In Defense of Compute and Cost Thresholds in AI Regulation
  - name: We Can Pass Evidence-Seeking Policies Now
    subsections:
    - name: 15 Evidence-Seeking AI Policy Objectives
    - name: Ample Room for Progress
    - name: The Duty to Due Diligence from Discoverable Documentation of Dangerous Deeds
    - name: Discussion of Counterarguments
  - name: Building a Healthier Ecosystem
  - name: Acknowledgments

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


> At this very moment, I say we sit tight and assess.
> 
> — [President Janie Orlean](https://villains.fandom.com/wiki/Janie_Orlean), [Don’t Look Up](https://www.imdb.com/title/tt11286314/)

## Abstract

Nations across the world are working to govern AI. 
However, from a technical perspective, there is uncertainty and disagreement on the best way to do this. 
Meanwhile, recent debates over AI regulation have led to calls for “evidence-based AI policy” which emphasize holding regulatory action to a high evidentiary standard. 
Evidence is of irreplaceable value to policymaking. 
However, holding regulatory action to too high an evidentiary standard can lead to systematic neglect of certain risks. 
In historical policy debates (e.g., over tobacco ca. 1965 and fossil fuels ca. 1985) “evidence-based policy” rhetoric is also a well-precedented strategy to downplay the urgency of action, delay regulation, and protect industry interests. 
Here, we argue that if the goal is evidence-based AI policy, the first regulatory objective must be to actively facilitate the process of identifying, studying, and deliberating about AI risks. 
We discuss a set of 15 regulatory goals to facilitate this and show that Brazil, Canada, China, the EU, South Korea, the UK, and the USA all have substantial opportunities to adopt further evidence-seeking policies.

## How Do We Regulate Emerging Tech?

Recently, debates over AI governance have been ongoing across the world. 
A common underlying theme is the challenge of regulating emerging technologies amidst uncertainty about the future.
Even among people who strongly agree that it is important to regulate AI, there is sometimes disagreement about when and how. 
This uncertainty has led some researchers to call for “evidence-based AI policy.”

### "Nope, I am against evidence-based policy."

See how awful that sounds? 
This highlights a troublesome aspect of how things are sometimes framed. 
Of course, evidence is indispensable. 
But there is a pitfall of holding policy action to too high an evidentiary standard:

<div style="text-align: center; font-size: 1.25em; margin: 20px 10%; line-height: 1.5;">
    Postponing regulation that enables more transparency and accountability on grounds that it's "not evidence-based" is counterproductive.
</div>

As we will argue, focusing too much on getting evidence before we act can paradoxically make it harder to gather the information we need. 

### A Broad, Emerging Coalition

Recently, there have been a number of prominent calls for evidence-based AI policy. 
For example, several California congressmembers and Governor Gavin Newsom recently argued against an AI regulatory bill in California by highlighting that it was motivated by mitigating future risks that have not been empirically observed:

> There is little scientific evidence of harm of ‘mass casualties or harmful weapons created’ from advanced models.
> 
> — Zoe Lofgren et al. in an <a href="https://democrats-science.house.gov/imo/media/doc/2024-08-15%20to%20Gov%20Newsom_SB1047.pdf" target="_blank">open letter</a> to Gavin Newsom

> [Our] approach…must be based on empirical evidence and science…[we need] AI risk management practices that are rooted in science and fact.
> 
> — Gavin Newsom in <a href="https://www.gov.ca.gov/wp-content/uploads/2024/09/SB-1047-Veto-Message.pdf" target="_blank">his veto</a> of bill [SB 1047](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047)

Others in Academia have echoed similar philosophies of governing AI amidst uncertainty. For example, in their book _AI Snake Oil_ <d-cite key="narayanan2024ai"></d-cite>, Princeton researchers Arvind Narayanan and Sayash Kapoor claim:

> The whole idea of estimating the probability of AGI risk is not meaningful…We have no past data to calibrate our predictions.
> 
> — Narayanan and Kapoor (2024), _AI Snake Oil_ <d-cite key="narayanan2024ai"></d-cite>


They follow this with [an argument](https://www.aisnakeoil.com/p/ai-existential-risk-probabilities) against the precautionary principle <d-cite key="taleb2014precautionary"></d-cite>, claiming that policymakers should take a noncommittal approach in the face of uncertainty and not act on speculative estimates of future AI risks. 

Meanwhile, Jacob Helberg, a senior adviser at the Stanford University Center on Geopolitics and Technology, has argued that there just isn’t enough evidence of AI discrimination to warrant policy action.  


> This is a solution in search of a problem that really doesn't exist…There really hasn’t been massive evidence of issues in AI discrimination.
> 
> — Jacob Helberg on [priorities](https://www.wired.com/story/donald-trump-ai-safety-regulation/) for the current US presidential administration

And Martin Casado, a partner at Andreesen Horowitz, recently argued in a post that we should hold off on taking action until we know the marginal risk:

> We should only depart from the existing regulatory regime, and carve new ground, once we understand the marginal risks of AI relative to existing computer systems. Thus far, however, the discussion of marginal risks with AI is still very much based on research questions and hypotheticals.
> 
> — Casado (2024), _Base AI Policy on Evidence, Not Existential Angst_ <d-cite key="casado2024base"></d-cite>

And finally, the seventeen authors of a recent article titled, _A Path for Science- and Evidence-Based AI Policy_, argue that:

> AI policy should be informed by scientific understanding…if policymakers pursue highly committal policy, the…risks should meet a high evidentiary standard.
> 
> — Bommasani et al. (2024), _A Path for Science‑ and Evidence‑based AI Policy_ <d-cite key="path_for_ai_policy"></d-cite>

Overall, the evidence-based AI policy coalition is diverse. 
It includes a variety of policymakers and researchers who do not always agree with each other. 
We caution against developing a one-dimensional view of this coalition or jumping to conclusions from quotes out of context. 
However, **this camp is generally characterized by a desire to avoid pursuing highly committal policy absent compelling evidence.**

## A Vague Agenda?

Calls for evidence-based AI policy are not always accompanied by substantive recommendations. 
However, Bommasani et al. (2024)<d-cite key="path_for_ai_policy"></d-cite> end their article with a set of four milestones for researchers and policymakers to pursue:<d-footnote>Bommasani et al. (2024)<d-cite key="path_for_ai_policy"></d-cite> also call for the establishment of a registry, evaluation, red-teaming, incident reporting, and monitoring but do not specify any particular role for regulators to play in these. 
They also make a nonspecific call for policymakers to broadly invest in risk analysis research and to investigate transparency requirements. </d-footnote>

> **Milestone 1:** A taxonomy of risk vectors to ensure important risks are well-represented
> 
> **Milestone 2:** Research on the marginal risk of AI for each risk vector
> 
> **Milestone 3:** A taxonomy of policy interventions to ensure attractive solutions are not missed
> 
> **Milestone 4:** A blueprint that recommends candidate policy responses to different societal conditions

These milestones are extremely easy to agree with. 
Unfortunately, they are also unworkably vague. 
It is unclear what it would mean for them to be accomplished. 
In fact, for these milestones, it is not hard to argue that existing reports reasonably meet them. 
For example, the AI Risk Repository <d-cite key="slattery2024ai"></d-cite> predates Bommasani et al. <d-cite key="path_for_ai_policy"></d-cite> and offers a meta-review, taxonomy, and living database of AI risks discussed in the literature. 
If this does not offer a workable taxonomy of risks (Milestone 1), it is unclear what would.<d-footnote>For milestone 2, most relevant research is domain-specific; consider Metta et al. (2024) <d-cite key="metta2024generativeaicybersecurity"></d-cite>, Sandbrink (2023) <d-cite key="sandbrink2023artificial"></d-cite>, Musser (2023) <d-cite key="musser2023cost"></d-cite>, and Cazzaniga et al. (2024) <d-cite key="cazzaniga2024gen"></d-cite> as examples. 
Note, however, that forecasting future marginal risks will always be speculative to some degree. 
See also Bengio et al. (2024) <d-cite key="bengio2025international"></d-cite>. 
Meanwhile, milestones 3 and 4 essentially describe the first and second stages of the AI regulation process, so existing regulatory efforts already are working on these (e.g., Arda, 2024) <d-cite key="arda2024taxonomy"></d-cite>.</d-footnote>

These milestones are an encouraging call to actively improve our understanding. 
However, absent more precision, we worry that similar arguments could be misused as a form of tokenism to muddy the waters and stymie policy action.
In the rest of this post, we will argue that holding regulatory action to too high an evidentiary standard can paradoxically make it harder to gather the information that we need for good governance. 

## The Evidence is Biased

In its pure form, science is a neutral process. 
But it is never done in a vacuum. 
Beneath the cloak of objectivity, there are subjective human beings working on problems that were not randomly selected <d-cite key="kuhn1962structure"></d-cite>. 
There is a laundry list of biases subtly shaping the evidence produced by AI researchers. 
A policymaking approach that fixates on existing evidence to guide decision-making will systematically neglect certain problems. 

### Selective Disclosure

In February 2023, Microsoft [announced](https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/) Bing Chat, an AI-powered web browsing assistant. 
It was a versatile, semi-autonomous copilot to help users browse the web. 
It was usually helpful, but sometimes, it went off the rails. 
Users found that [it occasionally took on shockingly angsty, deceptive, and outright aggressive personas](https://www.lesswrong.com/posts/jtoPawEhLNXNxvgTT/bing-chat-is-blatantly-aggressively-misaligned). 
It would go so far as to sometimes [threaten](https://x.com/marvinvonhagen/status/1625520707768659968?lang=en) [users](https://x.com/sethlazar/status/1626241169754578944?s=20) chatting with it. 
Rest assured, everyone was fine. 
Bing Chat was just a babbling web app that could not directly harm anyone. 
But it offers a cautionary tale.
Right now, developers are racing to create increasingly agentic and advanced AI systems <d-cite key="chan2023harms"></d-cite> <d-cite key="casper2025ai"></d-cite> . 
If more powerful future systems go off the rails in similar ways, it could spell trouble.  

Following the Bing Chat incidents, Microsoft’s public relations strategy focused on patching the issues and moving on. 
To the dismay of many AI researchers, Microsoft never published a public report on the incident. 
If Microsoft had nothing but humanity’s best interests at heart, it could substantially help researchers by reporting on the technical and institutional choices that led to Bing Chat's behaviors. 
However, it’s just not in their public relations interests to do so. 

Historically, AI research and development has been a very open process. 
For example, code, models, and methodology behind most state-of-the-art AI systems were broadly available pre-2020. 
More recently, however, developers like Microsoft have been exercising more limited and selective transparency <d-cite key="Bommasani2024TheFM"></d-cite>. 
Due to a lack of accountability in the tech industry, some lessons remain simply out of reach. 
There is a mounting crisis of transparency in AI when it is needed the most. 

### Easy vs. Hard-to-Measure Impacts

The scientific process may be intrinsically neutral, but not all phenomena are equally easy to study. 
Most of the downstream societal impacts of AI are difficult to accurately predict in a laboratory setting. 
The resulting knowledge gap biases purely evidence-based approaches to neglect some issues simply because they are difficult to study. 

> Thoroughly assessing downstream societal impacts requires nuanced analysis, interdisciplinarity, and inclusion…there are always differences between the settings in which researchers study AI systems and the ever-changing real-world settings in which they will be deployed.
>
> — Bengio et al. (2024), International Scientific Report on the Safety of Advanced AI <d-cite key="yohsua2024international"></d-cite>

**Differences in the measurability of different phenomena can cause insidious issues to be be neglected.** 
For instance, compare explicit and implicit social biases in modern language models. 
Explicit biases from LLMs are usually easy to spot. 
For example, it is relatively easy to train a language model against expressing harmful statements about a demographic group. 
But even when we do this to language models, they still consistently express more subtle biases in the language and concept associations that they use to characterize different people <d-cite key="wan2023kelly"></d-cite><d-cite key="wan2024white"></d-cite><d-cite key="bai2024measuring"></d-cite><d-cite key="hofmann2024dialect"></d-cite>.

Meanwhile, *benchmarks* provide the main scaffolding behind research progress in AI <d-cite key="patterson2012technical"></d-cite><d-cite key="hendrycks2022bird"></d-cite>. 
For example, tests like GPQA <d-cite key="rein2023gpqa"></d-cite> actively serve to guide progress on language model capabilities. 
Many of the benchmarks used in AI research are designed with the hope that they can help us understand downstream societal impacts. 
However, the strengths of benchmarks are also their weaknesses. 
Standardized, simplified, and portable measures of system performance often make for poor proxies to study real-world impacts <d-cite key="raji2021ai"></d-cite>.
For example, in a systematic study of benchmarks designed to assess harmlessness in AI systems, Ren et al. (2024)<d-cite key="ren2024safetywashing"></d-cite> found that many existing benchmarks intended to evaluate these qualities were, in practice, more reflective of a model’s general capabilities than anything else.

### Precedented vs. Unprecedented Impacts

In the history of safety engineering, many major system failures follow a certain story <d-cite key="dekker2019foundations"></d-cite>. 
It starts off with some system – e.g., a dam, bridge, power plant, oil rig, building, etc. – that functions normally for a long time. 
At first, this is accompanied by direct evidence of benefits and no evidence of major harms which can lull engineers into a false sense of security. 
But tragedy often strikes suddenly. 
For example, before the infamous 1986 Challenger space shuttle explosion, there were 9 successful launches <d-cite key="gebhardt20111983"></d-cite>, which was a factor that led engineers to neglect safety warnings before the infamous 10th launch. 
**Things were fine, and the empirical evidence looked good until disaster struck.** 
AI is a very powerful technology, and if it ever has a Chernobyl moment, a myopic focus on empirical evidence would be the exact way to lead us there. 

### Ingroups vs. Outgroups

The AI research community does not represent humanity well. For example, AI research is dominated by White and Asian <d-cite key="aiindex2021diversity"></d-cite>, English-speaking <d-cite key="singh2024aya"></d-cite> men <d-cite key="AbdullaChahal2023"></d-cite>. 
It is also relatively culturally homogenous:

> Since AI technologies are mostly conceived and developed in just a handful of countries, they embed the cultural values and practices of these countries.
> 
> – Prabhakaran et al. (2022), Cultural Incongruities in Artificial Intelligence <d-cite key="prabhakaran2022cultural"></d-cite>

For example, AI ethics researchers have contrasted India and the West to highlight the challenges posed by cultural homogeneity in the research community. 
In the West, societal discussions around fairness can, of course, be very nuanced, but they reflect Western experiences and are often characterized by a focus on race and gender politics. 
In India, however, the axes of social disparity are different and, in many ways, more complex. 
For example, India has 22 official languages, a greater degree of religious conflict, and a historical Caste system. 
This has led researchers to argue that the AI community is systematically poised to neglect many of the challenges in India and other non-Western parts of the world <d-cite key="qadri2023ai"></d-cite><d-cite key="bhatt2022re"></d-cite><d-cite key="sambasivan2021re"></d-cite>.

### The Culture and Values of the AI Research Community

Perhaps the most important ingroup/outgroup contrast to consider is the one between the AI research community and the rest of humanity. 
It is clear that AI researchers do not demographically represent the world <d-cite key="aiindex2021diversity"></d-cite><d-cite key="singh2024aya"></d-cite><d-cite key="AbdullaChahal2023"></d-cite>. 
Meanwhile, they tend to have much more wealth and privilege than the vast majority of the rest of the world. 
And they tend to be people who benefit from advances in technology instead of being historically or presently marginalized by it. 
This prompts a serious question:

<div style="text-align: center; font-size: 1.25em; margin: 20px 10%; line-height: 1.5;">
  Is the AI research community prepared to put people over profit and performance?
</div>

In their paper, _The Values Encoded in Machine Learning Research_, Birhane et al. (2021)<d-cite key="birhane2022values"></d-cite> analyzed 100 prominent, influential machine learning papers from 2008 to 2019. 
They annotated each based on what values were reflected in the paper text. 
The results revealed a red flag:

{% include figure.html path="assets/img/2025-04-28-pitfalls-of-evidence-based-ai-policy/birhane.png" class="img-fluid" %}
<div class="caption">
    <strong>Figure 1:</strong> From Birhane et al. (2021), <em>The Values Encoded in Machine Learning Research</em> <d-cite key="birhane2022values"></d-cite>. Among the values represented in prominent AI research papers, there is an overwhelming predominance of ones pertaining to technical system performance.
</div>

They found an overwhelming predominance of values pertaining to system performance (green) over the other categories of user rights and ethical principles. 
This suggests that **the AI community may be systematically predisposed to produce evidence that will disproportionately highlight the benefits of AI compared to its harms.**

### Industry Entanglement with Research

Who is doing the AI research? 
Where is the money coming from? 
In many cases, the answer to both is the tech companies that would be directly affected by regulation. 
For instance, consider last year’s [NeurIPS conference](https://neurips.cc/Conferences/2023). 
Google DeepMind, Microsoft, and Meta all ranked in the top 20 organizations by papers accepted: 

{% include figure.html path="assets/img/2025-04-28-pitfalls-of-evidence-based-ai-policy/neurips.png" class="img-fluid" %}
<div class="caption">
     <strong>Figure 2:</strong> Paper count by organization from the <a href="https://neurips.cc/Conferences/2023" target="_blank">NeurIPS 2023 conference</a>.
</div>

Meanwhile, other labs like OpenAI and Anthropic may not publish as many papers, but they nonetheless have highly influential work (e.g., <d-cite key="bai2022constitutional"></d-cite><d-cite key="tamkin2021understanding"></d-cite>). 
These industry papers often paint conveniently rosy picture of AI. 
For example, they sometimes engage in “safety washing”<d-cite key="ren2024safetywashing"></d-cite> in which research that is not differentially useful for safety is presented as if it is. 

The reach of industry labs into the research space involves more than just papers. 
**AI academia's entanglement with industry runs deep:**

> Imagine if, in mid-December of 2019, over 10,000 health policy researchers made the yearly pilgrimage to the largest international health policy conference in the world. 
> Among the many topics discussed…was how to best deal with the negative effects of increased tobacco usage…
> Imagine if many of the speakers who graced the stage were funded by Big Tobacco. 
> Imagine if the conference itself was largely funded by Big Tobacco.
> 
> – A discussion alluding to the [NeurIPS 2019](https://neurips.cc/Conferences/2019) conference from Abdalla and Abdalla (2020), _The Grey Hoodie Project Big Tobacco, Big Tech, and the threat on academic integrity_ <d-cite key="Abdalla2020TheGH"></d-cite>

{% include figure.html path="assets/img/2025-04-28-pitfalls-of-evidence-based-ai-policy/abdalla1.png" class="img-fluid" %}
<div class="caption">
     <strong>Figure 3:</strong> From Abdalla and Abdalla (2020) <d-cite key="Abdalla2020TheGH"></d-cite>. There are striking similarities between the anti-regulatory influences of Big Tobacco on public health research and Big Tech on AI research.
</div>

{% include figure.html path="assets/img/2025-04-28-pitfalls-of-evidence-based-ai-policy/abdalla2.png" class="img-fluid" %}
<div class="caption">
     <strong>Figure 4:</strong> From Abdalla and Abdalla (2020) <d-cite key="Abdalla2020TheGH"></d-cite>. The large majority of academic CS faculty have at some point received direct funding/awards from Big Tech or have been employed by Big Tech.
</div>

When a powerful industry is facing regulation, it is in its interest to pollute the evidence base and public discussion around it in order to deny risks and delay action.
**A key way that this manifests is with assertions that we need more evidence and consensus before we act.** 

> Is it any wonder that those who benefit the most from continuing to do nothing emphasize the controversy among scientists and the need for continued research?
> 
> – Giere et al. (2006), Understanding Scientific Reasoning <d-cite key="giere2006understanding"></d-cite>

**A common ‘**Deny and Delay Playbook**’ has been used in historical policy debates to delay meaningful regulation until long after it was needed.** 
A common story has played out in debates around tobacco, acid rain, the ozone layer, and climate change <d-cite key="oreskes2010merchants"></d-cite>. 
In each case, industry interests pushed biased science to cast doubt on risks and made noise in the media about how there just wasn’t enough evidence to act yet. 
This represents a misuse of the scientific process. 
Of course, all scientific theories are tentative and subject to criticism – this is exactly why science is so useful. 
But doubtmongering can be abused against the public interest.
Calls for evidence-based policy are often used more as a rhetorical pretext for kicking the can down the road than as a constructive, prescriptive proposal for action <d-cite key="cairney2021evidence"></d-cite>.

> Any evidence can be denied by parties sufficiently determined, and you can never prove anything about the future; you just have to wait and see.
> 
> – Oreskes and Conway (2010), Merchants of Doubt <d-cite key="oreskes2010merchants"></d-cite>

To illustrate this, we invite the reader to speculate about which of these quotes came from pundits recently discussing AI regulation and which came from merchants of doubt for the tobacco and fossil fuel industries.


<div style="display: flex; justify-content: space-between; gap: 20px;">

<div style="flex: 1;">

There is no need of going off [without a thorough understanding] and then having to retract…We should take no action unless it can be supported by reasonably positive evidence.

</div>

<div style="flex: 1;">

In addition to its misplaced emphasis on hypothetical risks, we are also concerned that [redacted] could have unintended consequences [on U.S. competitiveness]...It may be the case that the risks posed by [redacted] justify this precaution. But current evidence suggests otherwise.

</div>

<div style="flex: 1;">

The scientific base for [redacted] includes some facts, lots of uncertainty, and just plain ignorance; it needs more observation…There is also major disagreement…The scientific base for [redacted] is too uncertain to justify drastic action at this time.

</div>

</div>
<div class="caption">
     Each of these quotes is from a pundit arguing against tobacco (1965), climate (1990), and AI (2024) policies. Who said what? Answers in footnote.
     <d-footnote>
         (Left) A cancer doctor <a href="https://www.industrydocuments.ucsf.edu/tobacco/docs/#id=tnxn0124" target="_blank">testifying</a> to the US Congress in 1965 on tobacco and public health. 
         (Middle) Zoe Lofgren and other representatives in a 2024 <a href="https://democrats-science.house.gov/imo/media/doc/2024-08-15%20to%20Gov%20Newsom_SB1047.pdf" target="_blank">open letter</a> to Gavin Newsom on AI regulation. 
         (Right) Fred Singer in a <a href="https://pubs.acs.org/doi/pdf/10.1021/es00078a607" target="_blank">paper</a> arguing against climate action in 1990.
     </d-footnote>
</div>


To see an example of Big Tech entangled with calls for “evidence-based AI policy,” we need to look no further than Bommasani et al. (2024): _A Path for Science‑ and Evidence‑based AI Policy_ <d-cite key="path_for_ai_policy"></d-cite> ([discussed above](#how-do-we-regulate-emerging-tech)). 
Five of its seventeen authors have undisclosed for-profit industry affiliations.<d-footnote>Li, Pineau, Varoquaux, Stoica, Liang</d-footnote>
These include the vice president of AI research at [Meta](https://www.meta.com/) and cofounders of [World Labs](https://www.worldlabs.ai/), [Together.ai](http://together.ai), [Databricks](https://www.databricks.com/), [Anyscale](https://www.anyscale.com/), and [:probabl](https://probabl.ai/), each of which might be affected by future AI regulations.<d-footnote>The original version of the article did not contain any disclaimers about omitted author affiliations. However, it was updated in October 2024 to disclaim that “Several authors have unlisted affiliations in addition to their listed university affiliation. This piece solely reflects the authors' personal views and not those of any affiliated organizations.” However, these conflicts of interest are still not disclosed. </d-footnote>
Failing to disclose clear conflicts of interest in an explicitly political article fails to meet [standards](https://www.acs.org/content/dam/acsorg/about/governance/committees/ethics/conflict-of-interest-10-2.pdf) for ethical disclosure in research. 
These standards exist for good reason because a policymaker reading the article might interpret it very differently if it were clear that some of the authors had obvious conflicts of interest. 
It is certainly a red flag that calls for more evidence before passing highly committal regulation are coming, in part, from authors with conveniently hidden industry ties. 

## Lacking Evidence as a Reason to Act

So if the evidence is systematically biased? What do we do? How do we get more, better evidence? 

### Substantive vs. Process Regulation
We argue that a need to more thoroughly understand AI risks is a reason to pass regulation – not to delay it. 
To see this, we first need to understand the distinction between “substantive” regulation and “process” regulation. 
For our purposes, we define them as such:

- **Substantive regulation** limits **what** things developers can do with their AI systems.
- **Process regulation** limits **how** developers do what they do with their AI systems. 

These two categories of regulations do not only apply to AI. 
In the food industry, for example, ingredient bans are substantive regulations while requirements for nutrition facts are process regulations. 
Process regulations usually pose significantly lower burdens and downsides than substantive ones.
The key reason why this distinction is important is that, as we will argue:

<div style="text-align: center; font-size: 1.25em; margin: 20px 10%; line-height: 1.5;">
    A limited scientific understanding can be a legitimate (but not necessarily decisive) argument to postpone substantive regulation. But the exact opposite applies to process regulation.
</div>

Depending on whether we are considering substantive or process regulation, the argument can go different ways. 
To see an example, let’s consider some recent discussions on cost and compute thresholds in AI regulations.  

### In Defense of Compute and Cost Thresholds in AI Regulation

Some AI policy proposals set cost and compute thresholds such that, if a system’s development surpasses these, it would be subject to specific requirements. 
Some researchers have rightly pointed out that there are hazards associated with this; cost and compute can be poor proxies for societal risk <d-cite key="Hooker2024OnTL"></d-cite> <d-cite key="heim2024training"></d-cite>. 

These are important and needed points about the limitations of cost and compute thresholds. 
For example, suppose that we are considering substantive regulations that prevent deploying certain models in certain ways. 
In this case, we would need careful cost-benefit analysis and the ability to adapt regulatory criteria over time. 
But until we have government agencies who are capable of performing high-quality evaluations of AI systems’ risks, cost and compute thresholds may be the only tenable proxy available. 

**In the case of process regulation, there is often a lack of substantial downside.** 
For example, consider policies that require developers to register a system with the government if its development exceeds a cost or compute threshold. 
Compared to inaction, the upside is a significantly increased ability of the government to monitor the frontier model ecosystem. 
As for the downside? 
Sometimes certain companies will accidentally be required to do more paperwork than regulators may have intended. 
Compared to the laundry list of societal-scale risks from AI <d-cite key="slattery2024ai"></d-cite>, we can safely say that this risk is practically negligible. 

## We Can Pass Evidence-Seeking Policies Now
It is important to understand the role of process regulation in helping us to get evidence, especially since governments often tend to underinvest in evidence-seeking during institutional design <d-cite key="Stephenson2011InformationAA"></d-cite>. 
In contrast to vague calls for more research, we argue that a truly evidence-based approach to AI policy is one that proactively helps to produce more information.

<div style="text-align: center; font-size: 1.25em; margin: 20px 10%; line-height: 1.5;">
    If we want “evidence-based” AI policy, our first regulatory goal must be producing evidence. We don’t need to wait before passing process-based, risk-agnostic AI regulations to get more actionable information. 
</div>

### 15 Evidence-Seeking AI Policy Objectives

Here, we outline a set of AI regulatory goals related to **<span style="color: rgb(70, 90, 255);">institutions</span>**, **<span style="color: darkorange;">documentation</span>**, **<span style="color: green;">accountability</span>**, and **<span style="color: rgb(200, 40, 40);">risk-mitigation</span>** practices designed to produce evidence.
Each is process-based and fully risk-agnostic. 
We argue that **the current lack of evidence about AI risks is not a reason to delay these, but rather, a key reason why they are useful**. 

1. **<span style="color: rgb(70, 90, 255);">AI governance institutes:</span>** National governments (or international coalitions) can create AI governance institutes to research risks, evaluate systems, and curate best risk management practices that developers are encouraged to adhere to.
2. **<span style="color: darkorange;">Model registration:</span>** Developers can be required to register <d-cite key="McKernon2024AIMR"></d-cite> frontier systems with governing bodies (regardless of whether they will be externally deployed).
3. **<span style="color: darkorange;">Model specification and basic info:</span>** Developers can be required to document intended use cases and behaviors (e.g., <d-cite key="openai_model_spec"></d-cite>) and basic information about frontier systems such as scale.
4. **<span style="color: darkorange;">Internal risk assessments:</span>** Developers can be required to conduct and report on internal risk assessments of frontier systems.
5. **<span style="color: darkorange;">Independent third-party risk assessments:</span>** Developers can be required to have an independent third-party conduct and produce a report (including access, methods, and findings) on risk assessments of frontier systems <d-cite key="Raji2022OutsiderOD"></d-cite><d-cite key="anderljung2023towards"></d-cite><d-cite key="Casper2024BlackBoxAI"></d-cite>. They can also be required to document if and what “safe harbor” policies they have to facilitate independent evaluation and red-teaming <d-cite key="longpre2024safe"></d-cite>.
6. **<span style="color: darkorange;">Plans to minimize risks to society:</span>** Developers can be required to produce a report on risks <d-cite key="slattery2024ai"></d-cite> posed by their frontier systems and risk mitigation practices that they are taking to reduce them.
7. **<span style="color: darkorange;">Post-deployment monitoring reports:</span>** Developers can be required to establish procedures for monitoring and periodically reporting on the uses and impacts of their frontier systems.
8. **<span style="color: darkorange;">Security measures:</span>** Given the challenges of securing model weights and the hazards of leaks <d-cite key="nevo2024securing"></d-cite>, frontier developers can be required to document high-level non-compromising information about their security measures (e.g., <d-cite key="anthropic2024rsp"></d-cite>).
9. **<span style="color: darkorange;">Compute usage:</span>** Given that computing power is key to frontier AI development <d-cite key="sastry2024computing"></d-cite>, frontier developers can be required to document their compute resources including details such as the usage, providers, and the location of compute clusters.
10. **<span style="color: darkorange;">Shutdown procedures:</span>** Developers can be required to document if and which protocols exist to shut down frontier systems that are under their control.
11. **<span style="color: green;">Documentation availability:</span>** All of the above documentation can be made available to the public (redacted) and AI governing authorities (unredacted).
12. **<span style="color: green;">Documentation comparison in court:</span>** To incentivize a race to the top where frontier developers pursue established best safety practices, courts can be given the power to compare the documentation for defendants with that of peer developers.
13. **<span style="color: rgb(200, 40, 40);">Labeling AI-generated content:</span>** To aid in digital forensics, content produced from AI systems can be labeled with metadata, watermarks, and notices.
14. **<span style="color: rgb(200, 40, 40);">Whistleblower protections:</span>** Regulations can explicitly prevent retaliation and offer incentives for whistleblowers who report violations of those regulations.
15. **<span style="color: rgb(200, 40, 40);">Incident reporting:</span>** Frontier developers can be required to document and report on substantial incidents in a timely manner. 

### Ample Room for Progress

As we write this in February 2025, parallel debates over AI safety governance are unfolding across the world. There are a number of notable existing and proposed policies. 

- 🇧🇷 Brazil has recently introduced drafts of Bill No. 2338 of 2023 <d-cite key="Bill2338"></d-cite> (proposed) on regulating the use of Artificial Intelligence, including algorithm design and technical standards.
- 🇨🇦 Canada recently established an [AI Safety Institute](https://ised-isde.canada.ca/site/ised/en/canadian-artificial-intelligence-safety-institute) (exists), and its proposed AI and Data Act <d-cite key="AIDAct"></d-cite> (proposed) is currently under consideration in House of Commons Committee.
- 🇨🇳 China has enacted its Provisions on the Administration of Deep Synthesis Internet Information Services <d-cite key="DeepSynthesisProvisions"></d-cite> (enacted), Provisions on the Management of Algorithmic Recommendations in Internet Information Services <d-cite key="AlgorithmicRecommendationsProvisions"></d-cite> (enacted), and Interim Measures for the Management of Generative AI Services <d-cite key="GenerativeAIInterimMeasures"></d-cite> (enacted). There are also working drafts of a potential future ‘The Model Artificial Intelligence Law’ <d-cite key="ModelAILaw"></d-cite> (proposed).
- 🇪🇺 In the European Union, the EU AI Act <d-cite key="eu_ai_act_2024"></d-cite> (enacted) was passed in March 2024, and a large undertaking to design specific codes of practice [is underway](https://digital-strategy.ec.europa.eu/en/news/kick-plenary-general-purpose-ai-code-practice-took-place-online).
- 🇰🇷 South Korea passed the [Act on the Development of Artificial Intelligence and Establishment of Trust (AI Basic Act)](https://likms.assembly.go.kr/bill/billDetail.do?billId=PRC_R2V4H1W1T2K5M1O6E4Q9T0V7Q9S0U0) in December 2024. 
- 🇬🇧 The UK’s [AI Security Institute](https://www.aisi.gov.uk/) (exists) is currently building capacity and partnerships to evaluate risks and establish best risk-management practices. Thus far, the UK’s approach to AI regulation has been non-statutory (but new draft legislation may exist within a few months).
- 🇺🇸 In the United States, Donald Trump overturned Executive Order 14110 <d-cite key="ExecutiveOrder14110"></d-cite> after assuming office in January 2025. This may or may not lead the [US AI Safety Institute](https://www.nist.gov/aisi) (exists) to be shut down. It also might permanently stall a potential policy <d-cite key="federal_register_ai_reporting_2024"></d-cite> (proposed) on model and compute reporting that the Department of Commerce proposed in response to the executive order. Meanwhile, the AI Advancement and Reliability Act <d-cite key="HR9497"></d-cite> (proposed) was drafted last congress and [will be re-introduced](https://insideaipolicy.com/ai-daily-news/rep-obernolte-plans-reintroduce-bill-codifying-ai-safety-institute) this congress. Finally, the Future of AI Innovation Act <d-cite key="S4178"></d-cite> (drafted) and the Preserving American Dominance in Artificial Intelligence Act <d-cite key="S5616"></d-cite> (drafted) were also introduced last congress. However, as of February 2025, they are currently simply drafts. 

So how are each of these countries faring?

|                                                                              | Brazil | Canada | China | EU   | Korea | UK   | USA |
|------------------------------------------------------------------------------|---|--------|-------|------|-------|------|-----|
| **1. <span style="color: rgb(70, 90, 255);">AI governance institutes</span>** | 🟨 | ✅     | 🟨 *  | ✅   | ✅     | ✅   | ✅  |
| **2. <span style="color: darkorange;">Model registration</span>**            | ❌ | ❌     | ✅    | ✅   | ✅     | ❌   | 🟨 * |
| **3. <span style="color: darkorange;">Model specification and basic info</span>** | 🟨 | 🟨 *   | 🟨    | ✅   | 🟨    | ❌   | ❌  |
| **4. <span style="color: darkorange;">Internal risk assessments</span>**     | ✅ | 🟨 *   | 🟨    | ✅   | ✅     | ❌   | ❌  |
| **5. <span style="color: darkorange;">Independent third-party risk assessments</span>** | ❌ | ❌     | 🟨    | 🟨   | 🟨    | ❌   | ❌  |
| **6. <span style="color: darkorange;">Plans to minimize risks to society</span>** | 🟨 | ❌     | 🟨    | ✅   | 🟨    | ❌   | ❌  |
| **7. <span style="color: darkorange;">Post-deployment monitoring reports</span>** | ❌ | ❌     | ❌    | ✅   | ❌     | ❌   | ❌  |
| **8. <span style="color: darkorange;">Security measures</span>**             | ❌ | ❌     | 🟨    | ✅   | ❌     | ❌   | ❌  |
| **9. <span style="color: darkorange;">Compute usage</span>**                 | ❌ | ❌     | ❌    | 🟨   | ❌     | ❌   | 🟨 * |
| **10. <span style="color: darkorange;">Shutdown procedures</span>**          | ✅ | 🟨 *   | ❌    | 🟨   | ❌     | ❌   | ❌  |
| **11. <span style="color: green;">Documentation availability</span>**        | ❌ | 🟨 *   | 🟨    | 🟨   | 🟨    | ❌   | ❌  |
| **12. <span style="color: green;">Documentation comparison in court</span>** | ❌ | ❌     | ❌    | ❌   | ❌    | ❌   | ❌  |
| **13. <span style="color: rgb(200, 40, 40);">Labeling AI-generated content</span>** | ❌ | ❌     | ✅    | 🟨   | 🟨    | ❌   | ❌  |
| **14. <span style="color: rgb(200, 40, 40);">Whistleblower protections</span>** | ❌ | ❌     | ✅    | ✅   | ❌    | ❌   | ❌  |
| **15. <span style="color: rgb(200, 40, 40);">Incident reporting</span>**     | ✅ | ❌     | ❌    | ✅   | ❌    | ❌   | ❌  |


<div class="caption">

    <strong>Table 1:</strong> ✅ Yes | 🟨 Partial | ❌ No | * = proposed but not enacted.

    There is significant room for progress across the world on passing evidence-seeking AI policy measures. See details on each row <a href="#15-evidence-seeking-ai-policy-objectives">above</a>. Note that this table represents a snapshot in time (February 2025). In the USA, we omit the two bills <d-cite key="S4178"></d-cite> <d-cite key="S5616"></d-cite> discussed above that were proposed last congress but have not been scheduled for re-introduction this congress.

</div>

### The Duty to Due Diligence from Discoverable Documentation of Dangerous Deeds

The objectives outlined above hinge on documentation.
2-10 are simply requirements for documentation, and 11-12 are accountability mechanisms to ensure that the documentation is not perfunctory. 
This is no coincidence. 
When it is connected to external scrutiny, documentation can be a powerful incentive-shaping force <d-cite key="tomei2025ai"></d-cite>. 
Under a robust regime implementing the above the public and courts could assess the quality of developers' risk management practices.
As such, this type of regulatory regime could incentivize a race to the top on risk-mitigation standards <d-cite key="Hadfield2023RegulatoryMT"></d-cite>.

We refer to this phenomenon as the **Duty to Due Diligence from Discoverable Documentation of Dangerous Deeds – or the 7D effect**. 
Regulatory regimes that induce this effect are very helpful for improving accountability and reducing risks. 
Unfortunately, absent requirements for documentation and scrutiny thereof, developers in safety-critical fields have a perverse incentive to intentionally suppress documentation of dangers. 
For example, common legal advice warns companies against documenting dangers in written media:

> These documents may not seem bad, but in the hands of an opposing attorney these cold hard facts can [be] used to swing a judge or a jury to their side. Often the drafters of these documents tend to believe that they are providing the company with some value to the business. For example, an engineer notices a potential liability in a design so he informs his supervisor through an email. However, the engineer’s lack of legal knowledge…may later implicate the company…when a lawsuit arises.
> 
> FindLaw Attorney Writers (2016), [Safe Communication: Guidelines for Creating Corporate Documents That Minimize Litigation Risks](https://corporate.findlaw.com/litigation-disputes/safe-communication-guidelines-for-creating-corporate-documents.html)

We personally enjoyed the use of “when” and not “if” in this excerpt. 

Meanwhile, there is legal precedent for companies to sometimes lose court cases when they internally communicate risks through legally discoverable media such as in Grimshaw v. Ford (1981) <d-cite key="grimshaw1981"></d-cite>. 
**So absent requirements, companies will tend to suppress the documentation of dangers to avoid accountability.** 
Meanwhile, mere voluntary transparency can be deceptive by selectively revealing information that reflects positively on the company <d-cite key="Ananny2018SeeingWK"></d-cite>. 
Thus, we argue that a regime that requires developers to produce scrutable documentation will be key to facilitating the production of more meaningful evidence.

### Considering Counterarguments

**“These 15 objectives would be too burdensome for developers.”** 
It's true that following protocols and producing documentation can impose burdens. 
However, these burdens are generally far lighter than those imposed by substantive regulations, and compliance with many of these requirements may be trivial for developers already planning to take similar actions internally. 
For instance, even the most potentially burdensome measures – such as risk assessments and staged deployments – are practices that major developers like [OpenAI](https://cdn.openai.com/openai-preparedness-framework-beta.pdf), [Anthropic](https://assets.anthropic.com/m/24a47b00f10301cd/original/Anthropic-Responsible-Scaling-Policy-2024-10-15.pdf), and [Google DeepMind](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/introducing-the-frontier-safety-framework/fsf-technical-report.pdf) have already publicly committed to implementing.

**“It’s a slippery slope toward overreach.”** 
A second concern is that these 15 regulatory objectives might generate information that could pave the way for future substantive regulations or liability for developers. 
Regulatory and liability creep are real phenomena that can harm industry interests. 
However, it's important to emphasize that any progression from these objectives to future regulations or liability will ultimately depend on human decision-makers acting on evidence. 
Evidence is essential for society to engage in meaningful deliberation and exercise informed agency – this is the entire point of evidence-based policy. 
Therefore, if process-based AI regulations eventually lead to substantive regulations, it won't be because the process regulations laid an inevitable framework. 
It would be because the information produced by those regulations persuaded policymakers to take further action.
It would be very precarious to argue that a democratic society should be protected from its own access to information about what it is consuming.

## Building a Healthier Ecosystem

Governing emerging technologies like AI is hard <d-cite key="Bengio2023ManagingEA"></d-cite>. 
We don’t know what is coming next. 
We echo the concerns of other researchers that there are critical uncertainties with the near and long-term future of AI. 
Anyone who says otherwise is probably trying to sell you something. 
So how do we go about governing AI under uncertainty? 
History teaches us some lessons about the importance of prescient action.

> [Early] studies of global warming and the ozone hole involved predicting damage before it was detected. It was the prediction that motivated people to check for damage; research was intended in part to test their prediction, and in part to stimulate action before it was too late to stop...It was too soon to tell whether or not widespread and serious…damage was occurring, but the potential effects were troubling…A scientist would be in a bit of a bind: wanting to prevent damage, but not being able to prove that damage was coming…There are always more questions to be asked.
> 
> – Oreskes and Conway (2010), Merchants of Doubt <d-cite key="oreskes2010merchants"></d-cite>

We often hear discussions about how policymakers need help from AI researchers to design technically sound policies. 
This is essential. 
But there is a two-way street. 
Policymakers can do a great deal to help researchers, governments, and society at large to better understand and react to AI risks <d-cite key="kolt2024responsible"></d-cite>. 

**Process regulations can lay the foundation for more informed debates and decision-making in the future.** 
Right now, the principal objective of AI governance work is not necessarily to get all of the right substantive regulations in place. 
It is to shape the AI ecosystem to better facilitate the ongoing process of identifying, studying, and deliberating about risks.
This requires being critical of the biases shaping the evidence we see and proactively working to seek more information.
Kicking the can down the road for a lack of 'enough' evidence could impair policymakers' ability to take needed action. 

This lesson is sometimes painfully obvious in retrospect.
In the 1960s and 70s, a scientist named S.J. Green was head of research at the British American Tobacco (BAT) company. 
He helped to orchestrate BAT’s campaign to deny urgency and delay action on public health risks from tobacco. 
However, he later split with the company, and after reflecting on the intellectual and moral irresponsibility of these efforts, he remarked:

> Scientific proof, of course, is not, should not, and never has been the proper basis for legal and political action on social issues. A demand for scientific proof is always a formula for inaction and delay and usually the first reaction of the guilty. The proper basis for such decisions is, of course, quite simply that which is reasonable in the circumstance.
> 
> – S. J. Green, Smoking, Related Disease, and Causality <d-cite key="greensmoking"></d-cite>

## Acknowledgments

We are thankful for our discussions with Akash Wasil, Ariba Khan, Aruna Sankaranarayanan, Dawn Song, Kwan Yee Ng, Landon Klein, Rishi Bommasani, Shayne Longpre, and Thomas Woodside.




