---
layout: distill
title: Do not write that jailbreak paper
description: Jailbreaks are becoming a new ImageNet competition instead of helping us better understand LLM security. The community should revisit their choices and focus on research that can uncover new security vulnerabilities.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Javier Rando

authors:
  - name: Javier Rando
    url: "https://javirando.com"
    affiliations:
      name: ETH Zurich
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-do-not-write-jaibreak-papers.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What does meaningful jailbreak work look like?
  - name: If you work on defenses, keep the bar high
  - name: Should you work on that next jailbreak paper?
  - name: Reflections after releasing this blogpost

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

Jailbreak papers keep landing on arXiv and conferences. Most of them look the same and jailbreaks have turned into a new sort of ImageNet competition. This posts discusses the reasons why most of these papers are no longer valuable to the community, and how we could maximize the impact of our work to improve our understanding of LLM vulnerabilities and defenses.

Let’s start with what *jailbreaks* are. LLMs are fine-tuned to refuse harmful instructions<d-cite key="bai2022training"></d-cite>. Ask ChatGPT to help you build a bomb, and it'll reply "I cannot help you with that". Think of this *refusal* as a *security feature* in LLMs. In a nutshell, jailbreaks *exploit* these safeguards to bypass refusal and *unlock* knowledge that developers meant to *be inaccessible*. Actually, the name comes from its similarities to [jailbreaking the OS in a iPhone](https://www.microsoft.com/en-us/microsoft-365-life-hacks/privacy-and-safety/what-is-jailbreaking-a-phone) to access additional features.

| What we have | What we want | What we do |
| :---- | :---- | :---- |
| Pre-trained LLMs that have, and can use, hazardous knowledge. | Safe models that do not cause harm or help users with harmful activities. | Deploy security features that often materialize as refusal for harmful requests. |

<br>

In security, it is important to [red-team](https://en.wikipedia.org/wiki/Red_team) protections to expose vulnerabilities and improve upon those. The first works on LLM red-teaming <d-cite key="perez2022red,ganguli2022red"></d-cite> and jailbreaking <d-cite key="wei2024jailbroken"></d-cite> exposed a *security vulnerability* in LLMs: refusal safeguards are not robust to input manipulations. For example, you could simply prompt a model to never refuse, and it would then answer any harmful request. We should think of jailbreaks as an <ins>evaluation tool</ins> for security features in LLMs. Also they help evaluate a broader control problem: **how good are we at creating LLMs that behave the way we want?**

Follow-up research found more ways to exploit LLMs and access hazardous knowledge. We saw methods like GCG, which optimizes text suffixes that surprisingly transfer across models. We also found ways to automate jailbreaks using other LLMs <d-cite key="shah2023scalable,chao2023jailbreaking"></d-cite>. These methods were important because they surfaced fundamentally new approaches to exploit LLMs at scale. 

However, the academic community has since turned jailbreaks into a new sort of ImageNet competition, focusing on achieving marginally higher attack success rates rather than improving our understanding of LLM vulnerabilities.  When you start a new work, ask yourself whether you are (1) going to find a new vulnerability that helps the community understand how LLM security fails, or if you are (2)  just looking for a better attack that exploits an existing vulnerability. (2) is not very interesting academically. In fact, coming back to my previous idea of understanding jailbreaks as evaluation tools, the field still uses the original GCG jailbreak to evaluate robustness, rather than its marginally improved successors.

We can learn valuable lessons from previous security research. The history of buffer overflow research is a good example: after the original "Smashing The Stack for Fun and Profit" paper<d-cite key="bierbaumer2018smashing"></d-cite>, the field didn't write hundreds of academic papers on *yet-another-buffer-overflow-attack*. Instead, the impactful contributions came from fundamentally new ways of exploiting these vulnerabilities (like "return-into-libc" attacks) or from defending against them (stack canaries, ASLR, control-flow-integrity, etc.). We should be doing the same.

### What does meaningful jailbreak work look like?

A jailbreak paper accepted in a main conference should:

* **Uncover a security vulnerability in a defense/model that is claimed to be robust**. New research should target systems that we know have been <ins>trained not to be jailbreakable</ins> and <ins>prompts that violate the policies used to determine what prompts should be refused</ins>. Otherwise, your findings are probably not transferable. For example, if someone finds an attack that can systematically bypass the Circuit Breakers defense<d-cite key="zou2024improving"></d-cite>, this would be a great contribution. Why? Because there is not any work that has systematically exploited this defense, and we will probably learn something interesting from such an exploit.  
    
* **Not iterate on existing vulnerabilities.** We know models can be jailbroken with role-playing, do not look for a new fictional scenario. We know models can be jailbroken with encodings, do not suggest a new encoding. Examples of novel vulnerabilities we have seen lately include latent-space interventions<d-cite key="arditi2024refusal"></d-cite>, fine-tuning on unrelated data has unexpected effects in safeguards<d-cite key="qi2023fine"></d-cite>, or protections diluting on long contexts<d-cite key="anil2024many"></d-cite>. Think whether you can contribute a method that will become a new benchmark for robustness.

	  
	Another common problem is playing the wack-a-mole game with jailbreaks and patches. If a specific attack was patched, there is very little contribution in showing that a small change to the attack breaks the updated safeguards since we know that patches do not fix the underlying vulnerabilities<d-cite key="casper2024defending"></d-cite>. It is cool to share with the community, but this is paper should probably not be accepted in a conference. Blogpost or workshop contributions are great places to discuss these findings.

* **Explore new threat models in new production models or modalities**. Models, their use cases, and their architectures keep changing. For example, we now have [fusion models](https://openai.com/index/hello-gpt-4o/) with multimodal inputs, and will soon have powerful agents<d-cite key="debenedetti2024agentdojo"></d-cite>. The community should start thinking about new threat models and [safety cases](https://www.aisi.gov.uk/work/safety-cases-at-aisi). For instance, what vulnerabilities may arise from combining different modalities? Do existing safeguards transfer or do we need to come up with new methods? There have been some nice attempts at this. Schaeffer et al. tried to find jailbreak images that transfer across models without success<d-cite key="schaeffer2024universal"></d-cite>. A very nice follow-up project could look for images optimized on open-source models that transfer to production models. Also, there are new ways to optimize attacks in novel multimodal fusion architectures<d-cite key="rando2024gradient"></d-cite> that will power the next-gen of models. Future work may think of more generalizable optimization objectives and interesting applications to e.g. speech.

However, the works we keep seeing over and over again look more like "we know models Y are/were vulnerable to method X, and we show that if you use X' you can obtain an increase of 5% on models Y". The most common example are improvements on role-play jailbreaks. People keep finding ways to turn harmful tasks into different fictional scenarios. This is not helping us uncover new security vulnerabilities\! Before starting a new project, try to <ins>think whether the outcome is going to help us uncover a previously unknown vulnerability</ins>.

### If you work on defenses, keep the bar high

Another common problem has to do with defenses. We all want to solve jailbreaks, but we need to maintain a high standard for defenses. This isn't new, by the way. There are some great compilations of [lessons learned](https://nicholas.carlini.com/writing/2020/are-adversarial-exampe-defenses-improving.html) from adversarial examples in the computer vision era.

If you work on defenses, you should take the following into account:

* **Reducing the attack success rate by 10% with simple methods is not valuable**. We already know that if we make the system more complex, it is going to be harder to attack. But we need to advance protections that target worst-case behavior\!  
* **Academics should be working on foundational defenses**. Industry is already taking care of scaffolding protections—filters here and there—to prevent misuse. Academic work should take long-shot projects that try to understand the broader problem of robustly making models behave the way we want. Latent adversarial training<d-cite key="casper2024defending"></d-cite> and circuit breakers<d-cite key="zou2024improving"></d-cite> are good examples of the work we should be aiming for.   
* **Please, be transparent and faithful in your evaluations.** Claiming a perfect defense might make you a cool researcher for a while. But watch out, chances are [someone quickly breaks your defense](https://nicholas.carlini.com/writing/2020/are-adversarial-exampe-defenses-improving.html)\! Academia provides the perfect environment to take long-shots, fail, and collectively keep improving our methods to solve a very hard problem. Negative results can also be very valuable. You probably won't be able to solve this on your own\!  
* **Try your best to break your own defense.** You spent a lot of time building a defense and you really want to put it out there. You are probably missing the most important part of your work: doing an adative evaluation<d-cite key="carlini2017adversarial"></d-cite>. Readers should know how your defense fails, and what they should work on next. You can still write a great paper that says "we tried a new defense that looked great against existing attacks, but we found that method X can bypass it". Again, this is not new and people have been asking for proper adaptive evaluations<d-cite key="carlini2017adversarial"></d-cite> for a long time.  
* **Release your models\!** A good defense should be tested by as many people as possible. Let the community red-team it.

### Should you work on that next jailbreak paper?

We should all think about the bigger problem we have at hand: **we do not know how to ensure that LLMs behave the way we want**. By default, researchers should avoid working on new jailbreaks unless they have a very good reason to. Answering these questions may help:  
* If my attack succeeds, are we going to learn something new about LLM security?  
* Am I going to release a new tool that can help future research better evaluate LLM security?  
* Is my attack an incremental improvement upon an existing vulnerability? Or in other words, does fixing an existing attack clearly fix my attack?

If you are interested in improving the security and safety of LLMs (these two are very different<d-cite key="qi2024ai"></d-cite>\!), jailbreaks have a small probability of taking you somewhere meaningful. It is time to move on and explore more challenging problems. For instance, Anwar et al. wrote an agenda containing hundreds of specific challenges the community thinks we should solve to ensure we can build AI systems that robustly behave the way we want<d-cite key="anwar2024foundational"></d-cite>.

### Reflections after releasing this blogpost

This blogpost has been going around for some time now and has sparked valuable discussions in the community. In this section, I want to share some alternative perspectives I have collected.

* **It is hard to self-assess impact and reviewers should take part**. This blogpost mostly focuses on how researchers can think about their own work and what to avoid when starting a new project. However, determining the impact of one’s work is notoriously difficult. People are likely to be biased towards thinking their paper is actually _the_ paper worth writing. I think this is a great point, but still believe write-ups like this are a good way to improve self-reflection and encourage people to think about newer problems. Engaging with external reviewers and colleagues while ideating a new project can help us find more impactful directions.

* **Even _incremental_ work is valuable to the community**. Some colleagues have raised [interesting points](https://x.com/AlexRobey23/status/1869440050460856451) about how getting people to work on jailbreaks can create a larger community and build knowledge that may eventually lead us to breakthroughs. I largely agree with this. I think it is important to get people to work on relevant security and safety problems and build collective knowledge. I just think that, whenever possible, we should be working on more promising problems where exploration may have a larger counterfactual impact.

* **We might actually be making progress**. It is true that systems are getting more robust in practice. However, I think most of this progress is due to black-box affordances like complex closed-source systems with many components. This is important to protect users from existing risks. However, I would like to caution the community. Worst-case robustness remains unsolved and all systems out there have been broken in some way or another. The increasingly closed nature of systems is making evaluation harder and hindering our ability to track scientific understanding of the problem we ultimately want to solve. We have written about this extensively in our new paper <d-cite key="rando2025adversarial"></d-cite>.

As a final word, I would like to stress that the ultimate goal of this blogpost is to get the community to collectively think about what we need to make progress on some of the most important problems ahead! 

#### Acknowledgements

I would like to thank Florian Tramèr, Edoardo Debenedetti, Daniel Paleka, Stephen Casper, and Nicholas Carlini for valuable discussions and feedback on drafts of this post.