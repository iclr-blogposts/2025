---
layout: distill
title: The Usefulness Ceiling -- Why Frontier AI Isn't Ready to Replace Humans, And What We Should Do About It
description: LLMs and diffusion models have improved so much so quickly of late that some believe that humans are at risk of wholesale replacement in the near term. We argue that a host of challenges and obstacles, which we can collectively call The Usefulness Ceiling, make such an event unlikely in the near term. We should make good use of this opportunity, shifting our research away from short-term projects and performance hacks and towards more ambitious, interdisciplinary challenges.
date: 2025-04-28
future: true
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Replacement and Displacement -- Two Distinct Paradigms
  # you can additionally add subentries like so
    subsections:
    - name: It is much harder for AI to replace humans than to displace them
  - name: The Case Against Replacement
  - name: A Usefulness Ceiling -- The Case Against Replacement
    subsections:
    - name: Technological Limitations
    - name: Legal Limitations
    - name: Cultural Limitations
    - name: Economic Limitations
  - name: Conclusion
    subsections:
    - name: Realistic Scenarios Encourage Better Scientific Inquiry
    - name: Many Of Our Future Challenges Are Likely to be Interdisciplinary
---

## Introduction

LLMs and diffusion models have enjoyed a recent burst of attention because of a sudden surge in public awareness about the capabilities of these models, which have steadily improved over the last few decades, to the point where the most sophisticated AI systems today rival or exceed elite human performance on many complex tasks.

The justifiable excitement over these new developments, however, has been accompanied by growing anxiety. Many outsiders, and some insiders, believe that the current generation of AI represents a cultural and intellectual phase transition in terms of its potential. A natural consequence of such beliefs is the fear that humans are at imminent risk of being replaced, professionally and perhaps even personally, by AI agents. Doomsday scenarios have gained widespread attention in both the academic(https://arxiv.org/abs/2306.12001) literature(https://arxiv.org/html/2312.04714v3) and the mainstream(https://time.com/7086139/ai-safety-clock-existential-risks/) media(https://medium.com/@stevecohen_29296/prof-geoffrey-hinton-will-digital-intelligence-replace-biological-intelligence-fdbde098eebb), as have utopian fantasies(https://archive.is/EfSc5) in which the deployment of UBI will cushion the effects of rapid, large-scale job loss and BS-ification(https://bigthink.com/business/why-ai-could-transform-all-jobs-into-bs/). While dramatic predictions from prominent institutions(https://www.imf.org/en/Publications/fandd/issues/2023/12/Scenario-Planning-for-an-AGI-future-Anton-korinek), and even Nobel prizewinners, are not new in our field(https://newrepublic.com/article/187203/ai-radiology-geoffrey-hinton-nobel-prediction), and the effects of novel technologies are often surprising(https://www.cnn.com/2024/09/20/business/self-service-kiosks-mcdonalds-shake-shack/index.html), the public debate has undeniably grown louder in recent years. There have been high-profile calls to pause or discontinue frontier AI research(https://futureoflife.org/open-letter/pause-giant-ai-experiments/). In other words, the future of our field now depends not only what *is* true, but on what *ordinary people believe* is true.

The goal of this blog post is to think critically about some of the more extreme predictions for AI replacement. We establish definitions and, arguing from them, make the case that AI is likely to replace humans much more slowly than the worst case projections would have us believe. Furthermore, in several important domains, there is no more reason to believe today that AI agents will fully replace humans than there was ten or twenty years ago. 

This debate is not philosophical. The current generation of frontier AI models pose real, immediate dangers, particularly in the hands of malevolent humans. It is actively being weaponized(https://www.reuters.com/technology/artificial-intelligence/chinese-researchers-develop-ai-model-military-use-back-metas-llama-2024-11-01/) by nations(https://arstechnica.com/ai/2024/11/safe-ai-champ-anthropic-teams-up-with-defense-giant-palantir-in-new-deal/), deployed in complex scams(https://www.nyc.gov/site/dca/consumers/artificial-intelligence-scam-tips.page), and utilized for espionage(https://www.politico.com/news/2024/11/06/chinese-hackers-american-cell-phones-00187873) and influence(https://thehackernews.com/2024/11/inside-irans-cyber-playbook-ai-fake.html) campaigns. Overselling AI(https://www.indiatoday.in/technology/news/story/chatgpt-creator-sam-altman-says-jobs-will-go-away-because-of-ai-it-will-not-just-be-a-supplement-for-humans-2412116-2023-07-26) capabilities(https://www.youtube.com/watch?v=UXgJyDNBPrY&pp=ygUPYWkgcmVwbGFjZW1lbnQg) runs the risk of distracting us from more urgent threats and confusing people about what AI is really capable of. The disposition of research resources is not a zero-sum game; still, it is inefficient to direct significant effort towards dealing with risks that are highly unlikely to materialize anytime soon(https://www.aisnakeoil.com/), when large-scale near-term risks already exist.

This blog post aims to lower the temperature of the discourse around one of these long term threats (AI replacement of humans), and in so doing, embolden young researchers to prioritize longer-term, cross-disciplinary projects.

## Replacement and Displacement -- Two Distinct Paradigms

Every major technological transformation has brought with it significant job *displacement*(https://www.wikiwand.com/en/articles/Technological_unemployment) in the short term, which is then recovered in the longer term. However, some very smart people believe that this time, things will be different. In the extreme case, it has been argued that all human labor will be *replaced* by AI, perhaps fairly soon.

Let's consider the proposition critically. For starters, what do we mean by *displacement* and *replacement*? In the paradigm of displacement, changing cultural norms and technological breakthroughs trigger a shift in the behavior of complex economies. Old jobs vanish, and eventually, new ones take their place. This cycle happens on a fairly regular basis(https://ied.eu/project-updates/the-4-industrial-revolutions/) in developed societies, and while the short-term effects of these shifts are often challenging to deal with, the longer-term benefits are considerable. Job displacement, then, is a recurring phenomenon to be predicted and managed(https://www.mckinsey.com/featured-insights/future-of-work/jobs-lost-jobs-gained-what-the-future-of-work-will-mean-for-jobs-skills-and-wages), not an evil to be feared.

There is no obvious historical precedent for the idea of replacing human workers wholesale, although there are plenty of fictional(https://www.wikiwand.com/en/articles/The_Matrix) scenarios(https://en.wikipedia.org/wiki/WALL-E) which imagine what it might be like. However, we can perhaps model replacement on a smaller scale by imagining some employee. Let's call him Joe. Joe has a job, a job title, and some set of tasks he performs in order to earn his salary. If Joe was to be fired, and Jane hired to replace Joe, she (or someone else) would inherit responsiblity for all of the tasks Joe had previously done. If Joe's role was outmoded and his work no longer necessary, then we would use the model of displacement instead. One reasonable way to imagine what AI replacement might look like, then, is that instead of replacing Joe with Jane, Joe would be replaced by JaneAI, a model from some frontier AI company. It's easy to extrapolate that the effects of large-scale human replacement with AI, particularly in a capitalist society, could be catastrophic.

As a case study, let's consider three professions in particular which many pundits consider to be at extreme risk of replacement by GenAI; software developers, truck drivers, and VFX artists.

### It is much harder for AI to replace humans than to displace them

Human professions are displaced by technology on a regular basis, and there's no reason to imagine AI assistants will be any different. This is because humans, collectively, can quickly figure out how to use AI to solve the the problems it's good at, and how to correct its outputs or stop using it when it is no longer useful or safe. 

In other words, in order to be a useful assistant or tool, AI only needs to have good best-case accuracy for 1 of N tasks the human does. In order to be a cost-effective tool, it needs to be good enough and fast enough that it makes economic sense to use AI instead. Furthermore, it's fine if the tasks AI is useful for change over time, as long as, for some subset of tasks, it remains useful.

But, to replace a human entirely, the AI must have good worst-case accuracy over N of N tasks the human does, and it must be able to adopt new tasks and discard old ones whenever it proves necessary. 
 
Consider truck drivers. One task truck drivers do on a regular basis is drive trucks on highways, under normal driving conditions. This task can most likely be accomplished by advanced AI, since it essentially amounts to a more sophisticated version of cruise control. But truck drivers also sometimes have to do other things; park in ambiguously legal zone for brief periods of time in order to make timely deliveries, use their intuition to determine what standing zone is least likely to get them ticketed, get out of their truck to help unload packages, strategically adopt aggressive driving strategies to prevent other drivers from endangering their lives and timetables, refuel the truck when it runs out of gas, fix flat tires. Current frontier AI systems are not remotely close to solving even the easiest of these problems with the extremely high accuracy humans are capable of; furthermore, these problems are challenging to benchmark or simulate realistically, and challenging to acquire data for.

Before we can argue that AI will replace any particular job, at the very least, we must improve our understanding of what that job actually entails.

## A Usefulness Ceiling -- The Case Against Replacement

This section lists some reasons frontier AI might be expected to replace humans slowly or not at all in many important areas. Collectively, we can view these as a kind of upper bound or ceiling on how useful we can expect AI to be. This list, although not comprehensive, shows that many major hurdles still need to be overcome before the ceiling is broken.

### Technological Limitations
   
**We're going to run out of good data.** The recent improvements in commercial AI have primarily been driven by scale, not technical innovation. Scaling(https://arxiv.org/abs/2001.08361) laws(https://arxiv.org/abs/2203.15556) dictate that as the size of the model increases, so must the scope and quality of pretraining data. Businesses and individuals are much more aware of the commercial value of their data than they were even a few years ago. Many(https://www.engadget.com/twitter-shut-off-its-free-api-and-its-breaking-a-lot-of-apps-222011637.html) free(https://en.wikipedia.org/wiki/2023_Reddit_API_controversy) APIs(https://techcrunch.com/2024/02/05/meta-cuts-off-third-party-access-to-facebook-groups-leaving-developers-and-customers-in-disarray/) have been shut down. Knowledge is becoming siloed and data balkanized. Humans are adopting data poisoning strategies(https://www.reddit.com/r/FinalFantasy/comments/1d2ecsy/any_idea_when_ff7_rebirth_will_come_to_pc/) to push back on model owners. These changes will slow the development of future frontier models. Another major challenge future frontier models will face which older models did not is that there will be large amounts of their own content all over the internet, and this content will, in most cases, be impossible to reliably distinguish from human-generated content. When AI overtrains on synthetic data, it can lead to collapse(https://www.nature.com/articles/s41586-024-07566-y). Data annotation grows more expensive as the tasks become more challenging and harder to label; OpenAI's o1 annotators were paid extremely high hourly rates(https://openai.com/index/openai-o1-system-card/). Yet, if we are to replace humans with current generation frontier models, we will probably need lots of data to cover every single edge case for every single task.

If frontier AI is to replace all professional software developers, then it must master not only well-documented public programming languages, but poorly documented, misdocumented and undocumentted internal corporate code bases, proprietary programming languages, outmoded languages, and languages not yet imagined, and do all of these tasks at or above human expert levels of competence. It must document its code not only in English, but in many foreign languages. It must be able to forget old standards and memorize new ones whenever needed. Scaling will not solve many of these challenges because the data required will simply not exist.

**Scaling may stop working.** AI may not replace humans for a very simple reason; it is possible that scaling beyond a certain point may offer diminishing returns(https://techcrunch.com/2024/11/09/openai-reportedly-developing-new-strategies-to-deal-with-ai-improvement-slowdown/). The first sign that this is beginning to happen is likely to spark a rapid exodus of investors, considering the stratospheric cost(https://www.tomshardware.com/tech-industry/artificial-intelligence/meta-is-using-more-than-100-000-nvidia-h100-ai-gpus-to-train-llama-4-mark-zuckerberg-says-that-llama-4-is-being-trained-on-a-cluster-bigger-than-anything-that-ive-seen) of frontier model training.

**AI still can't do very much in the embodied world.** The vast majority of service professions, which dominate most developed economies, and most small-to-medium scale manufacturing jobs, occur in messy, ever-changing mixed-use environments. Humans are extremely well adapted to these environments; we built them for ourselves. AI is not.

While VFX artists do much of their compositing and 3d effects on computers, in most cases they must first acquire many of the visual elements of the scene the old-fashioned way, filming them in specialized studios. In order to completely replace VFX artists, AI would need to be able to physically stage and film components of scenes and integrate them into a cohesive whole, not just generate attractive video clips. This, too, is unlikely to happen in the near or even medium term.

**Frontier AI is still bad at managing complex, conflicting dependencies that change over time.** In order for AI to replace rather than displace humans, it will have to handle long-term strategizing and planning. Nearly all professions beyond the entry level require some degree of career and project management. Even very simple project management tasks entail hundreds of decision points, many of which are exponentially complex; which people should be included on this email chain, and which ones left out? What order should the truckers be allowed to unload? Is there room for negotiation if, say, one of them is late with a delivery? Many software developer roles require interfacing with customers, who often have trouble articulating what they need. Sometimes, it is even necessary to strategically withhold information from a customer for the good of the company. Can frontier AI be trusted to replace humans, if in so doing, it will have to make these decisions?

### Legal Limitations

**Regulatory hurdles will slow the development of AI.** The AI Act in the EU, the first major piece of regulation, has already become law. It is likely that other countries will soon follow suit. Countries which do not win the race to develop frontier AI models, in particular, will be strongly incentivized to severely limit domestic use of AI. This has already lead to major companies like Meta and Google withdrawing from certain markets; this, in turn, will make AI less economically feasible.

**Legal barriers will limit AI's ability to take human jobs (unions, accreditation).** Many human careers are organized in guild-like structures, either formal or informal. Such groups lend negotiating power to workers and protect them from abrupt termination. Employees also enjoy various legal protections in many countries which will prevent them from being replaced by AI, even when it would be more efficient to do so. Should AI actually succeed in replacing certain roles wholesale, it will strongly incentivize human workers without such protections to acquire them. Film unions have already come out against the use of generative AI; while this will not protect non-union VFX artists, it will encourage them to join a union.

### Cultural Limitations

**There will always be an AI double standard.** Humans are very good at work, but far from perfect. Many mistakes are made by human workers every day, some of them extremely destructive, or even deadly. However, as a society, we are generally tolerant of some (good faith) mistakes. After all, we are all 'only human'. But no one is tolerant of, or patient with, the shortcomings of a machine. Years of forced exposure to technology has made us impatient and dictatorial. We demand near-perfection from our mechanical servants. 

Self-driving car companies and home robotics companies have been learning this the hard way; some have gone out of business, in large part, due to a few highly publicized incidents. How many AI pilots will be allowed to crash planes? How many AI financial advisors will be allowed to misallocate their clients' life savings? How many AI sous chefs will be allowed to burn the sauce? How many, before humans start to look like the safer bet again?

**People don't know what they don't know (the prompting problem).** In a future scenario where AI has replaced human workers but not their supervisors, supervisors will need to tell their AI employees what to do. But people who do not work in a particular field usually have a limited understanding of how that field works; this can lead to instructions that underspecify or misspecify the actual requirements of the job.

It is a truism in the world of software development that customers(https://www.computerworld.com/article/1326978/when-users-don-t-know-what-they-want.html) and(https://www.goodreads.com/quotes/988332-some-people-say-give-the-customers-what-they-want-but) clients(https://softwareengineering.stackexchange.com/questions/18029/working-with-clients-who-dont-know-what-they-want) don't know what they want. While human experts and consultants are accustomed to politely pushing back on bad ideas from clients, AI models struggle to balance(https://techxplore.com/news/2023-12-chatgpt-wont-defend-weakness-large.html) refusal and instruction following. Appropriate boundaries are often learned in-context during social encounters, a task that humans are naturally adept at. It is unclear whether AI can close this gap.

**People prefer the company of other people.** Many have speculated that AI will replace humans in roles that have traditionally benefited from some degree of empathy, such as customer service(https://www.washingtonpost.com/technology/2023/10/03/ai-customer-service-jobs/) and elder care(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474924/). These predictions echo decades-old claims that kiosks would replace fast food workers(https://www.cnn.com/2024/09/20/business/self-service-kiosks-mcdonalds-shake-shack/index.html) and automation would replace checkout lines(https://apnews.com/article/amazon-fresh-just-walk-out-bb36bb24803bd56747c6f99814224265) in grocery stores. Despite the fact that the underlying technology is much more reliable, safe and cheap than frontier AI, by in large, humans have not been replaced. Customers prefer empathic, adaptable and personalizable(https://www.sciencedirect.com/science/article/pii/S0969698923000115) interactions; perhaps this is why businesses report that only 5% of humans prefer AI-exclusive customer service interactions(https://hiverhq.com/blog/ai-vs-human-in-customer-service). We have an evolutionary(https://psycnet.apa.org/record/1995-29052-001) imperative(https://www.amazon.com/Social-Why-Brains-Wired-Connect/dp/0307889106) to want to be seen and heard by other humans. The pandemic reminded us of what lengths we will go to to avoid feeling isolated and lonely. Isn't it possible, even likely, that people will be willing to pay a small premium in order to feel like they are building a relationship with another human? If AI starts to replace humans wholesale in customer-facing roles, expect startups to test this proposition quickly.

### Economic Limitations

**The cost of AI will go up along with its capabilities.** Frontier AI has grown more expensive in the past year. In some sense, this is unsurprising; the proven path to improving model capabilities is scale, but scale also leads to more expensive models(https://openai.com/index/openai-o1-system-card/) over time. Today, we are getting Frontier AI at a dramatic discount, because the industry is young, the hype is strong, and the VC money is flowing; OpenAI and Anthropic right now are where Google was back when it was inventing and giving away amazing products like Calendar, Docs, and Maps. But frontier AI burns money(https://www.nytimes.com/2024/09/27/technology/openai-chatgpt-investors-funding.html) far faster than those companies ever did, and sooner or later, those costs will be passed along to the customer. When AI gets more expensive, it becomes less competitive with humans.

## Conclusion

### Realistic Scenarios Encourage Better Scientific Inquiry

So we have built our case against replacement. But why make this argument? Because it allows us to focus our attention on the more likely (if less dramatic) scenario; namely, that this 5th industrial revolution will roughly follow the same pattern as the first four. This also presents a sobering challenge; the current generation of frontier AI will displace millions of workers. It has already rendered some companies(https://indianexpress.com/article/technology/artificial-intelligence/chegg-vs-chatgpt-how-an-edtech-giant-lost-its-business-to-ai-9665511/) superfluous, and others will likely follow. Can AI help us manage this transition more effectively, and distribute resources fairly? If we do produce agentic AI workers, what responsibility do we have for ensuring their welfare(https://arstechnica.com/ai/2024/11/anthropic-hires-its-first-ai-welfare-researcher/), and can we create safeguards to protect our models from abuse?

The meta-point to absorb here is that *realistic* scenarios naturally encourage more testable hypotheses and useful research questions than highly speculative ones. If we overindex on the idea that AI researchers(https://sakana.ai/ai-scientist/) are here today, and primed to replace us, we may succumb to short-term thinking, presenting research projects that amount to little more than LLM hacks. If, instead, we focus our attention on large unsolved problems that are highly likely to materialize, we can measure the time scales of our big bets in years instead of weeks. We can get back to focusing on what we do best; solving hard technical and theoretical problems.

### Many Of Our Future Challenges Are Likely to be Interdisciplinary

Researchers(https://iclr-blogposts.github.io/2024/blog/language-model-development-as-a-new-subfield/) have noted that many of the problems we will face in the near future are inherently interdisciplinary. Much like the internet before it, frontier AI is beginning to touch every facet of our lives, and as such, is an important area of research for many groups outside of computer science. Now is a fantastic time for us to make the most of the attention AI is receiving, forging cross-disciplinary alliances and multifaceted research projects which can make AI safer and more useful in practice.
