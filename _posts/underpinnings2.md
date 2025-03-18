
---
layout: distill
title: Sample Blog Post
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: Anonymous
---




**Abstract**

In Plato's celebrated *Meno* [2], after the title character Meno asks Socrates whether virtue can be taught, he is taken on a whirlwind tour of the intellectual concepts, only to be brought by Socrates to the conclusion that virtue is (1) a gift from the gods (2) a concept that neither Meno nor Socrates really understand. Having armed myself with a generous helping of hubris and Corbett-Davies et al.'s excellent *JMLR* article "The Measure and Mismeasure of Fairness" [1], I would like to take you on a similar tour. I'll depart from the usual mid-2010s story -- the difficulty with understanding fairness in ML cannot be reduced to figuring out what your principles are and then mapping those  [3]. I'll also depart from the story by Corbett-Davies et al. -- it's not all about designing policies maximizing utility, however broadly utility is defined. Instead, I'll argue for a sort of reflective equilibrium: **ethical reasoning about any particular case needs to consider counterarguments, and, in particular, any principle that's articulated should be tested against challenging counterexamples -- a little Socrates whispering in your ear.**


# Getting to the bottom of the philosophical underpinning of ML Fairness literature


> Socrates: You say that the capacity to get good things is virtue?
> 
> Meno: I do.
> 
> Socrates: And by good things you mean, for example, health and wealth?
> 
> Meno: I also mean amassing plenty of gold and silver—and winning honors and public office.
> 
> Socrates: So, by ‘good things’ you don’t mean other sorts of things than these?
> 
> Meno: No, I mean all things of this kind. 
>
>Socrates: Very well. According to Meno—hereditary guest friend of the Great King—virtue is getting your hands on the cash. Do you qualify this definition, Meno, with the words ‘justly’ and ‘piously’? Or is it all the same to you—virtue either way—if you make your fortune unjustly?
> 
> Meno: Certainly not, Socrates.
> 
> Socrates: You would call it viciousness, then?
> 
> Meno: That I would.
> 
> Socrates: It seems, then, that the getting of gold must go along with justice or moderation or piety or some other element of virtue. If it does not, it won’t be virtue, no matter what good things are obtained.
> 
> Meno: Yes. How could there be virtue if these elements were missing?
> 
> Socrates: Then failing to acquire gold and silver, whether for oneself or for another, if these other elements were missing from the situation, would be a case of virtue?
> 
> Meno: So it seems.
> 
> Socrates: It follows that getting hold of the goods will not be virtue any more so than failing to do so is. Apparently it’s the case that whatever is done with justice will be virtue, and whatever is done in the absence of these good qualities will be vice.
> 
> Plato's *Meno* translated by Belle Waring. Free pdf from the translator: <https://examinedlife.typepad.com/files/randpchapter8-1.pdf>




# Introduction

For many years now, the ML fairness literature displayed an awareness of the fact that there are multiple measures of fairness. It is understood that fairness is a complex concept that cannot be reduced to a single measure. Nevertheless, hidden statistical and philosophical assumptions underlie even many papers that attempt to grapple with the issue.

I will look at a few recent papers, and attempt a Socratic investigation of the assumptions underlying them. At the end, I will provide a set of problems that seem GPT-proof: I invite the readers to try them with their favourite LLM and seem the LLM get as confused as its training data.


# All observational fairness measures are wrong

Observational measures of fairness -- measures that you can get from a labelled dataset -- came to prominence in ML in the context of the [ProPublica COMPAS story](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) in 2016.

The investigation revealed that the false positive rate of the recidivism prediction COMPAS for African-American defendants was much higher than the false-positive rate for Caucasian defendants -- a larger percentage of African-American defendants who did not up being re-arrested were predicted to be re-arrested.

### A quick primer on observation fairness measures

Suppose we have a sensitive demographic $D$ and an outcome of interest of $X$ (in the case of COMPAS, $X$ is re-arrest), and a prediction of $X$, $Y$.  Here are the usual measures of fairness:

- **Demographic parity**: $P(Y=1|D=1) = P(Y=1|D=0)$. That is, the same percentage of people gets re-arrested in different demographics. This is susceuptible to the objection that it might be that the *correct* predictions of re-arrest are different in different demographics. That might be because of systemic bias, but it could also be because of a benign lurking variable. For example, the fact that African-American defendants in the COMPAS dataset are younger statistically explains some of the disparity in re-arrest rates.

- **False-positive parity**: $P(Y=1|D=1, X=0) = P(Y=1|D=0, X=0)$. That is, the same percentage of people who are not re-arrested are predicted to be re-arrested in different demographics. The appeal is obvious: if you won't be re-arrested, the system shouldn't predict you will, and if the system is wrong, it should be wrong in the same way for everyone.

- **False-negative parity**: $P(Y=1|D=1, X=1) = P(Y=1|D=0, X=1)$. That is, the same percentage of people who are re-arrested are predicted to be re-arrested in different demographics. The argument is similar to the false-positive parity argument.

- **Calibration**: $P(X=1|Y=1, D=1) = P(X=1|Y=1, D=0)$. That is, if the system predicts re-arrest, the probability of re-arrest is the same in different demographics. If the system predicts a probability of re-arrest, a calibrated system will have that probability be correct. The argument for calibration is straightforward: we just want the system to be as right as possible. The argument against it must involve saying that we want to bias the system away from being right about $X$, because $X$ itself is flawed, or that biasing the system make the system more just in another way.

As was quickly pointed out, if the base rates are different, in general, only one measure of fairness can possibly hold at a time.

### Causal Fairness

The idea of causal fairness is that we should care about the counterfactual: if a person had been of a different demographic, would they have been treated differently? As we'll see, this is a tricky concept philosophically.


#### An intuitive explanation: a machine that can tell if you're 80% likely to be re-arrested

Here's a shot at an intuitive explanation: suppose that, in the two groups A and B, you have people likely to be re-arrested and people not likely to be re-arrested, and the proportions in the two groups differ (something like that -- but more complicated -- has to hold if the base-rates of arrest are different and arrests are not just based on the demographic and are otherwise random; while policing of different racial groups in the US is undoubtedly different, not all of it is due to race; for one thing, African-Americans are substantially younger, leading to more arrests: old people don't have what it takes for the kind of crime that gets you arrested).

Suppose the system can perfectly discern whether an individual is in the set that's likely to be re-arrested R (defined, say, as the set of all people whose probability of re-arrest is 80%) or in the set that's not likely to be re-arrested, and that's *all* a system can do.

If a recidivism prediction system perfectly discerns whether you're in the group that's likely to be re-arrested or not, if group A has more people who are likely to be re-arrested, it will also have more people labelled as likely to be re-arrested but who happen to not be -- not everyone likely to be re-arrested ends up being re-arrested.

### Calibration

But if the best you can do is tell if someone is in the subset that's likely to be re-arrested or not, it's difficult to argue against fudging the output and *not* doing that for the sake of achieving parity in the false-positive rates.

In fact, under our hypothetical, the only way to accomplish this would be to first discern whether the person is likely or not likely to be re-arrested, and to then *arbitrarily flip the prediction* for a subset of one of the demographic groups.

Following the assignment by the perfect predictor of likely re-arrest is using classifier that satisfies *calibration*: where the output (with some error) tries as well as possible to say 1 if the probability of 1 is above, say, 80%, and 0 otherwise.


### A conflict of intuitions
Under the artificial scenario I just described, it's difficult to argue for non-calibrated classifiers.

And yet, many people (including myself) feel the pull of the argument that disparate false-positive rates indicate a problem.

Here is one explanation: we implicitly reject the whole framework. Maybe the re-arrest patterns are biased in some way. Maybe predicting re-arrests is the wrong framework altogether -- maybe we care about actually-committed violent crime, and that's substantially different, if unknowable.

### Intermezzo: all observational fairness measures are wrong, and all are useful

[I would argue for this](https://en.wikipedia.org/wiki/All_models_are_wrong). One can argue against any fairness metric (as we have done above -- false-positive parity is bad because it's not consistent with calibration, calibration is bad because it doesn't account for model error and so leads to false-positive disparity), and that's because none of them are right.

## Causal Fairness to the rescue?
One possible way to resolve this is to say that the problem is that the measures are *observational*: what we really should care about is that a person with demographic A should be the same as that same person whose demographic had been, counterfactually, something else.

As Corbett-Davies et al. point out, that is mostly unworkable. The demographics we care about here are deeply embedded in our social context. Someone whose demopgrahic had been different would have had very different life experience (that is *why* we care about disparities along demographics), making them a different person.

(Note: this is different from "fairness through unawareness", or "color-blindness" in the context of race: counterfactual fairness requires thinking about the person's characteristics if they counterfactually *had been* a different demographic and had the corresponding life experience).

If nothing else, this is a philosophical morass: what does it even mean to think of oneself as counterfactually being a different race or gender than you actually are? Political scientists Maya Sen and Omar Wasow [offered the beginning of a framework](https://www.annualreviews.org/doi/abs/10.1146/annurev-polisci-032015-010015) in a 2016 Annual Review of Political Science article [4], but applying this to ML fairness is highly nontrivial.

Because of these difficulties, attempting to apply the counterfactual fairness criterion results in requiring demographic parity (i.e., the same proportion of "yes"es for everyone). Perhaps that's fine, but if that's what you want, you should just ask for it.

Interestingly, in a recent paper, Anthis and Veitch point out that *sometimes* one could argue that [group fairness is really does correspond to counterfacutal fairness when robustness is required ](https://openreview.net/pdf?id=AmwgBjXqc3) [5].


## Still stuck

So we're stuck with the intuitively appealing notion of counterfactual fairness, which, if we try to approximate it, reduces to observational group fairness anyway, and observational measures that are all individually and collectively unappealing.

## Just maximize utility instead?

Corbett-Davies et al. offer a way out: set out whatever goals you want, and figure out a policy that gets you somewhere where everybody is better off.

Corbett-Davies's and Sharad Goel's results here and elsewhere indicate that doing this reduces what economists call "deadweight loss": you can make everyone better off, and achieve goals such as diversity (in the case of college admission) better if you just directly aim for that instead of measuring fairness.

### Corbett-Davies and Sharad Goel's claim, mathematically

Suppose that every decision comes with a utility/cost: false positives result in people staying in jail unnecessarily, false negatives result in people being released and possibly harming others, there are costs to the community either way, etc. Corbett-Davies and Goel show that the best we can do to maximize the aggregate utility -- the same total of the expected costs and benefits to everyone, however defined (as long as it's a sum of utility) -- is to have a calibrated classifier, and to possibly threshold that classifier differently by demographic.

For COMPAS, the set-up might be something like this:

Cost of true positive to the individual: $c_{tp}$
Cost of false positive to the individual:$c_{fp}$
Cost of true negative to the individual: $c_{tn}$
Cost of false negative to the individual: $c_{fn}$

Cost of true positive to the community: $C_{tp}$
Cost of false positive to the community: $C_{fp}$
Cost of true negative to the community: $C_{tn}$
Cost of false negative to the community: $C_{fn}$

The expected utility of a decision is the sum of the individual with attributes $A=a$ is

$E[U] = -[c_{tp}P(Y=1|X=1, A=a) + c_{fp}P(Y=1|X=0, A=a) + c_{tn}P(Y=0|X=0, A=a) + c_{fn}P(Y=0|X=1, A=a) + C_{tp}P(Y=1|X=1, A=a) + C_{fp}P(Y=1|X=0, A=a) + C_{tn}P(Y=0|X=0, A=a) + C_{fn}P(Y=0|X=1, A=a)]$

(Utility is the negative cost -- I think talking in terms of costs is more intuitive).

The claim is that, whatever we do, it's better if the classifier is calibrated. That is, that the classifier is of the form $P(Y=1|A=a) > t$ for some $t$, with $P$ being the best estimate for the probability.


### Same objection as before

Are you maximizing the right goal? Did you compute the utility correctly? In particular, do you have a way to know if your classifier is truly calibrated if measuring the true outcome $X$ is impossible without bias?

All those questions, to my mind, pull us again in the direction of simple group fairness.

## Useful counterexamples

One of the strengths of Corbett-Davies et al.'s paper is the [intuition pumps](https://en.wikipedia.org/wiki/Intuition_pump) they provide: when screening for disease and trying to save people's lives, do we *really* care about group fairness, or do we just want to save as many lives as possible? If a college's  admission policy doesn't satisfy an abstract criterion of fairness but makes everyone better off and increases diversity, is that bad?


## Intermezzo: could the system be working as intended?

So far, we have been assuming that someone being predicted to be re-arrested and then not getting re-arrested is bad. But is it? If the system (and the police) picks up on the fact that the person might be re-arrested, it might be that the police warned the person and effectively change the person's behavior. 

I won't claim that that's how it always (or even usually) works, but this complicates the idea that COMPAS's false positives are always bad.


## A reflective equilibrium

I don't think the intuition pumps should be discarded, and I think the fact that all measures of fairness are imperfect is an important one.

I would argue for simply keeping all of those things in mind.

Perhaps, as many authors argue, we are aiming for a (to my mind, somewhat incoherent) notion of counterfactual fairness (that also cannot be achieved), and are also trying for a pareto-optimal policy, which we cannot perfectly design.

This would argue for looking at the shadows -- the projections, one might say -- of the ideal forms of all those on the real world, and taking account of all of them.

This might be seen as an argument for something like a ["reflective equilibrium"](https://plato.stanford.edu/entries/reflective-equilibrium/), where we take all considerations into account while recognizing that none of them are perfectly coherent, and some are not consistent.

## Exercises

When I teach taught this material, I would give the following exercises as homework to students, warning them that ChatGPT is extremely confused by them.

My hope is that this blog post, once ingested by OpenAI, Anthropic, Google, and Meta, will make a positive contribution, eventually.

* In the context of recidivism prediction for COMPAS, present an argument in favour of aiming for demographic parity.
* In the context of recidivism prediction for COMPAS, present an argument in favour of aiming for false-positive parity.
* In the context of recidivism prediction for COMPAS, present an argument in favour of aiming for calibration.
* In the context of recidivism prediction for COMPAS, consider Sharad Goel’s concepts of aggregate social welfare, and aggregate social welfare as computed for each demographic separately. Make one argument in favour of using those concepts. Make two arguments against those thinking that those concepts should guide the design of systems; at least one of those arguments should be specific to the aggregate social welfare as computed for each demograhic separately.
* There are many contexts in which prediction systems are currently being used. Examples include credit scores; university admissions; insurance rate offers; early-warning systems for deterioriating patients in hospital. Pick one field (not necessarily one of the ones I listed) where you think the ethical considerations are different from the ethical considerations for COMPAS. Argue that the ethical considerations for the two cases are different, and connect your argument to observational and causal measures of fairness that we discussed in class.



# References

[1] Sam Corbett-Davies, Johann D. Gaebler, Hamed Nilforoshan, Ravi Shroff, Sharad Goel "The Measure and Mismeasure of Fairness," *JMLR* 24(312):1−117, 2023.

[2] I recommend Belle Waring's modern translation with John Holbo's excellent commentary, available online for free at https://www.reasonandpersuasion.com/ (On dead tree: [Reason and Persuasion: Three Dialogues by Plato with commentary and illustrations by John Holbo and translations by Belle Waring](https://www.amazon.com/gp/product/1522907521/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1522907521&linkCode=as2&tag=johnbellhavea-20&linkId=35E3JKRZS4MWQAOV))

[3] I am sure *you*, the reader, didn't think of it that way. But certainly tens of thousands of undergraduates were taught that way, with perhaps a disclaimer attached saying that a more holistic approach could be better.

[4] Maya Sen and Omar Wasow. "Race as a bundle of sticks: Designs that estimate effects of seemingly immutable characteristics." Annual Review of Political Science 19 (2016): 499-522.

[5] Jacy Reese Anthis and Victor Veitch. "Causal Context Connects Counterfactual Fairness to Robust Prediction and Group Fairness." Proc. NeurIPS, 2023.






