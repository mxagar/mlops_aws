# Machine Learning School with AWS

This guide started as a collection of notes after following the cohort-based course [Machine Learning School](https://svpino.gumroad.com/l/mlp) by [Santiago Valderrama](https://twitter.com/svpino) .

The forked Github repository with the code is this one:

[ml.school](https://github.com/mxagar/ml.school)

## Session 1

Zoom session: [https://discord.com/events/1079107756440690728/1169008338021916782](https://discord.com/events/1079107756440690728/1169008338021916782)

Code is already given, as well as walkthroughs!
Discord > Server Guide Archive.

All sessions are recorded!
He plans to edit them to concise clips, but that comes later in March

Sessions should be packed with signal, but he'll just provide the doors: we need to open them!

He wants to focus on the things that will be still used in 10 years.
Timeless ideas.

We need to unlearn what ML is
- books focus 80% on algorithms
- but the engineer focuses 80% on the data!

Structure of the program
	
1. Live sessions: interactive!
2. Code walkthroughs
3. Assignments:
    - There is also a class project
    - Recommendation: Team-up, 2-3 people max.
    - He's not reviewing any of these!

Feynman approach: he had a notebook where he wrote all the things he didn't understand. While doing so, he dissected every single part of the concepts he wrote down in that notebook.

You don't want to solve the wrong project; start every project asking 5 questions:

1. What is the problem we are trying to solve?
    - understand scope
    - identify every related problem we won't solve

2. Why is solving this problem important?
    - provide context, priorities
    - justify investments for solution

3. What do existing solutions look like?
    - learn from past experiences: best practices, pitfalls, opportunities to improve

4. What does the available data look like?
    - many companies who want to start doing ML, they don't even have the dataset yet!
    - identify quality issues and potential biases

5. How do we measure success?
    - We need to have a benchmark to assess where we want to go
    - We need to understand the answer to all these 5 questions perfectly!
    - Then, we know how to frame the problem to solve it.
    - We want to frame the problem so that we are successful at solving it; following Naval, we want to frame the problem so that we successfully solve it in 999/10000 universes, i.e., without luck.
    - A great strategy to find solutions to complex problems is to use *inversion*: turn the problem upside-down to think about it differently.
        - Instead of finding the needle in the haystack, make the haystack as small as possible.
        - Instead of being very intelligent, try not to be stupid.

Book Recommendations:

- Designing Machine Learning Systems
- Human in the Loop Machine Learning
- Interpretable Machine Learning

The first rule in machine learning: never start a project with ML, but just with the simplest thing that could possibly work.

Simple ideas are better than clever ideas. ML is a clever solution; a better alternative is to use simple solutions:
	
- prefer regex to LLMs
- prefer logistic regression to transformers
- etc.

Example: used luxury watches: set their price and sell them; which price?

![Example: Predict price of used luxury watches](./assets/mlschool_example_watches.jpg)

- In reality, we don't know if we are leaving money on the table; so a simple approach is to set a price and iteratively change it to see how the demand reacts.
- Another ruthless approach to simplicity is to become the software (Paul Graham): allow the person to be able to do what the model would do. That goes against scalability, but it doesn't matter; the scalability issue comes later.
- A simple system serves as a baseline!
    - They give early feedback.
    - They provide access to real production data.

Collect the data! There are no datasets in the real world; if we have a curated dataset, we should not trust it, because the chance of it not representing the production data is usually high.

Questions related to data:

1. How much data do we need?
		
    Start with enough data to build and evaluate a model.
    Only go and collect more data if after evaluating the model, we see that extra data adds more value.
    Learning curves are useful for that.
    We can also try different training sizes and see how the model evaluates. Maybe we see that there is a plateau after a dataset size after which it makes no sense to increase the data.
    Also, we need to we aware of the quality of the data we are adding: are these insignificant samples, or valuable ones?
	
2. How can we source it?
		
    Selection bias is important; it is impossible to prevent. If the data in production is different to the data used during training, the model behaves unexpectedly!
    Common causes of selection bias:
    - time: when the data was taken, the time-point changes the context
    - location: where the data is taken; obviously, we cannot collect data everywhere in the world, so we need to select
    - demographics: is the people involved in the data different? maybe the people we collect data from are younger than the real users?
    - response bias: every questionnaire gave Hillary Clinton as the winner, but Trump won; obviously, the participants did something different in the real moment

    If we collect the data, collect also the meta-data that explains an contextualizes your data! Example: day and time, place, camera used, temperature, etc. Those information is critical to test for selection bias.

3. How do we structure the data?

Models can't learn without ground truth: we need to label the data to train the model; then, the models are hopefully able to generalize.

Bad quality labels are much worse than unlabelled data!

Check the library [Cleanlab](https://github.com/cleanlab/cleanlab).

- The library can help us find labeling mistakes

It is sometimes not a good idea to outsource the labeling; human annotations are very expensive, but produce better quality results.

A way to scale labeling is to use weak supervision: use rules and support models to predict the labels.

Weak supervision is very fast and inexpensive, but not always works: if it worked always, why would we need ML?

Sometimes the labels from weak supervision labeling and inaccurate, etc.

Active learning is another approach to labeling.

- Sample a little bit of data.
- Train a model.
- Predict labels on unseen subest.
- From that predicted subset, pick a sample, the ones which aremost informative.
- Label that picked subset, train a new model.
- Predict unlabeled data, again.
- Repeat the process.
- In each iteration, we are adding new samples.
- The samples we add are the most informative ones!
- Which are the most informative ones?
- Random sampling: pick random samples; that's not really a subset of most informative samples.
- Uncertainty sampling: data-points the model is not sure about, i.e., samples that are close to decision boundaries. A cat that looks like a dog.
- Diversity sampling: identify samples that are not usual, i.e., underrepresented data-points. Edge cases, e.g., a dog which is swimming.
- We need to combine both uncertainty sampling and diversity sampling.

![Active Learning](./assets/mlschool_example_active_learning.jpg)
