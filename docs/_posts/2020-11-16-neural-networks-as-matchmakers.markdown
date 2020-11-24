---
layout: post
title:  "Neural networks as matchmakers"
date:   2020-11-16 14:34:25
author: "Alexander Orvik"
author_image: /assets/images/author_alex.jpg
author_bio: "An IT enthusiast"
categories: mediator feature
tags: featured
image: /assets/article_images/2020-11-16-tuning-a-neural-network/header-image.jpg
image2: /assets/article_images/2014-11-30-mediator_features/night-track-mobile.JPG
---

There is a lot to consider when dating around, and people are selective. Finding a prospective match one intends
to spend the rest of their life with motivates selective behaviour, which is understandable. 
This however, gives rise to an inherent imbalance in available data.

## Neural networks
A neural network is one of many machine learning methods, and one of the interesting attributes of such a network
is its ability to, in very basic terms, imitate pretty much anything. They are often referred to as 
_universal function approximators_ which means that they are able to imitate any arbitrary function. In our case
we want a function that takes in some attributes for two different people and then to predict whether or not
they are meant to be together.


The basic building block of a neural network is the neuron. Neurons are linked unidirectionally, meaning that 
the input is processed in a single direction. These neurons are also arranged in discrete layers which come 
in three different types, namely _input_, _hidden_ and _output_. The input layer is our entrypoint, it is where we feed
the network data which will be spit out at the other end in the output layer. In between we have the hidden layers
where a lot of the magic happens. Each connection between a pair of neurons has a _weight_ associated with it which,
depending on proper training, is what does the heavy lifting when it comes to returning a sensible value in the 
output layer.

![A simple neural network predicting clothing size](/tdt4173/assets/article_images/2020-11-16-tuning-a-neural-network/neural-net.png)

This is great and all, but how do we actually go about it? And why is this more difficult with imbalanced data?

## Balance 
Balance is important, not just in the force and in your life, but also in your datasets. Now why is that the case, 
and what does that mean for us? A neural network is not very smart out-of-the-box, like 
any other newborn entity it has to learn, which is done through training. The dataset a neural network uses to 
train on can be analogous to a teacher, so what happens when you have a terrible teacher? You don't really
learn much.

Consider a dataset from a speed dating event. A candidate will cover a lot of ground in terms of potential matches
in a short timespan, and as stated in the opening paragraph, people are picky. This means that it is not very unlikely
that only one in every tenth date is an actual match where both of the candidates would like to meet again. So if
you're not very careful when trying to teach the network the art of matchmaking a likely outcome will be a matchmaker
who does not believe in love.

Simply by always labeling two people as a mismatch with our 1-in-10 dataset a classification algorithm
would have an accuracy of 90%! Which might sound good when taken at face value, but in reality is a terrible value.
This is the heart of the problem with regards to imbalanced datasets.


## Balancing the imbalanced
Okay so if our dataset is not balanced, why don't we just collect more data where people are matches?


!["Just collect more data"](/tdt4173/assets/article_images/2020-11-16-tuning-a-neural-network/more_data.jpeg)


As great as that would be its not always possible to collect more data, so we have to make due with what we got, which
in this case is not an entirely awesome dataset. Then what toolset is at our disposal when we want to boost our
neural network's matchmaking skills?

Alright so we can't collect more data, but can we manipulate our existing data in some way? There are some strategies
one can employ in our case, over- and undersampling being two of these strategies. 
By oversampling we re-use our minority class, which in this case would be a match between two candidates, 
thereby increasing the minority class representation in the overall dataset. On the other end of the spectrum we have 
undersampling which is just removing instances of the majority class which levels out the representation of both classes.

### But what about the neural network?
Yes, what about the neural network! Let's say we don't really want to tamper with our data with different types of 
re-sampling. Reasons could vary but being scared of overfitting or a diminished dataset are perfectly valid. But 
where does that leave us? If we already have a small dataset there is already a danger of overfitting, so how can we
combat a small and imbalanced dataset with a neural network? Say hi to _dropout_ and _weight balancing_.

- **Dropout** - Ignoring a layer in the network during a training sample
- **Weight balancing** - Telling the network that certain classifications are more important than others

The process of dropping out a layer is determined by a pre-set percentage which determines the chance of a layer
simply being ignored. If ignored we do not care for the input or output of any neuron in the layer being dropped out.
But how does this actually aid us in our quest against overfitting? As a consequence this makes training
more noisy. Instead of adapting to correct the mistakes of previous layers the neurons after a _dropout layer_ need 
to adapt to a more _raw_ input.

We also have _weight balancing_. As stated this is a technique that tells the network that we care more for certain
classifications than others. This fits the goal of a matchmaker, the network should care more about the cases where it 
classifies two people as a match. Without weight balancing all classification outputs are weighted equally. In the case
of matchmaking we could for instance decide that a match should be weighted 0.8, i.e 80% and that a non-match
should be weighted 0.2.

## It's a wrap folks
There are numerous other ways to try hand handle issues with small imbalanced datasets, but covering all of them is, 
and should not be, the scope of this post. Here we have briefly covered the basics of a neural network and what
some of our options are when not given a great dataset to work with.


