---
layout: post
title:  "Why using ROC curves to evaluate imbalanced data is difficult"
date:   2020-11-20 09:28:25
author: "Tobias Treider Moe"
author_image: /assets/images/author_tobias.jpg
author_bio: "A machine learning enthusiast"
categories: mediator feature
tags: featured
image: /assets/article_images/2020-11-20-why-you-should-never-use-a-ROC-curve/header.jpg
image2: /assets/article_images/2014-11-30-mediator_features/night-track-mobile.JPG
---

When comparing different classification methods, the most frequent used metrics are _accuracy_ and _error rate_. 
However when working on an imbalanced dataset, this metrics can be very deceiving. Another way to make a comparison
is by plots, and one of the most used is the Receiver Operating Characteristics curves (ROC curve).

## What is a ROC curve
With the ROC curve we have a better visualization on the trade-offs between the benefits and costs of classification
regard to the data distribution. A ROC curve use the metrics _true positive rate_ (tpr) and _false positive rate_ (fpr)
to create the graph. The two metrics are mathematically defined as:

![TPR and FPR equations](/tdt4173/assets/article_images/2020-11-20-why-you-should-never-use-a-ROC-curve/eq.png)

Where TP and FP are the number of true positives and false positive respectively, P<sub>c</sub> and N<sub>c</sub> are 
the number of true positives and true negatives respectively. When plotting this two variables, we may get a plot like 
the one below:
![Example of a ROC curve](/tdt4173/assets/article_images/2020-11-20-why-you-should-never-use-a-ROC-curve/roc.png)
An classification have a ROC curve where the graph first go up to 1 on the y-axis then to 1 on the x-axis. From the 
example figure can we see that the graph are very close to an optimal ROC curve. And generally speaking is a classifier
better than another, if it has points closer to the upper left corner. The dotted line represent no learning, and if 
a classifiers have points here, it will only guess on the class labels. If a classifier has points below the dotted line
it performs worse than random guessing. If this is the case a fix may be to negate the classifiers guesses.

Areal under curve (AUC) is also something tha can be calculated with the ROC curve, which is just the areal under the 
classifiers curve. This metric can indicate how well a classifier are doing. Though it is possible for a classifier 
with lower AUC to perform much better in certain regions than a classifier with higher AUC. So be sure to look at the
curve as well when comparing classifiers.

## Problems with imbalanced data
It may seem that getting a high accuracy is very easy when working on a imbalanced dataset, but if we analyse the 
results, the reason is often that the classifier have learned to only predict the majority class. And when the majority
class has 90+% of the samples is a accuracy of 90+% not that impressive. This is way we use methods such as a ROC 
curve to evaluate the results, since it shows the result with regards to the data distribution. But there is still some
problems with evaluating with a ROC curve. In the image below are four classifieres plotted against each other in a 
ROC curve after being trained on a imbalanced dataset:
![Results on classification in a ROC curve](/tdt4173/assets/article_images/2020-11-20-why-you-should-never-use-a-ROC-curve/roc_features_all.png)
As we can see is AdaBoost and Random Forest competing in different sections, and at a section is Decision Tree doing 
little bit better than random guessing. Looking at the curve we get the feeling that some classifiers are performing 
better than other, though none of them are doing really well. So what are the problems when evaluating on a ROC curve 
when using imbalanced data:
- When there is a really high imbalance, the ROC curve can be too optimistic. (referenced from Davis 2006).
- If N<sub>c</sub> > P<sub>c</sub> and the classifier classifies a large amount of samples false positives. Will the 
still be little change in the FPR rate, which implies that it will not be shown in the ROC curve. (referenced from He 2009)
- The ROC curves are unable to derive the statistical significance of different classifiers performance, (referenced from Landgrebe 2006)
- The ROC curves have difficulty of providing insight on a classifiers performance over varying class probabilities or 
misclassification costs.

## Solving the problems with a PR curve
The problems with the ROC curves can be solved with using a Precision Recall curve (PR curve). This post is not going in
depth on what a PR curve are, but it simpy plot the _precision_ and the _recall_ against each other. Where precision is
how many labels are predicted correctly of all positive samples and recall are how many of the positive class were labeled
correctly. 

The reasons PR curves are a better option when working on imbalanced data is that:

- Can capture the performance of the classifiers when the number of false positives changes.
- Can capture all the benefits from the ROC space, since its curves has the same properties like the convex hull in ROC
space.(referenced from Davis 2006)

## Conclusion
In this post we have discussed what a ROC curve are and what problems it face when used on imbalanced data. The usage it
has when comparing different classifiers and presented another method that can substitute the need for a ROC curve, the
PR curve.

## Sources and further reading
- J. Davis and M. Goadrich. _The Relationship Between Precision-Recall and ROC Curves_. 2006.
- T.C.W Landgrebe and P. Paclik and R.P.W Duin. _Precision-recall operating characteristic (P-ROC) curves in imprecise environments_. 2006
- H. He and E. Garcia. _Learning  from  imbalanced  data_. 2009.