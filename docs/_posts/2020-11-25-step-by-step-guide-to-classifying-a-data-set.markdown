---
layout: post
title:  "Data classification in 1-2-3"
date:   2020-11-24 18:56:51
author: "Oskar Veggeland"
author_image: /assets/images/author_oskar.jpeg
author_bio: "A statistical learning enthusiast"
categories: mediator feature
tags: featured
image: /assets/article_images/2014-11-30-mediator_features/night-track.JPG
image2: /assets/article_images/2014-11-30-mediator_features/night-track-mobile.JPG
---

# Hello World!
I was told that my first blogpost should start with a joke, so here goes:
>Why don't jokes work in octal? Because 7 10 11

Now, while I got your intention, how would you like to train your very own classifition model on the data set of your choosing?
Great, lets get started!


# INSTALLS AND PREREQUISITES
This tutorial will be based on the source code from our [group github repository](https://github.com/alexaor/tdt4173).
Before we get started, lets go ahead and clone this repository to your computer.
The README should contain everything you need to know  to download the required dependencies for this tutorial. 
If you are unfamiliar with package managers and dependencies you should probably read up on this first. [Here is a useful link to get you started.](https://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/)


# Step 1 (Find and download your chosen data set)
The very first thing you have to do is to find the data set you want to train on. The only 
requirements is that it should be in a .csv format and should contain at least one binary label that you wish to predict.
[openml](https://www.openml.org/home) and [kaggle](https://www.kaggle.com/datasets) have lots of data sets you can take a swing at. 
When you have found your desired data set, move it to data/datasets/ and overwrite the existing file: 01_raw_data.csv.

To make sure that the downloaded source files are able to train properly you have to make a few adjustments in preprocess/preprocessor.py.
In the function create_data_set(), the function filter_desired_features() is called with a parameter, columns. This parameter contains indices corresponding to the features one wishes to use from the data set and needs to be changed in order to fit your chosen data.
The most important thing is that your chosen label feature is last index in the column-list.

For example, if your chosen label feature is given by the third column in your data and you wish to only keep feature 1,2 and 5 through 9 the column variable should look something like this:
```python
columns = np.r_[0, 1, 4:9, 2]
```



# Step 2 (Tuning and model selection)
By default, all models will be trained and evaluated on your data. This includes a decision tree, a random forest, an adaboost and a deep neural network.
If you wish to only use certain models, this can be done by changing the list of keys in the main function.

Most of the parameters can be tuned via gin configuration files. There are two such files in the configs/ folder.
These gin files will be read by the source code before building models, so changing the parameters here is the easiest way of tuning models.
The reason why we need two gin files is because individual models are built to compare classifcation with and without feature selection. 

> Note: If you wish to change the architecture of the neural network, this cannot be done through the gin files, but must be implemented in methods/dnn.py.
In the class function _create_model() you can change the return value by changing/inserting/deleting as many layers as you want. 

# Step 3 (Train models and analyse evaluation outputs)
Now that you have downloaded your very own data set and chosen the parameters of your choice, you are ready to train the models and see how they perform.
This is easily done by running the main file. A menu will appear to guide you further.

Not happy with the result? No problem! Just go back to step 2 and try again!

Happy analysing!