import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
@param X: Independent variable array
@param Y: Dependent variable vector
@param size: Size of test set (between 0 and 1)
@param seed: Optional seed for reproducing results
"""
def split_set(X, Y, size = 0.2, seed = None):
    return train_test_split(X, Y, test_size = size, random_state = seed) 


def preprocess():
    dataset = pd.read_csv('speeddating.csv')
    
    ### implement feature selection
    X = dataset.iloc[1:16, -3:-1].values
    Y = dataset.iloc[1:16, -1].values
    
    ### Transform categorical data
    
    ### implement scaling ###
    
    return split_set(X, Y)

X_train, X_test, Y_train, Y_test = preprocess()

print(X_train)
print(Y_train)