import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


"""
@param X: Independent variable array
@param Y: Dependent variable vector
@param size: Size of test set (between 0 and 1)
@param seed: Optional seed for reproducing results
"""
def split_set(X, Y, size = 0.2, seed = None):
    return train_test_split(X, Y, test_size = size, random_state = seed) 


"""
@param X: Array of variables to encode
@param index_list: List of column indexes that should be transformed
"""
def encode_columns(X, index_list):
    transformers=[('encoder'+str(i), OneHotEncoder(), [i]) for i in index_list]
    ct = ColumnTransformer(transformers, remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    return X




def preprocess():
    dataset = pd.read_csv('speeddating.csv')
    
    ### implement feature selection
    X = dataset.iloc[:4, :-1]
    Y = dataset.iloc[:4, -1].values
    print(X)
    
    ### Transform categorical data
    indexes = [i for i in range(X.shape[1])]
    print(indexes)
    print(encode_columns(X, indexes))
    ### Transform categorical data
    
    
    ### implement scaling ###
    
    return split_set(X, Y)

X_train, X_test, Y_train, Y_test = preprocess()
