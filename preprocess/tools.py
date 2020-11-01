import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
def impute_data(X):
    imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    return X_imputed    


"""
@param X_train: Array of training data
@param X_test: Array of test data
"""
def scale_data(X_train, X_test):
    sc = StandardScaler()
    X_train[:, 598:] = sc.fit_transform(X_train[:, 598:])
    X_test[:, 598:] = sc.transform(X_test[:, 598:])
    return X_train, X_test

"""
@param data_row: One dimensional array with eather categorical or non-categorical data
"""
def get_categorical_indexes(data_row):
    index_list = []
    for i in range(len(data_row)):
        try: 
            _val = float(data_row[i])
            pass
        except:
            index_list.append(i)
    return index_list
