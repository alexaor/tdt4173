import numpy as np
import pandas as pd
import pathlib
import os
from tools import create_reduced_set
from tools import impute_data
from tools import encode_data
from tools import get_categorical_indexes
from tools import split_set
from tools import scale_data
from tools import feature_select
from sklearn.model_selection import KFold
"""
    CHECKLIST:
- Reduce set: CHECK
- Impute missing data: CHECK
- Encode categorical data: CHECK
- Split data set: CHECK
- Normalize data set: CHECK

    TODO:
- Sjekke hvordan den håndterer stooore dataset
- Dobbeltsjekke 
"""



"""
:param source:  filename of csv to be reduced
:param target:  filename of new csv to be made
:param rows:    row indexes to be included in reduced data set
:param columns: column indexes to be included in reduced data set

:return void

Reduce csv from source to target. Keep only rows and columns specified by 
    arguments given in function call.

"""
def reduce_data_set(source, target, rows, columns):
    print("Creating reduced set from "+source+", and storing in "+target+"....")
    reduced, column_names = create_reduced_set(rows, columns, source)

    df = pd.DataFrame(reduced, columns = column_names)
    df.to_csv(target, index = False)
    

"""
:param source:  filename of dataset to be imputed
:param target:  filename of new csv to be made

:return void

Fill inn missing values from source and write new data set to target.
"""
def impute_data_set(source, target):
    print("Imputing data from "+source+" to "+target+"....")

    dataset = pd.read_csv(source, dtype = "str")
    X = dataset.iloc[:, :].values
    imputed = impute_data(X)
    
    df = pd.DataFrame(imputed, columns = dataset.columns.values)
    df.to_csv(target, index = False)

"""
:param source:  filename of dataset to be encoded
:param target:  filename of new csv to be made

:return void

Encode categorical data from source by means of one hot encoding. Write result
    to target.
"""
def encode_data_set(source, target):
    print("encoding data from "+source+" to "+target+"....")
    
    dataset = pd.read_csv(source)
    df = encode_data(dataset)
    df.to_csv(target, index = False)


"""
:param source:  filename of dataset to be reduced
:param target:  filename of new csv to be made
:param n:       number of features to be selected
:return void

Reduce dataset from source to n features. Write result to target.
"""
def feature_selection(source, target, n):
    print("Performing feature selection from "+source+" to "+target)
    
    X = pd.read_csv(source)
    
    X_new, cols= feature_select(X, n)

    column_vals = X.columns.values[np.r_[cols, -1]]
    print(column_vals)
    
    df = pd.DataFrame(X_new, columns = column_vals)
    df.to_csv(target, index = False)
    

"""
:param source:          filename of dataset to be split
:param train_target:    filename of new csv to be made with training data
:param test_target:     filename of new csv to be made with test data

:return void

Split dataset from source into training and test data. Write result to
    train_target and test_target respectively
"""
def split_data_set(source, train_target, test_target, test_size):
    print("Splitting data set from "+source+" to "+train_target+" and "+test_target)

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, :].values
    X_train, X_test = split_set(X, size = test_size, seed = 69) 

    df = pd.DataFrame(X_train, columns = dataset.columns.values)
    df.to_csv(train_target, index = False)
    df = pd.DataFrame(X_test, columns = dataset.columns.values)
    df.to_csv(test_target, index = False)


"""
:param train_source:    filename of training_set to be normalized
:param test_source:     filename of test_set to be normalized
:param train_target:    filename of csv to be made with scaled training_set
:param test_target:     filename of csv to be made with scaled test_set  

:return void

Fit transformer with data from train_source. Standarize train- and testset and
    write to corresponding targets. 
""" 
def standarize_data_set(train_source, test_source, train_target, test_target):
    print("Performing feature scaling on dataset from "+train_source+" and "+test_source)
    print("Result from scaling is found in "+train_target+" and "+test_target)

    ## Retrieve X and y sets ##
    training_set = pd.read_csv(train_source)
    X_train = training_set.iloc[:, :-1].values
    y_train = training_set.iloc[:, -1].values
    
    test_set = pd.read_csv(test_source)
    X_test = test_set.iloc[:, :-1].values
    y_test = test_set.iloc[:, -1].values

    ## Shape and standarize ##
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    X_train_norm, X_test_norm = scale_data(X_train, X_test)

    new_training_set = np.hstack((X_train_norm, y_train))
    new_test_set = np.hstack((X_test_norm, y_test))
    
    ## Write to target ##
    df = pd.DataFrame(new_training_set, columns = training_set.columns.values)
    df.to_csv(train_target, index = False)
    df = pd.DataFrame(new_test_set, columns = test_set.columns.values)
    df.to_csv(test_target, index = False)


def create_specified_data_set(filename, n_features = -1, columns = np.r_[:119, -1], rows = np.r_[:8378], test_size = 0.2, standarize = True):
    reduce_data_set("misc_sets/speeddating.csv", "misc_sets/reduced.csv", rows, columns)
    impute_data_set("misc_sets/reduced.csv", "misc_sets/imputed.csv")
    encode_data_set("misc_sets/imputed.csv", "misc_sets/encoded.csv")
    if n_features != -1:
        feature_selection("misc_sets/encoded.csv", "misc_sets/feature_reduced_set.csv", n_features)
        if standarize:
            split_data_set("misc_sets/feature_reduced_set.csv", "misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", test_size)
            standarize_data_set("misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", "datasets/"+filename+"_train.csv", "datasets/"+filename+"_test.csv")
        else:
            split_data_set("misc_sets/feature_reduced_set.csv", "datasets/"+filename+"_train.csv", "datasets/"+filename+"_test.csv", test_size)
        
    elif standarize:
        split_data_set("misc_sets/encoded.csv", "misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", test_size)
        standarize_data_set("misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", "datasets/"+filename+"_train.csv", "datasets/"+filename+"_test.csv")
    else:
        split_data_set("misc_sets/encoded.csv", "datasets/"+filename+"_train.csv", "datasets/"+filename+"_test.csv", test_size)


def main():
    # #%% Creates a reduced csv file with rows and columns specified in create_reduced_csv() ###
    # rows = np.r_[:200]  #Max row is 8378
    # columns = np.r_[:21, -1]  #Max col is 123
    # reduce_data_set("misc_sets/speeddating.csv", "misc_sets/reduced.csv", rows, columns)

    # #%% Imputing the missing data '?' ###
    # impute_data_set("misc_sets/reduced.csv", "misc_sets/imputed.csv")

    # #%% Encode categorical data from an imputed csv ###
    # encode_data_set("misc_sets/imputed.csv", "misc_sets/encoded.csv")

    # #%% Performe feature selection to reduce data set size and reduce overfitting ###
    # feature_selection("misc_sets/encoded.csv", "misc_sets/feature_reduced_set.csv", 20)
    
    # #%% Split dataset into training and test set ###
    # split_data_set("misc_sets/feature_reduced_set.csv", "misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", 0.2)

    # #%% Performe feature scaling on training set and test set ###
    # standarize_data_set("misc_sets/raw_training_set.csv", "misc_sets/raw_test_set.csv", "misc_sets/normalized_training_set.csv", "misc_sets/normalized_test_set.csv")
    create_specified_data_set("oskartest", rows = np.r_[:200])

if __name__ == "__main__":
    main()

