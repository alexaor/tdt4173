import numpy as np
import pandas as pd
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

def reduce_data_set(source, target):
    print("Creating reduced set from "+source+", and storing in "+target+"....")

    rows = np.r_[:200]
    columns = np.r_[:20, -1]
    reduced, names = create_reduced_set(rows, columns, source)

    df = pd.DataFrame(reduced, columns = names)
    df.to_csv(target, index = False)
    
def impute_data_set(source, target):
    print("Imputing data from "+source+" to "+target+"....")

    dataset = pd.read_csv(source, dtype = "str")
    X = dataset.iloc[:, :].values
    imputed = impute_data(X)
    
    df = pd.DataFrame(imputed, columns = dataset.columns.values)
    df.to_csv(target, index = False)

def encode_data_set(source, target):
    print("encoding data from "+source+" to "+target+"....")

    dataset = pd.read_csv(source)
    df = encode_data(dataset)
    df.to_csv(target, index = False)

def split_data_set(source, training_dir, test_dir):
    print("Splitting data set from "+source+" to "+training_dir+" and "+test_dir)

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, :].values
    X_train, X_test = split_set(X, seed = 69) 

    df = pd.DataFrame(X_train, columns = dataset.columns.values)
    df.to_csv(training_dir, index = False)
    df = pd.DataFrame(X_test, columns = dataset.columns.values)
    df.to_csv(test_dir, index = False)

    
def create_k_fold(source, k):
    target = "kfold/K"+str(k)
    print("Creating "+str(k)+"-fold data split from "+ source+" to "+target)
    
    dataset = pd.read_csv(source)
    X = dataset.iloc[:, :].values
    kf = KFold(n_splits = k, shuffle = True)
    count = 1
    for train_index, test_index in kf.split(X):
        train_target = target + "/train"+str(count)
        test_target = target + "/test"+str(count)
        count += 1
        
        ## normalize for different train/test splits
        

def standarize_data_set(training_source, test_source, training_dir, test_dir):
    print("Performing feature scaling on dataset from "+training_source+" and "+test_source)
    print("Result from scaling is found in "+training_dir+" and "+test_dir)

    training_set = pd.read_csv(training_source)
    X_train_raw = training_set.iloc[:, :-1].values
    y_train_raw = training_set.iloc[:, -1].values
    
    test_set = pd.read_csv(test_source)
    X_test_raw = test_set.iloc[:, :-1].values
    y_test_raw = test_set.iloc[:, -1].values

    y_train = y_train_raw.reshape(y_train_raw.shape[0], 1)
    y_test = y_test_raw.reshape(y_test_raw.shape[0], 1)

    X_train, X_test = scale_data(X_train_raw, X_test_raw)

    new_training_set = np.hstack((X_train, y_train))
    new_test_set = np.hstack((X_test, y_test))
    

    df = pd.DataFrame(new_training_set, columns = training_set.columns.values)
    df.to_csv(training_dir, index = False)
    df = pd.DataFrame(new_test_set, columns = test_set.columns.values)
    df.to_csv(test_dir, index = False)


def feature_selection(source, target, n):
    print("Performing feature selection from "+source+" to "+target)
    
    X = pd.read_csv(source)
    
    X_new, cols= feature_select(X, n)

    column_vals = X.columns.values[np.r_[cols, -1]]
    print(column_vals)
    
    df = pd.DataFrame(X_new, columns = column_vals)
    df.to_csv(target, index = False)
    
    
    
    

def main():
    #%% Creates a reduced csv file with rows and columns specified in create_reduced_csv() ###
    #reduce_data_set("datasets/speeddating.csv", "datasets/reduced.csv")

    #%% Imputing the missing data '?' ###
    #impute_data_set("datasets/reduced.csv", "datasets/imputed.csv")

    #%% Encode categorical data from an imputed csv ###
    #encode_data_set("datasets/imputed.csv", "datasets/encoded.csv")

    #%% Performe feature selection to reduce data set size and avoid overfitting (hopefully)
    feature_selection("datasets/encoded.csv", "datasets/feature_reduced_set.csv", 20)
    
    #%% Split dataset into training and test set ###
    #split_data_set("datasets/encoded.csv", "datasets/raw_training_set.csv", "datasets/raw_test_set.csv")

    #%% Performe feature scaling on training set and test set ###
    #standarize_data_set("datasets/raw_training_set.csv", "datasets/raw_test_set.csv", "datasets/normalized_training_set.csv", "datasets/normalized_test_set.csv")

    #%% Create K-fold set
    #create_k_fold("datasets/encoded.csv", 5)


if __name__ == "__main__":
    main()

