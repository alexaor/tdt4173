import numpy as np
import pandas as pd
from tools import create_reduced_set
from tools import impute_data
from tools import encode_data
from tools import get_categorical_indexes
from tools import split_set
""" CHECKLIST:
- Reduce set: CHECK
- Impute missing data: CHECK
- Encode categorical data: CHECK
- Split data set:
- Normalize data set:
"""

def reduce_data_set(source, target):
    print("Creating reduced set from "+source+", and storing in "+target+"....")

    rows = np.r_[:15]
    columns = np.r_[:15, -4:0]
    reduced = create_reduced_set(rows, columns, source)

    df = pd.DataFrame(reduced)
    df.to_csv(target)
    
def impute_data_set(source, target):
    print("Imputing data from "+source+" to "+target+"....")

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, 1:].values
    imputed = impute_data(X)
    
    df = pd.DataFrame(imputed)
    df.to_csv(target)

def encode_data_set(source, target):
    print("encoding data from "+source+" to "+target+"....")

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, 1:].values
    categorical_columns = get_categorical_indexes(X[1,:])
    encoded = encode_data(X, categorical_columns)

    df = pd.DataFrame(encoded)
    df.to_csv(target)

def split_data_set(source, training_dir, test_dir):
    print("Splitting data set from "+source+" to "+training_dir+" and "+test_dir)

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, 1:].values
    X_train, X_test = split_set(X, seed = 12) 

    df = pd.DataFrame(X_train)
    df.to_csv(training_dir)
    df = pd.DataFrame(X_test)
    df.to_csv(test_dir)


def main():
    #%% Creates a reduced csv file with rows and columns specified in create_reduced_csv() ###
    #reduce_data_set("datasets/speeddating.csv", "datasets/reduced.csv")

    #%% Imputing the missing data '?' ###
    #impute_data_set("datasets/reduced.csv", "datasets/imputed.csv")

    #%% Encode categorical data from an imputed csv ###
    #encode_data_set("datasets/imputed.csv", "datasets/encoded.csv")

    #%% Split dataset into training and test set ###
    split_data_set("datasets/encoded.csv", "datasets/raw_training_set.csv", "datasets/raw_test_set.csv")


if __name__ == "__main__":
    main()

