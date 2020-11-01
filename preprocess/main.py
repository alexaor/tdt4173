import numpy as np
import pandas as pd
from tools import create_reduced_set
from tools import impute_data
from tools import encode_data
from tools import get_categorical_indexes

""" CHECKLIST:
- Reduce set: CHECK
- Impute missing data: CHECK
- Encode categorical data: CHECK
- Split data set:
- Normalize data set:
"""

def reduce_data_set(source, target):
    print("Creating reduced set from "+source+", and storing in "+target+"....")

    rows = np.r_[:10]
    columns = np.r_[:15, -4:0]
    reduced = create_reduced_set(rows, columns)

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

def main():
    #%% Creates a reduced csv file with rows and columns specified in create_reduced_csv() ###
    #reduce_data_set("speeddating.csv", "reduced.csv")

    #%% Imputing the missing data '?' ###
    #impute_data_set("reduced.csv", "imputed.csv")

    #%% Encode categorical data from an imputed csv ###
    encode_data_set("imputed.csv", "encoded.csv")


if __name__ == "__main__":
    main()

