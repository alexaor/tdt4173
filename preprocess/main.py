import numpy as np
import pandas as pd
from tools import create_reduced_set
from tools import impute_data

""" CHECKLIST:
- Reduce set: CHECK
- Impute missing data: CHECK
- Encode categorical data:
- Split data set:
- Normalize data set:
"""

def reduce_data_set():
    rows = np.r_[:10]
    columns = np.r_[:20, -4:0]
    create_reduced_set(rows, columns, "reduced.csv")

def impute_data_set():
    source = "reduced.csv"
    target = "imputed.csv"

    print("Imputing "+source+" to "+target)

    dataset = pd.read_csv(source)
    X = dataset.iloc[:, 1:].values
    imputed = impute_data(X)
    
    df = pd.DataFrame(imputed)
    df.to_csv(target)

def encode_data_set():
    print("hello world")

def main():
    #%% Creates a reduced csv file with rows, columns and filename specified in create_reduced_csv() ###
    #reduce_data_set()

    #%% Imputing the missing data, make sure that you change the filename if neccesary ###
    #impute_data_set()

    #%% Encode categorical data, converting each category into its own column and setting value 1 if the corresponding row belongs to that category ###
    encode_data_set()


if __name__ == "__main__":
    main()

