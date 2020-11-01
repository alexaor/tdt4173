import numpy as np
import pandas as pd

def create_reduced_set():
    dataset = pd.read_csv('speeddating.csv', dtype = 'str')
    X = dataset.iloc[:6, np.r_[1:20,-2,-1]].values
    df = pd.DataFrame(X)
    df.to_csv("reduced.csv")

def transform_data():    
    dataset = pd.read_csv('speeddating.csv', dtype = 'str')
    X = dataset.iloc[:, :].values
    
    df = pd.DataFrame(X)
    df.to_csv('original.csv')
    
    ### Take care of missing data ###
    imputed_data = impute_data(X)
          
    df = pd.DataFrame(imputed_data)
    df.to_csv('imputed.csv')


    ### Transform categorical data ###
    categorical_indexes = get_categorical_indexes(X[0, :])
    transformed_data = encode_columns(X, categorical_indexes)
    
    df = pd.DataFrame(transformed_data)
    df.to_csv('transformed.csv')
    
    
create_reduced_set()
