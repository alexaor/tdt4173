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
def encode_columns(X, index_list):
    transformers=[('encoder'+str(i), OneHotEncoder(sparse = False), [index_list[i]])
                  for i in range(len(index_list))]
    ct = ColumnTransformer(transformers, remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    return X

def impute_data(X):
    imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    return X_imputed    


#Returns a list of indexes corresponding to the columns of our categorical data
def get_categorical_indexes(data_row):
    index_list = []
    for i in range(len(data_row)):
        try: 
            _val = int(data_row[i])
            pass
        except:
            index_list.append(i)
    return index_list

"""
Writes a CSV with imputations and transformed categorical data.
"""
def transform_data():    
    dataset = pd.read_csv('../speeddating.csv', dtype = 'str')
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
    
    

transform_data()