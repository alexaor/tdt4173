import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest


"""
@param rows: list of row slices in reduced set
@param columns: list of column slices in reduced set
"""
def create_reduced_set(rows, columns, source):
    print("\n___Creating reduced csv___")
    print("Row indexes: " + str(rows))
    print("Columns indexes: " + str(columns)+"\n")
    dataset = pd.read_csv(source)
    X = dataset.iloc[rows, columns].values
    column_values = dataset.columns.values[columns]
    return X, column_values


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


"""
@param X: Array of variables to encode
@param index_list: List of column indexes that should be transformed
"""
def encode_data(dataset):
    dataset = dataset.apply(lambda x: x.astype(str).str.lower())
    ## Retrieving values and column_names ##
    X = dataset.iloc[:, :].values
    column_names = dataset.columns.values
    cat_cols = get_categorical_indexes(X[1,:])
    cat_col_names = column_names[cat_cols]

    ## Transformation ##
    print("___transforming columns "+str(cat_cols)+"___")
    transformers=[(str(cat_col_names[i]), OneHotEncoder(sparse = False), [cat_cols[i]])
                  for i in range(len(cat_cols))]
    ct = ColumnTransformer(transformers, remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    
    ## Create DataFrame with proper column names ##
    column_names = np.delete(column_names, cat_cols)
    cat_col_names = ct.get_feature_names()
    cat_col_names = [raw.replace("_x0_", "") for raw in cat_col_names]
    cat_col_names = [element for element in cat_col_names if element[0] != "x"]
    column_names = np.concatenate((cat_col_names, column_names))
    
    df = pd.DataFrame(X, columns = column_names)
    return df


"""
@param X: Independent variable array
@param Y: Dependent variable vector
@param size: Size of test set (between 0 and 1)
@param seed: Optional seed for reproducing results
"""
def split_set(X, size = 0.2, seed = None):
    return train_test_split(X, test_size = size, random_state = seed) 



"""
@param X_train: Array of training data
@param X_test: Array of test data
"""
def scale_data(X_train_raw, X_test_raw):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_raw)
    X_test = sc.transform(X_test_raw)
    return X_train, X_test



def feature_select(train_set, test_set):
    X_train = train_set.iloc[:, :-1].values
    Y_train = train_set.iloc[:, -1].values
    X_test = test_set.iloc[:, :-1].values
    Y_test = test_set.iloc[:, -1].values
      
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))

    
    feature_selector = SelectKBest(k = 10)
    feature_selector.fit(X, Y)
    
    column_selections = feature_selector.get_support()
    #cat_col_names = [element for element in cat_col_names if element[0] != "x"]
    selected_cols = [col for col in range(len(column_selections)) if column_selections[col]]
    
    X_train_reduced = feature_selector.transform(X_train)
    X_test_reduced = feature_selector.transform(X_test)
    
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    
    new_training_set = np.hstack((X_train_reduced, Y_train))
    new_test_set = np.hstack((X_test_reduced, Y_test))
    
    return new_training_set, new_test_set, selected_cols
    