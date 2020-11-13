import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest


def reduce_impute_encode(df, rows, columns):
    ## Import and reduce dataset ##
    Z = df.iloc[rows, columns].values
    column_labels = df.columns.values[columns]

    ## Impute dataset ##
    imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
    imputer.fit(Z)
    Z_imputed = imputer.transform(Z)
    df = pd.DataFrame(Z_imputed, columns = column_labels)
    
    ## Encoding data ##    
    df = encode_data(df)
    return df.astype(np.float)


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
def feature_scale(Z_train, Z_test):
    X_train = Z_train.iloc[:, :-1].values
    y_train = Z_train.iloc[:, -1].values
    X_test = Z_test.iloc[:, :-1].values
    y_test = Z_test.iloc[:, -1].values
    
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    cols = Z_train.columns.values
    df_train = pd.DataFrame(np.hstack((X_train_scaled, y_train)), columns = cols)
    df_test = pd.DataFrame(np.hstack((X_test_scaled, y_test)), columns = cols)
    
    return df_train, df_test



def feature_selection(Z, n_features):
    X = Z.iloc[:, :-1].values
    Y = Z.iloc[:, -1].values

    feature_selector = SelectKBest(k = n_features)
    feature_selector.fit(X, Y)
    
    column_selections = feature_selector.get_support()
    selected_cols = [col for col in range(len(column_selections)) if column_selections[col]]
    
    labels = Z.columns.values[np.r_[selected_cols, -1]]
    
    X = feature_selector.transform(X)
    Y = Y.reshape(Y.shape[0], 1)
    X_new = np.hstack((X, Y))
    
    df = pd.DataFrame(X_new, columns=labels)

    return df
    