import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest


def filter_desired_features(dataframe, columns):
    """
    Method to pick out specified features from a data set

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing data to choose features from
    columns : numpy.ndarray
        Indexes of columns to be selected from the original dataframe

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the imputed data
    """
    
    Z = dataframe.iloc[:, columns].values
    column_labels = dataframe.columns.values[columns]
    
    df = pd.DataFrame(Z, columns = column_labels)
    return df
    
def impute_data(dataframe):
    """
    Method to impute missing values, denoted by '?', in a data set. 

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing data to be imputed

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the imputed data
    """
    
    Z = dataframe.iloc[:, :].values
    column_labels = dataframe.columns.values
    
    imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
    imputer.fit(Z)
    Z_imputed = imputer.transform(Z)
    
    df = pd.DataFrame(Z_imputed, columns = column_labels)
    return df
    

def get_categorical_indexes(data_row):
    """
    Helper function. Finds the indexes of categorical data by iterating through a single data row.

    Parameters
    ----------
    data_row : numpy.ndarray
        Any row of features from the data-set

    Returns
    -------
    index_list : List
        A list of indexes to features containing categorical data
    """
    index_list = []
    for i in range(len(data_row)):
        try: 
            _val = float(data_row[i])
            pass
        except:
            index_list.append(i)
    return index_list


def encode_data(dataframe):
    """
    Method for encoding gategorical features of a dataset

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing data to be encoded

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the encoded data
    """
    
    #avoid multiple encodings for spelling differences
    dataframe = dataframe.apply(lambda x: x.astype(str).str.lower())
    
    # Retrieving values and column_names
    X = dataframe.iloc[:, :].values
    column_names = dataframe.columns.values
    cat_cols = get_categorical_indexes(X[1,:])
    cat_col_names = column_names[cat_cols]

    # Transform data by means of one hot encoding
    transformers=[(str(cat_col_names[i]), OneHotEncoder(sparse = False), [cat_cols[i]])
                  for i in range(len(cat_cols))]
    ct = ColumnTransformer(transformers, remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    
    # Return dataframe with column-names adjusted to reflect their encodings
    column_names = np.delete(column_names, cat_cols)
    cat_col_names = ct.get_feature_names()
    cat_col_names = [raw.replace("_x0_", "") for raw in cat_col_names]
    cat_col_names = [element for element in cat_col_names if element[0] != "x"]
    column_names = np.concatenate((cat_col_names, column_names))
    
    df = pd.DataFrame(X, columns = column_names)
    return df.astype(np.float)


def feature_selection(dataframe, n_features):
    """
    Method for selecting the best features for classification by comparing f-values

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing data to be feature selected
    n_featurs : int
        Number of features to be selected

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the feature selected data set
    """
    
    X = dataframe.iloc[:, :-1].values
    Y = dataframe.iloc[:, -1].values

    feature_selector = SelectKBest(k = n_features)
    feature_selector.fit(X, Y)
    
    column_selections = feature_selector.get_support()
    selected_cols = [col for col in range(len(column_selections)) if column_selections[col]]
    labels = dataframe.columns.values[np.r_[selected_cols, -1]]
    
    X = feature_selector.transform(X)
    Y = Y.reshape(Y.shape[0], 1)
    X_new = np.hstack((X, Y))
    
    df = pd.DataFrame(X_new, columns=labels)
    return df
    

def standarize_data(df_train, df_test):
    """
    Method for standarizing a train and test set. Note: The standarizer is only trained on the training data

    Parameters
    ----------
    df_train : pandas.DataFrame
        Dataframe containing training set to be standarized
    df_test : pandas.DataFrame
        Dataframe containing test set to be standarized

    Returns
    -------
    df_train_std : pandas.DataFrame
        Dataframe containing standarized training set
    df_test_std : pandas.DataFrame
        Dataframe containing standarized test set
    """
    
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values
    
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cols = df_train.columns.values
    df_train_std = pd.DataFrame(np.hstack((X_train_std, y_train)), columns = cols)
    df_test_std = pd.DataFrame(np.hstack((X_test_std, y_test)), columns = cols)
    
    return df_train_std, df_test_std