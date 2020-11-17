import numpy as np
import pandas as pd
import pathlib

from sklearn.model_selection import train_test_split
from preprocess.tools import filter_desired_features
from preprocess.tools import impute_data
from preprocess.tools import encode_data
from preprocess.tools import feature_selection
from preprocess.tools import standarize_data


def create_data_set(filename, n_features=-1,
                    columns=np.r_[2:27,39:61,73:109,-1], test_size=0.2, 
                    feature_scale=True) -> None:
    """
    Method for creating a data set with specified properties.
    New data set will be written to the dataset folder

    Parameters
    ----------
    filename : string
        Name of the created dataset
    n_features : int, optional
        Number of features selected. Defaults to -1, no feature selection      
    columns : numpy.ndarray, optional
        Indexes of features to keep from the original data set. 
        Default chooses all indexes containing prior knowledge of participants.
    test_size : float, optional
        Specified how much of the data is test set. Default to 0.2
    feature_scale : Bool, optional
        Specifies if feature scaling is desired. Defaults to True.
        
    
    """
    
    ### Defining dataset paths
    origin = pathlib.Path('preprocess/speeddating.csv')
    train_path = pathlib.Path('preprocess/datasets/'+filename+"_train.csv")
    test_path = pathlib.Path('preprocess/datasets/'+filename+"_test.csv")
        
    ### Loading dataset
    Z = pd.read_csv(origin, dtype="str")
    
    ### Remove unwanted attributes
    Z = filter_desired_features(Z, columns)
    
    ### Impute missing data
    Z = impute_data(Z)
    
    ### Encode categorical data
    Z = encode_data(Z)
    
    if n_features != -1:
        ### Choose best n_features
        Z = feature_selection(Z, n_features)

    ### Split data set into test and training
    Z_train, Z_test = train_test_split(Z, test_size=test_size)

    ### Feature scaling by means of standarization
    if feature_scale:
        Z_train, Z_test = standarize_data(Z_train, Z_test)
        
    ### Save processed dataset
    Z_train.to_csv(train_path, index=False)
    Z_test.to_csv(test_path, index=False)