import numpy as np
import pandas as pd
import os

from configs.project_settings import DATASET_PATH

from sklearn.model_selection import train_test_split

from preprocessing.utils import filter_desired_features
from preprocessing.utils import impute_data
from preprocessing.utils import encode_data
from preprocessing.utils import feature_selection
from preprocessing.utils import standarize_data


def create_data_set(filename, n_features=-1, test_size=0.2,
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
    test_size : float, optional
        Specified how much of the data is test set. Default to 0.2
    feature_scale : Bool, optional
        Specifies if feature scaling is desired. Defaults to True.
    """

    # Defining dataset paths
    origin = os.path.join(DATASET_PATH, "01_raw_speeddating.csv")
    train_path = os.path.join(DATASET_PATH, f"02_{filename}_train.csv")
    test_path = os.path.join(DATASET_PATH, f"02_{filename}_test.csv")

    # Loading dataset
    Z = pd.read_csv(origin, dtype="str")

    # Remove unwanted attributes
    columns = np.r_[2:27, 39:61, 73:109, -1]
    Z = filter_desired_features(Z, columns)

    # Impute missing data
    Z = impute_data(Z)

    # Encode categorical data
    Z = encode_data(Z)

    if n_features != -1:
        # Choose best n_features
        Z = feature_selection(Z, n_features)

    # Split data set into test and training
    Z_train, Z_test = train_test_split(Z, test_size=test_size)

    # Feature scaling by means of standarization
    if feature_scale:
        Z_train, Z_test = standarize_data(Z_train, Z_test)

    # Save processed dataset
    Z_train.to_csv(train_path, index=False)
    Z_test.to_csv(test_path, index=False)
