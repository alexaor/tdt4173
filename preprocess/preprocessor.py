import numpy as np
import pandas as pd
import pathlib

from sklearn.model_selection import train_test_split

from preprocess.tools import feature_selection
from preprocess.tools import feature_scale
from preprocess.tools import reduce_impute_encode


def create_data_set(filename, n_features=-1, columns=np.r_[2:6,7:12,14:21,39:45,51:56,73:90,107,-1], rows=np.r_[:8378], test_size=0.2, standarize=True):
    ### Defining dataset paths
    origin = pathlib.Path('preprocess/speeddating.csv')
    train_path = pathlib.Path('preprocess/datasets/'+filename+"_train.csv")
    test_path = pathlib.Path('preprocess/datasets/'+filename+"_test.csv")
        
    ### Loading dataset
    Z = pd.read_csv(origin, dtype="str")
    
    ### Pick out columns and rows, impute missing data and encode categorical data
    Z = reduce_impute_encode(Z, rows, columns)
    if n_features != -1:
        ### Choose best n_features
        Z = feature_selection(Z, n_features)

    ### Split data set into test and training
    Z_train, Z_test = train_test_split(Z, test_size=test_size)

    ### Feature scaling by means of standarization
    if standarize:
        Z_train, Z_test = feature_scale(Z_train, Z_test)
        
    ### Save processed dataset
    Z_train.to_csv(train_path, index=False)
    Z_test.to_csv(test_path, index=False)


def main():
    create_data_set("full_set", rows=np.r_[:10])


if __name__ == "__main__":
    main()

