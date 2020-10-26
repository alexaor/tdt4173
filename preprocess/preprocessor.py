#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:11:13 2020

@author: oveggeland
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_transformer import transform_data


def scale_data(X_train, X_test):
    sc = StandardScaler()
    X_train[:, 598:] = sc.fit_transform(X_train[:, 598:])
    X_test[:, 598:] = sc.transform(X_test[:, 598:])
    return X_train, X_test

def main():
    #transform_data()
    print("data transformation finished")
    
    dataset = pd.read_csv('transformed.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    X_train, X_test = scale_data(X_train, X_test)
    
main()