#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading module for breast cancer classification.
Handles loading the breast cancer dataset from sklearn.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=42):
    """
    Load the breast cancer dataset and split it into training and test sets.
    
    Parameters:
    -----------
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split.
    
    Returns:
    --------
    dict
        Contains train/test split data and feature information:
        - X_train: training features
        - X_test: test features
        - y_train: training labels
        - y_test: test labels
        - feature_names: names of features
        - target_names: names of target classes
    """
    # Load the breast cancer dataset
    cancer = load_breast_cancer()
    
    # Convert to pandas DataFrame for easier manipulation
    data = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    target = pd.Series(cancer.target, name='target')
    
    # Dataset information
    print(f"Dataset dimensions: {data.shape}")
    print(f"Number of classes: {len(cancer.target_names)}")
    print(f"Class names: {cancer.target_names}")
    print(f"Class distribution: {np.bincount(cancer.target)}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': cancer.feature_names,
        'target_names': cancer.target_names
    }


if __name__ == "__main__":
    # Test the module
    data = load_data()
    print("Data loaded successfully!")