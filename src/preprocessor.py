#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing module for breast cancer classification.
Handles feature scaling, selection and dimensionality reduction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


class Preprocessor:
    """
    A class to handle preprocessing tasks for breast cancer data.
    """
    
    def __init__(self, scaler=None, feature_selector=None, pca=None):
        """
        Initialize the preprocessor with optional components.
        
        Parameters:
        -----------
        scaler : object, default=None
            Feature scaling transformer. If None, StandardScaler is used.
        feature_selector : object, default=None
            Feature selection transformer. If None, SelectKBest is used.
        pca : object, default=None
            PCA transformer for dimensionality reduction. If None, PCA is not applied.
        """
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.feature_selector = feature_selector
        self.pca = pca
        self.feature_importances = None
    
    def fit_transform(self, X_train, y_train=None, X_test=None):
        """
        Fit the preprocessing pipeline to training data and transform both train and test data.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series, default=None
            Training labels, needed for supervised feature selection.
        X_test : pandas.DataFrame, default=None
            Test features. If None, only X_train is transformed.
        
        Returns:
        --------
        dict
            Transformed data:
            - X_train_transformed: transformed training features
            - X_test_transformed: transformed test features (if X_test was provided)
        """
        # Step 1: Scale the features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Step 2: Apply feature selection if needed
        X_train_selected = X_train_scaled
        if self.feature_selector is not None and y_train is not None:
            X_train_selected = pd.DataFrame(
                self.feature_selector.fit_transform(X_train_scaled, y_train),
                columns=X_train.columns[self.feature_selector.get_support()],
                index=X_train.index
            )
            # Store feature importances if available
            if hasattr(self.feature_selector, 'scores_'):
                self.feature_importances = pd.Series(
                    self.feature_selector.scores_,
                    index=X_train.columns
                ).sort_values(ascending=False)
        
        # Step 3: Apply PCA if needed
        X_train_transformed = X_train_selected
        if self.pca is not None:
            X_train_transformed = pd.DataFrame(
                self.pca.fit_transform(X_train_selected),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                index=X_train.index
            )
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            X_test_selected = X_test_scaled
            if self.feature_selector is not None:
                X_test_selected = pd.DataFrame(
                    self.feature_selector.transform(X_test_scaled),
                    columns=X_test.columns[self.feature_selector.get_support()],
                    index=X_test.index
                )
            
            X_test_transformed = X_test_selected
            if self.pca is not None:
                X_test_transformed = pd.DataFrame(
                    self.pca.transform(X_test_selected),
                    columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                    index=X_test.index
                )
            
            return {
                'X_train_transformed': X_train_transformed,
                'X_test_transformed': X_test_transformed
            }
        
        return {'X_train_transformed': X_train_transformed}
    
    def transform(self, X):
        """
        Apply the preprocessing pipeline to transform data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Features to transform.
        
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            Transformed features.
        """
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        
        # Step 1: Scale the features
        if self.scaler is not None:
            if is_dataframe:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=index
                )
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Step 2: Apply feature selection if needed
        if self.feature_selector is not None:
            if is_dataframe:
                X_selected = pd.DataFrame(
                    self.feature_selector.transform(X_scaled),
                    columns=X.columns[self.feature_selector.get_support()],
                    index=index
                )
            else:
                X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Step 3: Apply PCA if needed
        if self.pca is not None:
            if is_dataframe:
                X_transformed = pd.DataFrame(
                    self.pca.transform(X_selected),
                    columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                    index=index
                )
            else:
                X_transformed = self.pca.transform(X_selected)
        else:
            X_transformed = X_selected
        
        return X_transformed
        
    def plot_feature_importances(self, top_n=10):
        """
        Plot the top N most important features.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to display.
        """
        if self.feature_importances is None:
            print("No feature importances available. Run fit_transform first.")
            return
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importances.nlargest(top_n)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()


def create_default_preprocessor(n_features=10, use_pca=False, n_components=2):
    """
    Create a preprocessor with default settings.
    
    Parameters:
    -----------
    n_features : int, default=10
        Number of top features to select.
    use_pca : bool, default=False
        Whether to apply PCA.
    n_components : int, default=2
        Number of PCA components if use_pca is True.
    
    Returns:
    --------
    Preprocessor
        A configured preprocessor instance.
    """
    feature_selector = SelectKBest(f_classif, k=n_features)
    pca = PCA(n_components=n_components) if use_pca else None
    scaler = StandardScaler()

    return Preprocessor(
        scaler=scaler,
        feature_selector=feature_selector,
        pca=pca
    )

if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_data
    
    data = load_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Create and apply preprocessor
    preprocessor = create_default_preprocessor(n_features=15, use_pca=True, n_components=5)

    transformed_data = preprocessor.fit_transform(X_train, y_train, X_test)

    # print(preprocessor.scaler)
    # print(preprocessor.feature_selector)
    # print(preprocessor.pca)
    # print(preprocessor.feature_importances)
    
    print("\nTransformed training data shape:", transformed_data['X_train_transformed'].shape)
    print("Transformed test data shape:", transformed_data['X_test_transformed'].shape)
    
    # Plot feature importances
    preprocessor.plot_feature_importances()