#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction module for breast cancer classification.
Provides functionality for making predictions with a trained model.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPredictor:
    """
    A class to make predictions with a trained breast cancer classification model.
    """
    
    def __init__(self, model=None, scaler=None, model_path=None, feature_names=None):
        """
        Initialize the predictor with a model and optional scaler.
        
        Parameters:
        -----------
        model : object, default=None
            Trained model object. If None, model_path must be provided.
        scaler : object, default=None
            Trained scaler for feature normalization.
        model_path : str, default=None
            Path to a saved model file. Used if model is None.
        feature_names : list, default=None
            List of feature names for the input data.
        """
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        
        self.model = model
        if model_path is not None and model is None:
            self.model = joblib.load(model_path)
        
        self.scaler = scaler
        self.feature_names = feature_names
        self._last_prediction = None
        self._last_proba = None
    
    def predict(self, X, apply_scaling=True):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        
        Returns:
        --------
        numpy.ndarray
            Predicted classes.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply scaling if necessary
        X_processed = X
        if apply_scaling and self.scaler is not None:
            if isinstance(X, pd.DataFrame):
                X_processed = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_processed = self.scaler.transform(X)
        
        # Make prediction
        self._last_prediction = self.model.predict(X_processed)
        
        # Store prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self._last_proba = self.model.predict_proba(X_processed)
        
        return self._last_prediction
    
    def predict_proba(self, X, apply_scaling=True):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        
        Returns:
        --------
        numpy.ndarray
            Prediction probabilities.
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction.")
        
        # Make prediction if not already made
        if self._last_proba is None:
            self.predict(X, apply_scaling)
        
        return self._last_proba
    
    def predict_and_explain(self, X, apply_scaling=True, class_names=None):
        """
        Make a prediction and provide a basic explanation.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        class_names : list, default=None
            Names of the target classes ['Benign', 'Malignant'].
        
        Returns:
        --------
        dict
            Prediction results and explanation.
        """
        if class_names is None:
            class_names = ['Benign', 'Malignant']
        
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Make prediction
        predictions = self.predict(X, apply_scaling)
        
        # Prepare results
        results = {
            'predictions': predictions,
            'predicted_classes': [class_names[p] for p in predictions],
        }
        
        # Add probability information if available
        if hasattr(self.model, 'predict_proba'):
            probas = self.predict_proba(X, apply_scaling=False)  # Already scaled in the predict call
            results['probabilities'] = probas
            results['confidence'] = np.max(probas, axis=1)
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_') and isinstance(X, pd.DataFrame):
            feature_importances = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            results['feature_importances'] = feature_importances
        
        return results
    
    def plot_feature_importance(self, X=None, top_n=10, figsize=(10, 6)):
        """
        Plot feature importances if available.
        
        Parameters:
        -----------
        X : DataFrame, default=None
            Features to get column names from.
        top_n : int, default=10
            Number of top features to show.
        figsize : tuple, default=(10, 6)
            Figure size.
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not provide feature importances.")
            return
        
        plt.figure(figsize=figsize)
        
        # Get feature names
        if X is not None and isinstance(X, pd.DataFrame):
            feature_names = X.columns
        elif self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature {i}' for i in range(len(self.model.feature_importances_))]
        
        # Create Series of feature importances
        importances = pd.Series(
            self.model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        
        # Plot top N features
        top_importances = importances.head(top_n)
        sns.barplot(x=top_importances.values, y=top_importances.index)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importances


def load_model_for_prediction(model_path, scaler_path=None, feature_names=None):
    """
    Load a saved model and scaler for prediction.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file.
    scaler_path : str, default=None
        Path to the saved scaler file.
    feature_names : list, default=None
        List of feature names for the input data.
    
    Returns:
    --------
    ModelPredictor
        Initialized model predictor.
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load scaler if provided
    scaler = None
    if scaler_path is not None:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: Scaler file not found: {scaler_path}")
    
    return ModelPredictor(model=model, scaler=scaler, feature_names=feature_names)


if __name__ == "__main__":
    # Test the predictor
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Create predictor
    predictor = ModelPredictor(model=model, scaler=scaler, feature_names=cancer.feature_names)
    
    # Make predictions on test data
    predictions = predictor.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Make predictions with explanation
    sample = X_test.iloc[:5]
    results = predictor.predict_and_explain(sample, class_names=cancer.target_names)
    
    # Print results
    for i, pred_class in enumerate(results['predicted_classes']):
        confidence = results['confidence'][i] if 'confidence' in results else None
        print(f"Sample {i+1}: Predicted as {pred_class}" + 
              (f" with confidence {confidence:.4f}" if confidence is not None else ""))
    
    # Plot feature importance
    predictor.plot_feature_importance(X_test)