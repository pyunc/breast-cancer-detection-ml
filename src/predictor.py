#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction module for breast cancer classification.
Provides functionality for making predictions with a trained model.
Also exposes a FastAPI interface for making predictions via HTTP.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPredictor:
    """
    A class to make predictions with a trained breast cancer classification model.
    """

    def __init__(self, model=None, preprocessor=None, model_path=None, feature_names=None):
        """
        Initialize the predictor with a model and optional scaler.

        Parameters:
        -----------
        model : object, default=None
            Trained model object. If None, model_path must be provided.
        scaler : object, default=None
            Trained scaler for feature normalization.
        preprocessor : object, default=None
            Trained preprocessor for feature transformation.
        model_path : str, default=None
            Path to a saved model file. Used if model is None.
        feature_names : list, default=None
            List of feature names for the input data.
        """
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be provided.")

        self.model = model
        if model_path is not None and model is None:
            # self.model = joblib.load(model_path)
            self.model = model_path

        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self._last_prediction = None
        self._last_proba = None

    def predict(self, X, apply_scaling=True, apply_preprocessing=True):
        """
        Make predictions with the trained model.

        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        apply_preprocessing : bool, default=True
            Whether to apply preprocessing before scaling.

        Returns:
        --------
        numpy.ndarray
            Predicted classes.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)

        # Apply preprocessing if necessary
        X_processed = X
        if apply_preprocessing and self.preprocessor is not None:
            try:
                # Try to use transform method first
                if hasattr(self.preprocessor, 'transform'):
                    if isinstance(X, pd.DataFrame):
                        X_processed = pd.DataFrame(
                            self.preprocessor.transform(X),
                            columns=X.columns if hasattr(self.preprocessor, 'get_feature_names_out') else X.columns,
                            index=X.index
                        )
                    else:
                        X_processed = self.preprocessor.transform(X)
                # If transform doesn't exist, try __call__ method for function-based preprocessors
                elif callable(self.preprocessor):
                    if isinstance(X, pd.DataFrame):
                        processed_array = self.preprocessor(X.values)
                        X_processed = pd.DataFrame(
                            processed_array,
                            columns=X.columns,
                            index=X.index
                        )
                    else:
                        X_processed = self.preprocessor(X)
                else:
                    print("Warning: Preprocessor doesn't have transform method or is not callable. Skipping preprocessing.")
            except Exception as e:
                print(f"Error during preprocessing: {str(e)}. Continuing without preprocessing.")

        # Apply scaling if necessary
        if apply_scaling:
            if isinstance(X_processed, pd.DataFrame):
                X_processed = pd.DataFrame(
                    self.preprocessor.scaler.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            else:
                X_processed = self.preprocessor.scaler.transform(X_processed)
        
        # Apply feature selection if available
        if hasattr(self.preprocessor, 'feature_selector') and self.preprocessor.feature_selector is not None:
            if isinstance(X_processed, pd.DataFrame):
                X_processed = pd.DataFrame(
                    self.preprocessor.feature_selector.transform(X_processed),
                    columns=X_processed.columns[self.preprocessor.feature_selector.get_support()] if hasattr(self.preprocessor.feature_selector, 'get_support') else X_processed.columns,
                    index=X_processed.index
                )
            else:
                X_processed = self.preprocessor.feature_selector.transform(X_processed)

        # Apply PCA if available
        if hasattr(self.preprocessor, 'pca') and self.preprocessor.pca is not None:
            if isinstance(X_processed, pd.DataFrame):
                X_processed = pd.DataFrame(
                    self.preprocessor.pca.transform(X_processed),
                    columns=[f'PC{i+1}' for i in range(self.preprocessor.pca.n_components_)],
                    index=X_processed.index
                )
            else:
                X_processed = self.preprocessor.pca.transform(X_processed)

        # Make prediction
        self._last_prediction = self.model.predict(X_processed)

        # Store prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self._last_proba = self.model.predict_proba(X_processed)

        return self._last_prediction

    def predict_proba(self, X, apply_scaling=True, apply_preprocessing=True):
        """
        Get prediction probabilities.

        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        apply_preprocessing : bool, default=True
            Whether to apply preprocessing before scaling.

        Returns:
        --------
        numpy.ndarray
            Prediction probabilities.
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction.")

        # Make prediction if not already made
        if self._last_proba is None:
            self.predict(X, apply_scaling, apply_preprocessing)

        return self._last_proba

    def predict_and_explain(self, X, apply_scaling=True, apply_preprocessing=True, class_names=None):
        """
        Make a prediction and provide a basic explanation.

        Parameters:
        -----------
        X : array-like or DataFrame
            Features to predict on.
        apply_scaling : bool, default=True
            Whether to apply feature scaling before prediction.
        apply_preprocessing : bool, default=True
            Whether to apply preprocessing before scaling.
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
        predictions = self.predict(X, apply_scaling, apply_preprocessing)

        # Prepare results
        results = {
            'predictions': predictions,
            'predicted_classes': [class_names[p] for p in predictions],
        }

        # Add probability information if available
        if hasattr(self.model, 'predict_proba'):
            probas = self.predict_proba(X, apply_scaling=False, apply_preprocessing=False)  # Already processed in the predict call
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


def load_model_for_prediction(model_path, preprocessor_path=None, feature_names=None):
    """
    Load a saved model, scaler, and preprocessor for prediction.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file.
    scaler_path : str, default=None
        Path to the saved scaler file.
    preprocessor_path : str, default=None
        Path to the saved preprocessor file.
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

    # Load preprocessor if provided
    preprocessor = None
    if preprocessor_path is not None:
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        else:
            print(f"Warning: Preprocessor file not found: {preprocessor_path}")

    
    # print(preprocessor.scaler,
    # preprocessor.feature_selector,
    # preprocessor.pca,
    # preprocessor.feature_importances)


    return ModelPredictor(model=model, preprocessor=preprocessor, feature_names=feature_names)



if __name__ == "__main__":
    # Test the ModelPredictor class
    model_path = "models/logistic_regression.joblib"
    preprocessor_path = "models/preprocessor.joblib"

    # Load the model and create a predictor
    predictor = load_model_for_prediction(model_path = model_path, preprocessor_path = preprocessor_path)

    # Example input data
    example_data = np.random.rand(1, 30)  # Replace with actual feature data

    # Make a prediction
    prediction = predictor.predict(example_data)
    print(f"Prediction: {prediction}")