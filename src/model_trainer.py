#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training module for breast cancer classification.
Implements various classification models and hyperparameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
import joblib
import os


class ModelTrainer:
    """
    A class to train and evaluate classification models for breast cancer detection.
    """
    
    def __init__(self, models=None, param_grids=None):
        """
        Initialize the model trainer with specified models and hyperparameter grids.
        
        Parameters:
        -----------
        models : dict, default=None
            Dictionary of model name to model instance.
            If None, default models are used.
        param_grids : dict, default=None
            Dictionary of model name to hyperparameter grid.
            If None, default parameter grids are used.
        """
        self.models = models if models is not None else self._get_default_models()
        self.param_grids = param_grids if param_grids is not None else self._get_default_param_grids()
        self.best_models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    @staticmethod
    def _get_default_models():
        """Get default classification models."""
        return {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier()
        }
    
    @staticmethod
    def _get_default_param_grids():
        """Get default hyperparameter grids for tuning."""
        return {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Manhattan or Euclidean
            }
        }
    
    def train_and_evaluate(self, X_train, y_train, X_test=None, y_test=None, 
                           cv=5, scoring='accuracy', tune_hyperparams=True):
        """
        Train and evaluate models with optional hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training labels.
        X_test : pandas.DataFrame, default=None
            Test features. If None, only cross-validation is performed.
        y_test : pandas.Series, default=None
            Test labels.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, default='accuracy'
            Scoring metric for cross-validation.
        tune_hyperparams : bool, default=True
            Whether to perform hyperparameter tuning.
        
        Returns:
        --------
        dict
            Dictionary of model evaluation results.
        """
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if tune_hyperparams and name in self.param_grids:
                print(f"Performing hyperparameter tuning for {name}...")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=self.param_grids[name],
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model and parameters
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
                
                print(f"Best parameters: {best_params}")
                print(f"Best CV {scoring}: {cv_score:.4f}")
                
                self.best_models[name] = best_model
                
            else:
                # Just perform cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
                cv_score = cv_scores.mean()
                print(f"Cross-validation {scoring}: {cv_score:.4f} ± {cv_scores.std():.4f}")
                
                # Train on full training set
                model.fit(X_train, y_train)
                self.best_models[name] = model
            
            # Evaluate on test set if provided
            if X_test is not None and y_test is not None:
                if tune_hyperparams and name in self.param_grids:
                    current_model = self.best_models[name]
                else:
                    current_model = model.fit(X_train, y_train)
                
                y_pred = current_model.predict(X_test)
                y_pred_proba = current_model.predict_proba(X_test)[:, 1] if hasattr(current_model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                print(f"Test set evaluation for {name}:")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                
                self.results[name] = {
                    'model': current_model,
                    'cv_score': cv_score,
                    'test_metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            else:
                self.results[name] = {
                    'model': self.best_models[name],
                    'cv_score': cv_score
                }
        
        # Determine best model based on CV score
        self.best_model_name = max(self.results, key=lambda k: self.results[k]['cv_score'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name} with CV {scoring}: "
              f"{self.results[self.best_model_name]['cv_score']:.4f}")
        
        if X_test is not None and y_test is not None:
            test_metrics = self.results[self.best_model_name]['test_metrics']
            print(f"Best model test metrics: {test_metrics}")
        
        return self.results
    
    def plot_confusion_matrix(self, y_true, model_name=None, normalize=False):
        """
        Plot the confusion matrix for a model.
        
        Parameters:
        -----------
        y_true : array-like
            True labels.
        model_name : str, default=None
            Name of the model to plot. If None, uses the best model.
        normalize : bool, default=False
            Whether to normalize the confusion matrix.
        """
        if not self.results:
            print("No model results available. Run train_and_evaluate first.")
            return
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results or 'y_pred' not in self.results[model_name]:
            print(f"No predictions available for {model_name}.")
            return
        
        y_pred = self.results[model_name]['y_pred']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_true):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        y_true : array-like
            True labels.
        """
        if not self.results:
            print("No model results available. Run train_and_evaluate first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            if 'y_pred_proba' in result and result['y_pred_proba'] is not None:
                y_pred_proba = result['y_pred_proba']
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = result['test_metrics']['roc_auc']
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def save_model(self, model_name=None, directory='models'):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        model_name : str, default=None
            Name of the model to save. If None, uses the best model.
        directory : str, default='models'
            Directory where to save the model.
        
        Returns:
        --------
        str
            Path to the saved model.
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.best_models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.best_models[model_name]
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path


def train_default_models(X_train, y_train, X_test=None, y_test=None, tune_hyperparams=True):
    """
    Train and evaluate default models.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training labels.
    X_test : pandas.DataFrame, default=None
        Test features.
    y_test : pandas.Series, default=None
        Test labels.
    tune_hyperparams : bool, default=True
        Whether to perform hyperparameter tuning.
    
    Returns:
    --------
    ModelTrainer
        Trained model trainer instance.
    """
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=tune_hyperparams)
    
    if X_test is not None and y_test is not None:
        trainer.plot_confusion_matrix(y_test)
        trainer.plot_roc_curves(y_test)
    
    return trainer


if __name__ == "__main__":
    # Test the model trainer
    from data_loader import load_data
    from preprocessor import create_default_preprocessor
    
    print("Loading data...")
    data = load_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print("\nPreprocessing data...")
    preprocessor = create_default_preprocessor(n_features=15)
    transformed_data = preprocessor.fit_transform(X_train, y_train, X_test)
    
    X_train_transformed = transformed_data['X_train_transformed']
    X_test_transformed = transformed_data['X_test_transformed']
    
    print("\nTraining models...")
    trainer = train_default_models(X_train_transformed, y_train, X_test_transformed, y_test, tune_hyperparams=False)
    
    # Save the best model
    trainer.save_model()