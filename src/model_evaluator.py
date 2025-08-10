#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation module for breast cancer classification models.
Provides detailed metrics and visualizations for assessing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve


class ModelEvaluator:
    """
    A class to evaluate and visualize model performance.
    """
    
    def __init__(self, model, model_name=None):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        model : object
            Trained classification model with predict and predict_proba methods.
        model_name : str, default=None
            Name of the model for display purposes.
        """
        self.model = model
        self.model_name = model_name or type(model).__name__
        self.metrics = {}
    
    def evaluate(self, X, y_true, threshold=0.5):
        """
        Evaluate the model on the given data.
        
        Parameters:
        -----------
        X : array-like
            Features to evaluate on.
        y_true : array-like
            True labels.
        threshold : float, default=0.5
            Classification threshold for predict_proba.
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        y_pred = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)
            y_prob_pos = y_prob[:, 1]  # Probability of positive class
        else:
            y_prob_pos = None
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_prob_pos is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_pos)
            metrics['average_precision'] = average_precision_score(y_true, y_prob_pos)
        
        # Store predictions and probabilities for later visualization
        metrics['y_true'] = y_true
        metrics['y_pred'] = y_pred
        metrics['y_prob_pos'] = y_prob_pos
        
        # Store the metrics
        self.metrics = metrics
        
        return metrics
    
    def print_metrics(self):
        """Print the evaluation metrics."""
        if not self.metrics:
            print("No evaluation metrics available. Run evaluate first.")
            return
        
        print(f"\nEvaluation metrics for {self.model_name}:")
        
        for metric_name, metric_value in self.metrics.items():
            if metric_name in ['y_true', 'y_pred', 'y_prob_pos', 'confusion_matrix']:
                continue
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.metrics['y_true'], self.metrics['y_pred'],
                                   target_names=['Benign', 'Malignant']))
    
    def plot_confusion_matrix(self, normalize=False, figsize=(8, 6)):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        normalize : bool, default=False
            Whether to normalize the confusion matrix.
        figsize : tuple, default=(8, 6)
            Figure size.
        """
        if 'confusion_matrix' not in self.metrics:
            print("No confusion matrix available. Run evaluate first.")
            return
        
        plt.figure(figsize=figsize)
        cm = self.metrics['confusion_matrix']
        
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
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, figsize=(8, 6)):
        """
        Plot the ROC curve.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size.
        """
        if 'y_prob_pos' not in self.metrics or self.metrics['y_prob_pos'] is None:
            print("No probability scores available. Model may not support predict_proba.")
            return
        
        plt.figure(figsize=figsize)
        
        y_true = self.metrics['y_true']
        y_prob_pos = self.metrics['y_prob_pos']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        roc_auc = self.metrics['roc_auc']
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, figsize=(8, 6)):
        """
        Plot the precision-recall curve.
        
        Parameters:
        -----------
        figsize : tuple, default=(8, 6)
            Figure size.
        """
        if 'y_prob_pos' not in self.metrics or self.metrics['y_prob_pos'] is None:
            print("No probability scores available. Model may not support predict_proba.")
            return
        
        plt.figure(figsize=figsize)
        
        y_true = self.metrics['y_true']
        y_prob_pos = self.metrics['y_prob_pos']
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
        avg_precision = self.metrics['average_precision']
        
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.axhline(y=sum(y_true) / len(y_true), color='r', linestyle='--', 
                   label=f'Baseline (AP = {sum(y_true) / len(y_true):.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
    
    def plot_calibration_curve(self, n_bins=10, figsize=(8, 6)):
        """
        Plot the calibration curve to assess probability calibration.
        
        Parameters:
        -----------
        n_bins : int, default=10
            Number of bins for calibration curve.
        figsize : tuple, default=(8, 6)
            Figure size.
        """
        if 'y_prob_pos' not in self.metrics or self.metrics['y_prob_pos'] is None:
            print("No probability scores available. Model may not support predict_proba.")
            return
        
        plt.figure(figsize=figsize)
        
        y_true = self.metrics['y_true']
        y_prob_pos = self.metrics['y_prob_pos']
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob_pos, n_bins=n_bins)
        
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=self.model_name)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def plot_learning_curve(self, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 5), figsize=(10, 6)):
        """
        Plot the learning curve to assess model performance with varying training set sizes.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
        cv : int, default=5
            Number of cross-validation folds.
        train_sizes : array-like, default=np.linspace(.1, 1.0, 5)
            Training set sizes to plot.
        figsize : tuple, default=(10, 6)
            Figure size.
        """
        plt.figure(figsize=figsize)
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.title(f"Learning Curve - {self.model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    
    def plot_all(self, X=None, y=None):
        """
        Generate all evaluation plots.
        
        Parameters:
        -----------
        X : array-like, default=None
            Features for learning curve. Required only for the learning curve plot.
        y : array-like, default=None
            Target for learning curve. Required only for the learning curve plot.
        """
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_calibration_curve()
        
        if X is not None and y is not None:
            self.plot_learning_curve(X, y)


def evaluate_model(model, X, y_true, model_name=None, plot_all=True, X_train=None, y_train=None):
    """
    Evaluate a model and optionally generate all evaluation plots.
    
    Parameters:
    -----------
    model : object
        Trained classification model.
    X : array-like
        Features to evaluate on.
    y_true : array-like
        True labels.
    model_name : str, default=None
        Name of the model for display purposes.
    plot_all : bool, default=True
        Whether to generate all evaluation plots.
    X_train : array-like, default=None
        Training features for learning curve. Required only if plot_all=True.
    y_train : array-like, default=None
        Training labels for learning curve. Required only if plot_all=True.
    
    Returns:
    --------
    ModelEvaluator
        The evaluator instance.
    """
    evaluator = ModelEvaluator(model, model_name)
    evaluator.evaluate(X, y_true)
    evaluator.print_metrics()
    
    if plot_all:
        if X_train is not None and y_train is not None:
            evaluator.plot_all(X_train, y_train)
        else:
            evaluator.plot_all()
    
    return evaluator


if __name__ == "__main__":
    # Test the model evaluator
    from data_loader import load_data
    from preprocessor import create_default_preprocessor
    from model_trainer import train_default_models
    
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
    
    print("\nEvaluating best model...")
    best_model = trainer.best_model
    model_name = trainer.best_model_name
    
    evaluator = evaluate_model(best_model, X_test_transformed, y_test, model_name,
                              X_train=X_train_transformed, y_train=y_train)