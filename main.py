#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run the breast cancer classification pipeline.
This script orchestrates the entire workflow from data loading to prediction.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
from preprocessor import create_default_preprocessor
from model_trainer import train_default_models
from model_evaluator import evaluate_model
from predictor import ModelPredictor


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def run_pipeline(tune_hyperparams=False, n_features=15, use_pca=False, n_components=5,
                save_model=True, models_dir='models', visualize=True):
    """
    Run the complete breast cancer classification pipeline.
    
    Parameters:
    -----------
    tune_hyperparams : bool, default=False
        Whether to perform hyperparameter tuning.
    n_features : int, default=15
        Number of top features to select.
    use_pca : bool, default=False
        Whether to apply PCA for dimensionality reduction.
    n_components : int, default=5
        Number of PCA components if use_pca is True.
    save_model : bool, default=True
        Whether to save the best model.
    models_dir : str, default='models'
        Directory to save models.
    visualize : bool, default=True
        Whether to generate visualizations.
    
    Returns:
    --------
    dict
        Results dictionary containing trained objects and evaluation metrics.
    """
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Step 1: Load and split the data
    print("\n[Step 1] Loading data...")
    data = load_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    feature_names = data['feature_names']
    target_names = data['target_names']
    
    # Step 2: Preprocess the data
    print("\n[Step 2] Preprocessing data...")
    preprocessor = create_default_preprocessor(
        n_features=n_features, 
        use_pca=use_pca, 
        n_components=n_components
    )
    transformed_data = preprocessor.fit_transform(X_train, y_train, X_test)
    
    X_train_transformed = transformed_data['X_train_transformed']
    X_test_transformed = transformed_data['X_test_transformed']
    
    # Optional: Visualize feature importances
    if visualize and hasattr(preprocessor, 'feature_importances') and preprocessor.feature_importances is not None:
        print("\nTop feature importances:")
        plt.figure(figsize=(10, 6))
        top_features = preprocessor.feature_importances.nlargest(10)
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.show()
    
    # Step 3: Train models
    print(f"\n[Step 3] Training models{' with hyperparameter tuning' if tune_hyperparams else ''}...")
    trainer = train_default_models(
        X_train_transformed, y_train, 
        X_test_transformed, y_test, 
        tune_hyperparams=tune_hyperparams
    )
    
    # Step 4: Evaluate the best model
    print("\n[Step 4] Evaluating best model...")
    best_model = trainer.best_model
    best_model_name = trainer.best_model_name
    
    evaluator = evaluate_model(
        best_model, X_test_transformed, y_test, 
        model_name=best_model_name,
        plot_all=visualize,
        X_train=X_train_transformed, 
        y_train=y_train
    )
    
    # Step 5: Save the preprocessor and best model
    if save_model:
        print(f"\n[Step 5] Saving model and preprocessor...")
        create_directory(models_dir)
        
        # Save preprocessor
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Save scaler separately for convenience
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        joblib.dump(preprocessor.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Save best model
        model_path = trainer.save_model(best_model_name, models_dir)
    
    # Return results
    results = {
        'data': data,
        'preprocessor': preprocessor,
        'trainer': trainer,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'evaluator': evaluator,
        'X_train_transformed': X_train_transformed,
        'X_test_transformed': X_test_transformed
    }
    
    print("\n" + "="*80)
    print(f"Pipeline completed successfully! Best model: {best_model_name}")
    print("="*80 + "\n")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Breast Cancer Classification Pipeline")
    
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    
    parser.add_argument('--features', type=int, default=15,
                        help='Number of top features to select')
    
    parser.add_argument('--pca', action='store_true',
                        help='Apply PCA for dimensionality reduction')
    
    parser.add_argument('--components', type=int, default=5,
                        help='Number of PCA components')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    
    parser.add_argument('--no-viz', action='store_true',
                        help='Do not generate visualizations')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = run_pipeline(
        tune_hyperparams=args.tune,
        n_features=args.features,
        use_pca=args.pca,
        n_components=args.components,
        save_model=not args.no_save,
        visualize=not args.no_viz
    )