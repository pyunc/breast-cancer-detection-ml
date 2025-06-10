"""
Model training pipeline for the breast cancer detection project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import train_default_models

def train_models_pipeline(X_train, y_train, X_test, y_test, tune_hyperparams=False):
    """Run the model training step of the pipeline."""
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    
    print(f"\n[Step 3] Training models{' with hyperparameter tuning' if tune_hyperparams else ''}...")
    trainer = train_default_models(
        X_train, y_train, 
        X_test, y_test, 
        tune_hyperparams=tune_hyperparams
    )
    
    return {
        'trainer': trainer,
        'best_model': trainer.best_model,
        'best_model_name': trainer.best_model_name
    }
