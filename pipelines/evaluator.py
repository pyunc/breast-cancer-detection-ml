"""
Model evaluation pipeline for the breast cancer detection project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_evaluator import evaluate_model

def evaluate_pipeline(model, X_test, y_test, model_name, X_train=None, y_train=None, visualize=False):
    """Run the model evaluation step of the pipeline."""
    print("\n" + "="*80)
    print("MODEL EVALUATION PIPELINE")
    print("="*80)
    
    print("\n[Step 4] Evaluating best model...")
    evaluator = evaluate_model(
        model, X_test, y_test, 
        model_name=model_name,
        plot_all=visualize,
        X_train=X_train, 
        y_train=y_train
    )
    
    return {'evaluator': evaluator}
