"""
Model saving pipeline for the breast cancer detection project.
"""
import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_model_pipeline(preprocessor, trainer, best_model_name, models_dir='models'):
    """Run the model saving step of the pipeline."""
    print("\n" + "="*80)
    print("MODEL SAVING PIPELINE")
    print("="*80)
    
    print(f"\n[Step 5] Saving model and preprocessor...")
    create_directory(models_dir)
    
    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Save scaler separately for convenience
    # scaler_path = os.path.join(models_dir, 'scaler.joblib')
    # joblib.dump(preprocessor.scaler, scaler_path)
    # print(f"Scaler saved to {scaler_path}")
    
    # Save best model
    model_path = trainer.save_model(best_model_name, models_dir)
    
    return {
        'preprocessor_path': preprocessor_path,
        # 'scaler_path': scaler_path,
        'model_path': model_path
    }
