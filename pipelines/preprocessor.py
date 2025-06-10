"""
Data preprocessing pipeline for the breast cancer detection project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessor import create_default_preprocessor

def preprocess_pipeline(data, n_features=15, use_pca=False, n_components=5, visualize=False):
    """Run the preprocessing step of the pipeline."""
    print("\n" + "="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
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
    
    return {
        'preprocessor': preprocessor,
        'X_train_transformed': X_train_transformed,
        'X_test_transformed': X_test_transformed,
        'y_train': y_train,
        'y_test': y_test
    }
