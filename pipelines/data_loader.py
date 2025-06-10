"""
Data loading pipeline for the breast cancer detection project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data

def load_data_pipeline():
    """Run the data loading step of the pipeline."""
    print("\n" + "="*80)
    print("DATA LOADING PIPELINE")
    print("="*80)
    
    print("\n[Step 1] Loading data...")
    data = load_data()
    print(f"Loaded data with {len(data['X_train'])} training samples and {len(data['X_test'])} test samples")
    
    return data
