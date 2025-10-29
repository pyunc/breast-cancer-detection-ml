#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction pipeline for the breast cancer detection project.
This module provides functionality to make predictions using trained models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from pipelines.predictor import app

from pipelines.predictor import initialize_model    

if __name__ == "__main__":
    # Test the FastAPI app
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the breast cancer prediction API server')
    parser.add_argument('--model-path', type=str, required=False, help='Path to the saved model file', default='/home/paulo-huna/Documents/breast-cancer-detection-ml/models/logistic_regression.joblib')
    parser.add_argument('--preprocessor-path', type=str, required=False, help='Path to the saved preprocessor file', default='/home/paulo-huna/Documents/breast-cancer-detection-ml/models/preprocessor.joblib')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    
    args = parser.parse_args()

    global model_predictor
    
    try:
        print(f"ml_service starting with args: model={args.model_path}, preprocessor={args.preprocessor_path}")
        print(f"ENV MODEL_URL={os.getenv('MODEL_URL')}")
        print(f"ENV PREPROCESSOR_URL={os.getenv('PREPROCESSOR_URL')}")
        
        if not initialize_model(args.model_path, args.preprocessor_path):
            print("Failed to initialize model. Exiting.")
            sys.exit(1)
        
        print(f"Model loaded successfully from {args.model_path} (or env fallback)")

        # Start the server
        print(f"Starting FastAPI server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

