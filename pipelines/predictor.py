import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
# load dot env file
from dotenv import load_dotenv
load_dotenv()



from src.predictor import ModelPredictor, load_model_for_prediction  # noqa: E402

# Add FastAPI dependency for model access - this is the improvement
def get_model_predictor():
    """Dependency to get the current model predictor instance."""
    global model_predictor
    if model_predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please initialize the model first."
        )
    return model_predictor

# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """
    Pydantic model for prediction request data validation.
    """
    features: List[float] = Field(..., description="List of feature values for prediction")
    apply_scaling: bool = Field(True, description="Whether to apply feature scaling")
    apply_preprocessing: bool = Field(True, description="Whether to apply preprocessing")

class PredictionBatchInput(BaseModel):
    """
    Pydantic model for batch prediction request data validation.
    """
    data: List[List[float]] = Field(..., description="List of samples for prediction")
    apply_scaling: bool = Field(True, description="Whether to apply feature scaling")
    apply_preprocessing: bool = Field(True, description="Whether to apply preprocessing")

class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response data.
    """
    prediction: int = Field(..., description="Predicted class (0: Benign, 1: Malignant)")
    predicted_class: str = Field(..., description="Predicted class name")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[List[float]] = Field(None, description="Class probabilities")
    status: str = Field("Success", description="Status of the prediction (Success, Failed)")
    message: Optional[str] = Field(None, description="Additional status information")

class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction response data.
    """
    predictions: List[int] = Field(..., description="Predicted classes")
    predicted_classes: List[str] = Field(..., description="Predicted class names")
    confidences: Optional[List[float]] = Field(None, description="Prediction confidences")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")

class ModelExplanationResponse(BaseModel):
    """
    Pydantic model for model explanation response data.
    """
    predictions: List[int] = Field(..., description="Predicted classes")
    predicted_classes: List[str] = Field(..., description="Predicted class names")
    confidence: Optional[List[float]] = Field(None, description="Prediction confidences")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    feature_importances: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for breast cancer classification using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the model and predictor
model_predictor = None
class_names = ['Benign', 'Malignant']

# Add an initialization function to be called at startup
def initialize_model(model_path, preprocessor_path=None):
    """Initialize the model predictor when the API starts."""
    global model_predictor
    try:
        model_predictor = load_model_for_prediction(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        print(f"Model loaded successfully from {model_path}")
        if preprocessor_path:
            print(f"Preprocessor loaded successfully from {preprocessor_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


@app.get("/")
async def root():
    """Root endpoint to check if API is running."""

    # get environment variables for APP_NAME, APP_VERSION, APP_AUTHOR
    app_name = os.getenv("APP_NAME", "Breast Cancer Prediction API")
    app_version = os.getenv("APP_VERSION", "1.0.0")
    app_author = os.getenv("APP_AUTHOR", "Your Name")
    app_server = os.getenv("APP_SERVER", "Your Server")


    return {
        "message": f"{app_name} is running",
        "version": app_version,
        "author": app_author,
        "server": app_server
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput, predictor: ModelPredictor = Depends(get_model_predictor)):
    """
    Make a single prediction with the loaded model.
    """
    try:
        # Convert input data to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = predictor.predict(
            features, 
            apply_scaling=input_data.apply_scaling, 
            apply_preprocessing=input_data.apply_preprocessing
        )
        
        # Build response
        response = {
            "prediction": int(prediction[0]),
            "predicted_class": class_names[prediction[0]],
            "status": "Success",
            "message": "Prediction completed successfully"
        }
        
        # Add probabilities if available
        if hasattr(predictor.model, 'predict_proba'):
            probas = predictor.predict_proba(
                features, 
                apply_scaling=False, 
                apply_preprocessing=False
            )
            response["probabilities"] = probas[0].tolist()
            response["confidence"] = float(np.max(probas[0]))
            
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ModelExplanationResponse)
async def explain(input_data: PredictionBatchInput, predictor: ModelPredictor = Depends(get_model_predictor)):
    """
    Make predictions and provide explanations.
    """
    try:
        # Convert input data to numpy array
        features = np.array(input_data.data)
        
        # Make prediction with explanation
        results = predictor.predict_and_explain(
            features, 
            apply_scaling=input_data.apply_scaling,
            apply_preprocessing=input_data.apply_preprocessing, 
            class_names=class_names
        )
        
        # Convert feature importances to dict if present
        if 'feature_importances' in results:
            results['feature_importances'] = results['feature_importances'].to_dict()
        
        # Handle probabilities
        if 'probabilities' in results:
            results['probabilities'] = results['probabilities'].tolist()
            
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@app.get("/model-info")
async def model_info(predictor: ModelPredictor = Depends(get_model_predictor)):
    """
    Get information about the loaded model.
    """
    info = {
        "model_type": type(model_predictor.model).__name__,
        "has_feature_names": model_predictor.feature_names is not None,
        # "has_scaler": model_predictor.scaler is not None,
        "has_preprocessor": model_predictor.preprocessor is not None,
        "supports_probabilities": hasattr(model_predictor.model, 'predict_proba'),
    }
    
    if hasattr(model_predictor.model, 'feature_importances_') and model_predictor.feature_names is not None:
        # Only return top 10 for the info endpoint
        importances = pd.Series(
            model_predictor.model.feature_importances_,
            index=model_predictor.feature_names
        ).sort_values(ascending=False).head(10)
        
        info["top_features"] = importances.to_dict()
    