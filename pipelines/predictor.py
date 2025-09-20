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

# Added import for remote artifact download
import requests  # noqa: E402

# NEW: logging for debug visibility
import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("inference")



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

#TODO load model from digital ocean space or local path based on env variable
# TODO think on integration with cloud
# TODO in inference-pipeline-deployment.yaml for this part 
# - --model-path
# https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/models/29-0cb1adcc63c10a4b5b571bf5dbe221edf4e0c82d/logistic_regression.joblib
# - /opt/ml/models/logistic_regression.joblib
# --preprocessor-path
# https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/models/29-0cb1adcc63c10a4b5b571bf5dbe221edf4e0c82d/preprocessor.joblib

# https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/models/29-0cb1adcc63c10a4b5b571bf5dbe221edf4e0c82d/logistic_regression.joblib
# https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/models/29-0cb1adcc63c10a4b5b571bf5dbe221edf4e0c82d/preprocessor.joblib
# default local 

# i have to direct to the digital ocean bucket and load from there

# Add an initialization function to be called at startup
def initialize_model(model_path, preprocessor_path=None):
    """Initialize the model predictor when the API starts.

    Supports remote (HTTP/HTTPS) artifact download and environment variable fallbacks:
      - MODEL_URL
      - PREPROCESSOR_URL
    """
    global model_predictor

    logger.info("initialize_model called")
    logger.info(f"Incoming model_path arg: {model_path}")
    logger.info(f"Incoming preprocessor_path arg: {preprocessor_path}")
    logger.info(f"Env MODEL_URL: {os.getenv('MODEL_URL')}")
    logger.info(f"Env PREPROCESSOR_URL: {os.getenv('PREPROCESSOR_URL')}")

    # Environment variable fallback if arguments are empty/placeholder/None
    PLACEHOLDERS = {None, "", "__MODEL_URL__", "__PREPROCESSOR_URL__"}
    if model_path in PLACEHOLDERS:
        logger.warning("model_path is empty/placeholder; falling back to env MODEL_URL")
        model_path = os.getenv("MODEL_URL")
    if preprocessor_path in PLACEHOLDERS:
        logger.info("preprocessor_path is empty/placeholder; falling back to env PREPROCESSOR_URL")
        preprocessor_path = os.getenv("PREPROCESSOR_URL")

    def _download_if_url(path: str, target_dir: str = "/opt/ml/models"):
        if not path or not isinstance(path, str):
            return path
        if not path.startswith(("http://", "https://")):
            return path
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(path.split("?")[0])
        local_path = os.path.join(target_dir, f"remote_{filename}")
        if not os.path.exists(local_path):
            logger.info(f"Downloading remote artifact: {path}")
            resp = requests.get(path, timeout=120)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Saved to {local_path}")
        else:
            logger.info(f"Using cached artifact at {local_path}")
        return local_path

    try:
        if not model_path:
            raise ValueError("MODEL path could not be resolved from args or env")
        if model_path in PLACEHOLDERS:
            raise ValueError(f"MODEL path is unresolved placeholder: {model_path}")

        model_path = _download_if_url(model_path)
        if preprocessor_path:
            if preprocessor_path in PLACEHOLDERS:
                logger.warning(f"Ignoring unresolved preprocessor placeholder: {preprocessor_path}")
                preprocessor_path = None
            else:
                preprocessor_path = _download_if_url(preprocessor_path)

        logger.info(f"Resolved model_path: {model_path}")
        logger.info(f"Resolved preprocessor_path: {preprocessor_path}")

        model_predictor = load_model_for_prediction(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        logger.info(f"Model loaded successfully from {model_path}")
        if preprocessor_path:
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        return True
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
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
        "server": app_server,
        "model_env": os.getenv("MODEL_URL"),
        "preprocessor_env": os.getenv("PREPROCESSOR_URL"),
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
    
    return info
