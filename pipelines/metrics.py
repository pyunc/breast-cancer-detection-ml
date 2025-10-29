"""
Prometheus metrics module for ML API monitoring.

This module contains all Prometheus metrics definitions and middleware
for monitoring the ML API performance and health.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Request metrics
request_count = Counter(
    'ml_api_requests_total', 
    'Total number of requests', 
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'ml_api_request_duration_seconds', 
    'Request duration in seconds', 
    ['method', 'endpoint']
)

# ML-specific metrics
prediction_count = Counter(
    'ml_api_predictions_total', 
    'Total number of predictions', 
    ['prediction_class']
)

model_load_time = Gauge(
    'ml_api_model_load_time_seconds', 
    'Time taken to load the model'
)

# System metrics
active_connections = Gauge(
    'ml_api_active_connections', 
    'Number of active connections'
)

error_count = Counter(
    'ml_api_errors_total', 
    'Total number of errors', 
    ['error_type']
)

# Additional ML metrics
prediction_confidence = Histogram(
    'ml_api_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

model_inference_time = Histogram(
    'ml_api_model_inference_duration_seconds',
    'Time taken for model inference',
    ['model_type']
)

# =============================================================================
# Metrics Utilities
# =============================================================================

def record_prediction_metrics(prediction_class: str, confidence: float = None, inference_time: float = None):
    """Record metrics for a prediction."""
    prediction_count.labels(prediction_class=prediction_class).inc()
    
    if confidence is not None:
        prediction_confidence.observe(confidence)
    
    if inference_time is not None:
        model_inference_time.labels(model_type="logistic_regression").observe(inference_time)

def record_error_metrics(error_type: str):
    """Record error metrics."""
    error_count.labels(error_type=error_type).inc()

def record_model_load_time(load_time: float):
    """Record model loading time."""
    model_load_time.set(load_time)

def get_metrics_response():
    """Get Prometheus metrics in the correct format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =============================================================================
# FastAPI Middleware
# =============================================================================

class PrometheusMiddleware:
    """FastAPI middleware for Prometheus metrics collection."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        start_time = time.time()
        active_connections.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            process_time = time.time() - start_time
            request_duration.labels(
                method=request.method, 
                endpoint=request.url.path
            ).observe(process_time)
            
            request_count.labels(
                method=request.method, 
                endpoint=request.url.path, 
                status=response.status_code
            ).inc()
            
            return response
            
        except Exception as e:
            # Record error metrics
            error_count.labels(error_type=type(e).__name__).inc()
            logger.error(f"Request failed: {str(e)}")
            raise
            
        finally:
            active_connections.dec()

# =============================================================================
# Metrics Middleware Factory Function
# =============================================================================

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    FastAPI middleware function for Prometheus metrics collection.
    This can be used with app.middleware("http") decorator.
    """
    start_time = time.time()
    active_connections.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        process_time = time.time() - start_time
        request_duration.labels(
            method=request.method, 
            endpoint=request.url.path
        ).observe(process_time)
        
        request_count.labels(
            method=request.method, 
            endpoint=request.url.path, 
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        # Record error metrics
        error_count.labels(error_type=type(e).__name__).inc()
        logger.error(f"Request failed: {str(e)}")
        raise
        
    finally:
        active_connections.dec()

# =============================================================================
# Health Check Metrics
# =============================================================================

def get_health_metrics():
    """Get current metrics for health checking."""
    return {
        "total_requests": sum([
            float(sample.value) for sample in request_count.collect()[0].samples
        ]),
        "active_connections": active_connections._value._value,
        "total_errors": sum([
            float(sample.value) for sample in error_count.collect()[0].samples
        ]),
        "model_loaded": model_load_time._value._value > 0
    }

def get_comprehensive_request_metrics():
    """
    Get comprehensive request metrics showing detailed status breakdown.
    
    Returns:
        Dict containing:
        - Total requests by status code
        - Success rate
        - Error breakdown
        - Performance metrics
        - Prediction statistics
    """
    try:
        # Collect request metrics by status
        request_samples = request_count.collect()[0].samples
        
        total_requests = 0
        success_requests = 0
        failed_requests = 0
        status_breakdown = {}
        endpoint_breakdown = {}
        
        for sample in request_samples:
            labels = sample.labels
            count = int(float(sample.value))
            
            # Extract labels
            method = labels.get('method', 'Unknown')
            endpoint = labels.get('endpoint', 'Unknown') 
            status = labels.get('status', 'Unknown')
            
            total_requests += count
            
            # Status code breakdown
            if status not in status_breakdown:
                status_breakdown[status] = 0
            status_breakdown[status] += count
            
            # Endpoint breakdown
            endpoint_key = f"{method} {endpoint}"
            if endpoint_key not in endpoint_breakdown:
                endpoint_breakdown[endpoint_key] = {'total': 0, 'statuses': {}}
            endpoint_breakdown[endpoint_key]['total'] += count
            if status not in endpoint_breakdown[endpoint_key]['statuses']:
                endpoint_breakdown[endpoint_key]['statuses'][status] = 0
            endpoint_breakdown[endpoint_key]['statuses'][status] += count
            
            # Success vs failed
            if status == '200':
                success_requests += count
            else:
                failed_requests += count
        
        # Calculate success rate
        success_rate = (success_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Get error metrics
        error_samples = error_count.collect()[0].samples if error_count.collect()[0].samples else []
        total_errors = sum([float(sample.value) for sample in error_samples])
        error_breakdown = {}
        
        for sample in error_samples:
            error_type = sample.labels.get('error_type', 'Unknown')
            count = int(float(sample.value))
            error_breakdown[error_type] = count
        
        # Get prediction metrics
        prediction_samples = prediction_count.collect()[0].samples if prediction_count.collect()[0].samples else []
        total_predictions = sum([float(sample.value) for sample in prediction_samples])
        prediction_breakdown = {}
        
        for sample in prediction_samples:
            pred_class = sample.labels.get('prediction_class', 'Unknown')
            count = int(float(sample.value))
            prediction_breakdown[pred_class] = count
        
        # Get performance metrics
        current_connections = active_connections._value._value
        model_load_time_val = model_load_time._value._value
        
        return {
            "timestamp": time.time(),
            "request_metrics": {
                "total_requests": total_requests,
                "successful_requests": success_requests,
                "failed_requests": failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "status_breakdown": status_breakdown,
                "endpoint_breakdown": endpoint_breakdown
            },
            "error_metrics": {
                "total_application_errors": int(total_errors),
                "error_breakdown": error_breakdown
            },
            "prediction_metrics": {
                "total_predictions": int(total_predictions),
                "prediction_breakdown": prediction_breakdown
            },
            "performance_metrics": {
                "active_connections": current_connections,
                "model_load_time_seconds": model_load_time_val,
                "model_loaded": model_load_time_val > 0
            },
            "summary": {
                "api_health": "Excellent" if success_rate == 100 and total_errors == 0 else 
                             "Good" if success_rate >= 99 else "Needs Attention",
                "all_requests_answered": success_rate == 100 and total_errors == 0,
                "total_interactions": total_requests + int(total_errors)
            }
        }
        
    except Exception as e:
        logger.error(f"Error collecting comprehensive metrics: {str(e)}")
        return {
            "error": f"Failed to collect metrics: {str(e)}",
            "timestamp": time.time()
        }