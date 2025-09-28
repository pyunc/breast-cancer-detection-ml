#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single request script for testing the breast cancer detection ML service.
This module provides functionality to send prediction requests to the ML API.
"""

import requests
import json
from typing import Dict, Any, Optional

def make_prediction_request(
    host: str = "localhost",
    port: int = 8000,
    features: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Make a prediction request to the ML service endpoint.
    
    Args:
        host (str): The host where the ML service is running (default: localhost)
        port (int): The port where the ML service is running (default: 8000)
        features (Dict[str, float], optional): The feature values for prediction.
                                             If None, uses sample data.
    
    Returns:
        Dict[str, Any]: The response from the ML service
    
    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    # Default sample data if no features provided
    if features is None:
        features = [17.99 + 0.1, 10.38 - 0.05, 122.8, 1001, 0.1184, 0.2776, 
                        0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                        0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    
    url = f"http://{host}:{port}/predict"
    
    # Wrap features in the expected request format
    request_data = {"features": features}
    
    try:
        print(f"Making prediction request to {url}")
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        print(f"Response received: {json.dumps(result, indent=2)}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to ML service at {url}")
        print("Make sure the ML service is running.")
        raise
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        raise
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from server")
        raise

def check_api_docs(host: str = "localhost", port: int = 8000) -> None:
    """
    Check the API documentation to understand the expected request format.
    
    Args:
        host (str): The host where the ML service is running
        port (int): The port where the ML service is running
    """
    docs_url = f"http://{host}:{port}/docs"
    openapi_url = f"http://{host}:{port}/openapi.json"
    
    print(f"API Documentation available at: {docs_url}")
    print(f"OpenAPI schema available at: {openapi_url}")
    
    try:
        response = requests.get(openapi_url, timeout=10)
        if response.status_code == 200:
            schema = response.json()
            print("\nAPI Schema Information:")
            if 'paths' in schema and '/predict' in schema['paths']:
                predict_schema = schema['paths']['/predict']
                print(f"Available methods: {list(predict_schema.keys())}")
                if 'post' in predict_schema:
                    post_info = predict_schema['post']
                    print(f"Request schema: {json.dumps(post_info.get('requestBody', {}), indent=2)}")
        else:
            print(f"Could not fetch API schema (HTTP {response.status_code})")
    except Exception as e:
        print(f"Error fetching API schema: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make a prediction request to the ML service')
    parser.add_argument('--host', type=str, default='localhost', help='ML service host')
    parser.add_argument('--port', type=int, default=8000, help='ML service port')
    parser.add_argument('--features-file', type=str, help='JSON file containing feature values')
    parser.add_argument('--check-api', action='store_true', help='Check API documentation and schema')
    
    args = parser.parse_args()
    
    # Check API documentation if requested
    if args.check_api:
        print("Checking API documentation...")
        check_api_docs(args.host, args.port)
        print("\n")
    
    features = None
    if args.features_file:
        try:
            with open(args.features_file, 'r') as f:
                features = json.load(f)
            print(f"Loaded features from {args.features_file}")
        except FileNotFoundError:
            print(f"Error: Features file {args.features_file} not found")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in features file {args.features_file}")
            exit(1)
    
    try:
        result = make_prediction_request(
            host=args.host,
            port=args.port,
            features=features
        )
        
        print("\n" + "="*50)
        print("PREDICTION RESULT:")
        print("="*50)
        if 'prediction' in result:
            prediction = result['prediction']
            print(f"Prediction: {'MALIGNANT' if prediction == 1 else 'BENIGN'}")
        if 'probability' in result:
            prob = result['probability']
            print(f"Confidence: {prob:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"Failed to make prediction request: {str(e)}")
        exit(1)