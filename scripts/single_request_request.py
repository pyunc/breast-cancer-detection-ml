#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single request script for testing the breast cancer detection ML service.
This module provides functionality to send prediction requests to the ML API.
Now supports multiprocessing for concurrent load testing.
"""

import requests
import json
import time
import random
from typing import Dict, Any, Optional, List, Tuple
from multiprocessing import Pool, Manager, Value
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def make_prediction_request(
    host: str = "localhost",
    port: int = 8000,
    features: Optional[List[float]] = None,
    request_id: int = None,
    verbose: bool = False
) -> Tuple[int, Dict[str, Any], float]:
    """
    Make a prediction request to the ML service endpoint.
    
    Args:
        host (str): The host where the ML service is running (default: localhost)
        port (int): The port where the ML service is running (default: 8000)
        features (List[float], optional): The feature values for prediction.
                                         If None, uses sample data with slight randomization.
        request_id (int): Unique identifier for this request
        verbose (bool): Whether to print detailed request information
    
    Returns:
        Tuple[int, Dict[str, Any], float]: (request_id, response, response_time)
    
    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    # Default sample data if no features provided with slight randomization
    if features is None:
        base_features = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 
                        0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                        0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        # Add slight randomization to features for more realistic testing
        features = [f + random.uniform(-0.1, 0.1) for f in base_features]
    
    url = f"http://{host}:{port}/predict"
    
    # Wrap features in the expected request format
    request_data = {"features": features}
    
    start_time = time.time()
    
    try:
        if verbose:
            print(f"[Request {request_id}] Making prediction request to {url}")
        
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response_time = time.time() - start_time
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        
        if verbose:
            print(f"[Request {request_id}] Response received in {response_time:.3f}s: {json.dumps(result, indent=2)}")
        
        return request_id, result, response_time
        
    except requests.exceptions.ConnectionError as e:
        if verbose:
            print(f"[Request {request_id}] Error: Could not connect to ML service at {url}")
        raise
    except requests.exceptions.Timeout as e:
        if verbose:
            print(f"[Request {request_id}] Error: Request timed out")
        raise
    except requests.exceptions.HTTPError as e:
        if verbose:
            print(f"[Request {request_id}] Error: HTTP {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"[Request {request_id}] Error making request: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[Request {request_id}] Error: Invalid JSON response from server")
        raise

def make_concurrent_requests(
    host: str,
    port: int,
    num_requests: int,
    num_workers: int,
    features: Optional[List[float]] = None,
    delay_between_requests: float = 0.0
) -> Dict[str, Any]:
    """
    Make multiple concurrent requests using ThreadPoolExecutor.
    
    Args:
        host (str): ML service host
        port (int): ML service port  
        num_requests (int): Total number of requests to make
        num_workers (int): Number of concurrent workers
        features (List[float], optional): Feature values for prediction
        delay_between_requests (float): Delay between request batches
        
    Returns:
        Dict[str, Any]: Statistics about the requests
    """
    print(f"\n🚀 Starting concurrent load test:")
    print(f"   • Target: {host}:{port}")
    print(f"   • Total requests: {num_requests}")
    print(f"   • Concurrent workers: {num_workers}")
    print(f"   • Delay between requests: {delay_between_requests}s")
    print("="*60)
    
    results = []
    errors = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all requests
        future_to_id = {
            executor.submit(
                make_prediction_request, 
                host, port, features, i, False
            ): i for i in range(num_requests)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_id):
            completed += 1
            request_id = future_to_id[future]
            
            try:
                req_id, result, response_time = future.result()
                results.append({
                    'request_id': req_id,
                    'response_time': response_time,
                    'prediction': result.get('prediction'),
                    'predicted_class': result.get('predicted_class'),
                    'confidence': result.get('confidence'),
                    'status': result.get('status', 'Unknown')
                })
                
                # Progress indicator
                if completed % max(1, num_requests // 10) == 0:
                    progress = (completed / num_requests) * 100
                    print(f"Progress: {completed}/{num_requests} ({progress:.1f}%) - Latest: {response_time:.3f}s")
                    
            except Exception as e:
                errors.append({
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if results:
        response_times = [r['response_time'] for r in results]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Predictions distribution
        predictions = [r['prediction'] for r in results if r['prediction'] is not None]
        malignant_count = sum(1 for p in predictions if p == 1)
        benign_count = sum(1 for p in predictions if p == 0)
    else:
        avg_response_time = min_response_time = max_response_time = 0
        malignant_count = benign_count = 0
    
    stats = {
        'total_requests': num_requests,
        'successful_requests': len(results),
        'failed_requests': len(errors),
        'success_rate': (len(results) / num_requests) * 100 if num_requests > 0 else 0,
        'total_time': total_time,
        'requests_per_second': num_requests / total_time if total_time > 0 else 0,
        'avg_response_time': avg_response_time,
        'min_response_time': min_response_time,
        'max_response_time': max_response_time,
        'malignant_predictions': malignant_count,
        'benign_predictions': benign_count,
        'errors': errors[:10]  # Show first 10 errors
    }
    
    return stats

def print_load_test_results(stats: Dict[str, Any]) -> None:
    """Print formatted load test results."""
    print("\n" + "="*60)
    print("🎯 LOAD TEST RESULTS")
    print("="*60)
    print(f"📊 Request Statistics:")
    print(f"   • Total Requests: {stats['total_requests']}")
    print(f"   • Successful: {stats['successful_requests']}")
    print(f"   • Failed: {stats['failed_requests']}")
    print(f"   • Success Rate: {stats['success_rate']:.2f}%")
    print()
    print(f"⏱️  Performance Metrics:")
    print(f"   • Total Time: {stats['total_time']:.2f}s")
    print(f"   • Requests/Second: {stats['requests_per_second']:.2f}")
    print(f"   • Avg Response Time: {stats['avg_response_time']:.3f}s")
    print(f"   • Min Response Time: {stats['min_response_time']:.3f}s")
    print(f"   • Max Response Time: {stats['max_response_time']:.3f}s")
    print()
    print(f"🔬 Prediction Results:")
    print(f"   • Malignant Predictions: {stats['malignant_predictions']}")
    print(f"   • Benign Predictions: {stats['benign_predictions']}")
    
    if stats['errors']:
        print(f"\n❌ Errors (showing first 10):")
        for error in stats['errors']:
            print(f"   • Request {error['request_id']}: {error['error_type']} - {error['error']}")
    
    print("="*60)

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
    
    parser = argparse.ArgumentParser(description='Make prediction requests to the ML service with optional concurrent/load testing')
    parser.add_argument('--host', type=str, default='localhost', help='ML service host')
    parser.add_argument('--port', type=int, default=8000, help='ML service port')
    parser.add_argument('--features-file', type=str, help='JSON file containing feature values')
    parser.add_argument('--check-api', action='store_true', help='Check API documentation and schema')
    
    # Load testing options
    parser.add_argument('--concurrent', action='store_true', help='Enable concurrent/load testing mode')
    parser.add_argument('--requests', type=int, default=500000, help='Number of requests to make (concurrent mode)')
    parser.add_argument('--workers', type=int, default=10, help='Number of concurrent workers (concurrent mode)')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between request batches in seconds')
    
    # Legacy single request mode options
    parser.add_argument('--single-requests', type=int, default=5, help='Number of sequential requests (single mode)')
    parser.add_argument('--sleep-time', type=float, default=0.3, help='Sleep time between requests (single mode)')
    
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
        if args.concurrent:
            # Concurrent/Load testing mode
            print("🔥 CONCURRENT LOAD TESTING MODE")
            stats = make_concurrent_requests(
                host=args.host,
                port=args.port,
                num_requests=args.requests,
                num_workers=args.workers,
                features=features,
                delay_between_requests=args.delay
            )
            print_load_test_results(stats)
            
        else:
            # Single sequential request mode (legacy)
            print("📝 SEQUENTIAL REQUEST MODE")
            print(f"Making {args.single_requests} sequential requests...")
            
            for i in range(args.single_requests):
                try:
                    request_id, result, response_time = make_prediction_request(
                        host=args.host,
                        port=args.port,
                        features=features,
                        request_id=i+1,
                        verbose=True
                    )
                    
                    print("\n" + "="*50)
                    print(f"PREDICTION RESULT #{i+1}:")
                    print("="*50)
                    print(f"Response Time: {response_time:.3f}s")
                    if 'prediction' in result:
                        prediction = result['prediction']
                        print(f"Prediction: {'MALIGNANT' if prediction == 1 else 'BENIGN'}")
                    if 'confidence' in result:
                        confidence = result['confidence']
                        print(f"Confidence: {confidence:.4f}")
                    if 'probabilities' in result:
                        probs = result['probabilities']
                        print(f"Probabilities: [Benign: {probs[0]:.4f}, Malignant: {probs[1]:.4f}]")
                    print("="*50)
                    
                    if i < args.single_requests - 1:  # Don't sleep after last request
                        time.sleep(args.sleep_time)
                        
                except Exception as e:
                    print(f"Request {i+1} failed: {str(e)}")
                    continue
        
    except KeyboardInterrupt:
        print("\n🛑 Load test interrupted by user")
        exit(0)
    except Exception as e:
        print(f"❌ Failed to execute requests: {str(e)}")
        exit(1)