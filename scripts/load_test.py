import subprocess
import json
import time
import concurrent.futures

def make_prediction_request(url, features=None, apply_scaling=False, apply_preprocessing=True):
    """
    Send a POST request to the prediction endpoint using curl
    """
    if features is None:
        features = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                   1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 
                   0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    
    payload = {
        "features": features,
        "apply_scaling": apply_scaling,
        "apply_preprocessing": apply_preprocessing
    }
    
    cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

def run_concurrent_requests(url, num_requests=10, max_workers=5):
    """
    Make multiple requests concurrently
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the different request parameters
        futures = []
        for i in range(num_requests):
            # Create slightly varying feature sets
            features = [17.99 + i*0.1, 10.38 - i*0.05, 122.8, 1001 + i, 0.1184, 0.2776, 
                        0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                        0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
            
            # Vary the preprocessing parameters
            apply_scaling = bool(i % 2)
            apply_preprocessing = bool((i + 1) % 2)
            
            # Submit the request to the thread pool
            future = executor.submit(
                make_prediction_request,
                url=url,
                features=features,
                apply_scaling=False,
                apply_preprocessing=True
            )
            futures.append((i+1, future))
        
        # Process results as they complete
        for req_id, future in futures:
            stdout, stderr = future.result()
            print(f"Request {req_id} Response: {stdout}")
            if stderr:
                print(f"Request {req_id} Error: {stderr}")


# argparse function for url and number of requests and max workers
import argparse
import sys

def parse_args():

    parser = argparse.ArgumentParser(description="Load test the prediction endpoint.")
    
    parser.add_argument('--url', type=str, default='http://localhost:8501/predict',
                        help='URL of the prediction endpoint')
    parser.add_argument('--num_requests', type=int, default=10000,
                        help='Total number of requests to send')
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Maximum number of concurrent requests')
    
    args = parser.parse_args()
    
    if args.num_requests <= 0 or args.max_workers <= 0:
        print("Number of requests and max workers must be positive integers.")
        sys.exit(1)
    
    return args

def main():

    args = parse_args()

    # Use the parsed arguments

    url = args.url
    num_requests = args.num_requests
    max_workers = args.max_workers

    # Define how many concurrent requests and max worker threads
    # num_requests = 10000  # Total number of requests
    # max_workers = 20   # Maximum number of concurrent requests
    # url = 'http://localhost:3000/predict'
    
    print(f"Starting load test with {num_requests} requests using {max_workers} concurrent workers")
    start_time = time.time()

    # url = 'http://164.90.241.255:8000/predict'
    
    run_concurrent_requests(url, num_requests, max_workers)
    
    elapsed = time.time() - start_time
    print(f"Load test completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()