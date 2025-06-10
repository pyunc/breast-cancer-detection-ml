from locust import HttpUser, task, between
import json
import random

class MLApiUser(HttpUser):
    wait_time = between(1, 3)  # Wait between 1-3 seconds between tasks
    
    @task
    def predict_cancer(self):
        # Base feature set
        features = [17.99 + 0.1, 10.38 - 0.05, 122.8, 1001, 0.1184, 0.2776, 
                        0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
                        0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        
        # Add some randomness
        features = [f + random.uniform(-0.1, 0.1) * f for f in features]
        
        # Randomly choose preprocessing options
        apply_scaling = random.choice([True, False])
        apply_preprocessing = random.choice([True, False])
        
        payload = {
            "features": features,
            "apply_scaling": apply_scaling,
            "apply_preprocessing": apply_preprocessing
        }
        
        # Send POST request to the predict endpoint
        with self.client.post("/predict", 
                             json=payload, 
                             catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # You can add validation logic here
                    if "prediction" in result:
                        response.success()
                    else:
                        response.failure("Response missing prediction field")
                except json.JSONDecodeError:
                    response.failure("Response not valid JSON")
            else:
                response.failure(f"Request failed with status code: {response.status_code}")

# locust -f massive_load_test.py --host=http://159.65.210.82:8000                