class DesignPattern:

    def __init__(self, base):
        
        self.base = base

    def __call__(self, exponential):

        self.exponential = exponential

        result = self.execute()

        self.base = result

        return self

    def execute(self):
        executed = self.base ** self.exponential
        return executed


dp_object = DesignPattern(base = 2)

dp_object = dp_object(exponential = 2)

dp_object = dp_object(exponential = 4)

dp_object = dp_object(exponential = 4)

print(dp_object)

import requests

path = 'https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/models/29-0cb1adcc63c10a4b5b571bf5dbe221edf4e0c82d/logistic_regression.joblib'

response = requests.get(path, timeout=120)
print(response.status_code)
# convert resposnse content into joblib file
with open('logistic_regression.joblib', 'wb') as f:
    f.write(response.content)