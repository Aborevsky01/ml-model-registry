import time
import requests

REGISTRY_URL = "http://127.0.0.1:8000"
model_name = "mnist-cnn"

def main():
    t0 = time.time()
    r1 = requests.get(f"{REGISTRY_URL}/models/{model_name}/latest", params={"stage": "PRODUCTION"})
    t1 = time.time()

    r2 = requests.get(f"{REGISTRY_URL}/models/{model_name}/latest", params={"stage": "PRODUCTION"})
    t2 = time.time()

    print("First request:", t1 - t0, "s")
    print("Second request:", t2 - t1, "s")
    print("Equal responses:", r1.json() == r2.json())

if __name__ == "__main__":
    main()
