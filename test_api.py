import requests
import time

def test_health_endpoint():
    print("Testing health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_root_endpoint():
    print("\nTesting root endpoint...")
    try:
        response = requests.get("http://127.0.0.1:8000/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Give the server a moment to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    test_health_endpoint()
    test_root_endpoint()
