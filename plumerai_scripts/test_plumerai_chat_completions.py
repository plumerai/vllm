import requests
import msgpack


def test_plumerai_chat_completions_vs_regular():
    """Compare the new msgpack endpoint with the regular JSON endpoint."""
    
    request_data = {
        "model": "google/gemma-3-12b-it", 
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }
    
    # Regular JSON endpoint
    json_headers = {"Content-Type": "application/json", "Authorization": "Bearer plumerserve"}
    json_url = "http://localhost:8000/v1/chat/completions"
    
    # Msgpack endpoint
    msgpack_data = msgpack.packb(request_data)
    msgpack_headers = {"Content-Type": "application/msgpack", "Authorization": "Bearer plumerserve"}
    msgpack_url = "http://localhost:8000/v1/plumerai/chat/completions"
    
    # Test regular endpoint
    json_response = requests.post(
        json_url, json=request_data, headers=json_headers)
    
    # Test msgpack endpoint
    msgpack_response = requests.post(
        msgpack_url, data=msgpack_data, headers=msgpack_headers)
    
    # Both endpoints should return 200
    assert json_response.status_code == 200, f"JSON endpoint failed: {json_response.text}"
    assert msgpack_response.status_code == 200, f"Msgpack endpoint failed: {msgpack_response.text}"
    
    # Compare response structures
    json_result = json_response.json()
    msgpack_result = msgpack_response.json()
    
    # Both responses should have the same structure
    assert set(json_result.keys()) == set(msgpack_result.keys())
    assert json_result["model"] == msgpack_result["model"]
    assert json_result["object"] == msgpack_result["object"]


if __name__ == "__main__":
    test_plumerai_chat_completions_vs_regular()
    print("Test passed: plumerai chat completions match regular endpoint.")