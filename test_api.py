import requests
import os
import sys
import json

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    response = requests.get(f"{base_url}/")
    print(f"Health endpoint status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Message: {result.get('message')}")
        print(f"Data: {result.get('data')}")
        return result.get('success', False)
    else:
        print(f"Error: {response.text}")
        return False

def test_audio_analysis_url(base_url, audio_url):
    """Test the audio analysis endpoint with URL to audio file"""
    endpoint = f"{base_url}/api/v1/audioanalysis/"
    
    payload = {"url": audio_url}
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending URL request to {endpoint}...")
    print(f"URL: {audio_url}")
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")
            
            if result.get('success') and 'data' in result:
                data = result['data']
                print(f"Data keys: {list(data.keys())}")
                
                if 'used_threshold' in data:
                    print(f"Used LUFS threshold: {data['used_threshold']}")
                
                if 'transcription' in data:
                    transcription = data['transcription']
                    print(f"Transcription keys: {list(transcription.keys())}")
                
                return True
            else:
                print("No data in response")
                return False
        else:
            result = response.json()
            print(f"Error: {result.get('message', response.text)}")
            return False
    except Exception as e:
        print(f"Error during request: {e}")
        return False

if __name__ == "__main__":
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Testing API at {base_url}")
    
    # Test health endpoint
    health_ok = test_health_endpoint(base_url)
    print(f"Health check {'passed' if health_ok else 'failed'}")
    
    if health_ok:
        # Test URL-based analysis if URL provided
        if len(sys.argv) > 2:
            audio_url = sys.argv[2]
            print("\n=== Testing audio analysis with URL ===")
            analysis_ok = test_audio_analysis_url(base_url, audio_url)
            print(f"URL-based test {'passed' if analysis_ok else 'failed'}")
        else:
            print("Skipping audio analysis test - no URL provided")
            print("Usage: python test_api.py [base_url] [audio_url]")
            print("Example: python test_api.py http://localhost:8000 https://example.com/audio.mp3") 