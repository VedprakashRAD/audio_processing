import requests
import os
import sys

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

def test_audio_analysis(base_url, audio_file_path):
    """Test the audio analysis endpoint"""
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return False
    
    endpoint = f"{base_url}/api/v1/audioanalysis/"
    
    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        params = {"lufs_threshold_value": 18.0}
        
        print(f"Sending request to {endpoint}...")
        try:
            response = requests.post(endpoint, files=files, params=params)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result.get('success')}")
                print(f"Message: {result.get('message')}")
                
                if result.get('success') and 'data' in result:
                    data = result['data']
                    print(f"Data keys: {list(data.keys())}")
                    
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
    
    health_ok = test_health_endpoint(base_url)
    print(f"Health check {'passed' if health_ok else 'failed'}")
    
    if health_ok and len(sys.argv) > 2:
        audio_file_path = sys.argv[2]
        analysis_ok = test_audio_analysis(base_url, audio_file_path)
        print(f"Audio analysis test {'passed' if analysis_ok else 'failed'}")
    elif health_ok:
        print("Skipping audio analysis test - no audio file provided")
        print("Usage: python test_api.py [base_url] [audio_file_path]")
        print("Example: python test_api.py http://localhost:8000 test1.mp3") 